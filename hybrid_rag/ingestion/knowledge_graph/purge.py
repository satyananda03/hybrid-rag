"""Purge a document's chunks and its knowledge-graph contributions — mixed into :class:`IngestionRAG`.

Shared by both writers of the graph: the pipeline's resume path wipes a
re-processed document before extracting it again, and deletion removes it for
good. Entities and relations other documents still reference are not dropped
but rebuilt from the remaining chunks (``rebuild_knowledge_from_chunks``).

``self`` here is the :class:`IngestionRAG` instance — the storages are its fields.
"""


import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.core.source_ids import make_relation_chunk_key, subtract_source_ids
from hybrid_rag.core.utils import compute_mdhash_id
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import _MULTIMODAL_TYPES, type_key
from hybrid_rag.ingestion.knowledge_graph.extraction import rebuild_knowledge_from_chunks

logger = logging.getLogger(__name__)


def _relation_vdb_ids(edges) -> list[str]:
    """Relation VDB ids for ``edges``; a relation is stored under either direction."""
    return [
        compute_mdhash_id(a + b, prefix="rel-")
        for src, tgt in edges
        for a, b in ((src, tgt), (tgt, src))
    ]


def split_source_ids(source_id: str | None) -> list[str]:
    """Chunk ids in a ``GRAPH_FIELD_SEP``-joined ``source_id``, empties dropped."""
    return [c for c in (source_id or "").split(GRAPH_FIELD_SEP) if c]


async def tracked_sources(tracking, keys: list[str]) -> dict[str, list[str]]:
    """Chunk ids per key from a chunk-tracking store, read in one query.

    Keys without a row are left out, so callers fall back to the graph's
    ``source_id`` for them.
    """
    if not tracking or not keys:
        return {}
    rows = await tracking.get_by_ids(keys)
    return {
        key: [c for chunk_id in row.get("chunk_ids") or [] for c in split_source_ids(chunk_id)]
        for key, row in zip(keys, rows)
        if row
    }


def _remaining_sources(
    tracked: list[str],
    graph_source_id: str | None,
    chunk_ids: list[str],
    chunk_ids_set: set[str],
) -> list[str] | None:
    """Sources an entity or relation keeps once ``chunk_ids`` are gone.

    ``[]`` means delete it, a non-empty list means rebuild it from those
    chunks, ``None`` means it is untouched. Chunk tracking is trusted first,
    the graph's ``source_id`` otherwise. A graph ``source_id`` that still names
    a chunk being deleted (left stale by an earlier failed delete) forces a
    rebuild even when tracking is already right, so the graph is resynced.
    """
    graph_sources = split_source_ids(graph_source_id)
    existing = tracked or graph_sources
    remaining = subtract_source_ids(existing, chunk_ids)
    if not remaining:
        return []
    if remaining != existing or set(graph_sources) & chunk_ids_set:
        return remaining
    return None


@dataclass
class _GraphImpact:
    """What removing a document's chunks does to the graph elements it touched."""

    entities_to_delete: set[str] = field(default_factory=set)
    entities_to_rebuild: dict[str, list[str]] = field(default_factory=dict)
    relationships_to_delete: set[tuple[str, str]] = field(default_factory=set)
    relationships_to_rebuild: dict[tuple[str, str], list[str]] = field(default_factory=dict)


class _PurgeMixin:
    """Remove a document's chunks and graph contributions, rebuilding what others still use."""

    async def _remove_relationships(self, edges: set[tuple[str, str]]) -> None:
        """Delete relations from the relation VDB, the graph, and chunk tracking.

        The stores are independent, so they are written concurrently.
        """
        writes = [
            self.relationships_vdb.delete(_relation_vdb_ids(edges)),
            self.chunk_entity_relation_graph.remove_edges(list(edges)),
        ]
        if self.relation_chunks:
            writes.append(
                self.relation_chunks.delete(
                    [make_relation_chunk_key(src, tgt) for src, tgt in edges]
                )
            )
        await asyncio.gather(*writes)

    async def _remove_entities(self, entities: set[str]) -> set[tuple[str, str]]:
        """Delete entities from the graph, the entity VDB, and chunk tracking.

        Edges still attached to them go with the nodes in the graph, so only
        their VDB rows and chunk tracking are cleaned here. Returns those
        residual edges. They are read before anything is deleted; the stores
        are then written concurrently.
        """
        nodes_edges = await self.chunk_entity_relation_graph.get_nodes_edges_batch(
            list(entities)
        )
        residual_edges = {
            tuple(sorted(edge)) for edges in nodes_edges.values() if edges for edge in edges
        }
        writes = [
            self.chunk_entity_relation_graph.remove_nodes(list(entities)),
            self.entities_vdb.delete(
                [compute_mdhash_id(entity, prefix="ent-") for entity in entities]
            ),
        ]
        if self.entity_chunks:
            writes.append(self.entity_chunks.delete(list(entities)))
        if residual_edges:
            writes.append(self.relationships_vdb.delete(_relation_vdb_ids(residual_edges)))
            if self.relation_chunks:
                writes.append(
                    self.relation_chunks.delete(
                        [make_relation_chunk_key(src, tgt) for src, tgt in residual_edges]
                    )
                )
        await asyncio.gather(*writes)
        return residual_edges

    async def _prune_isolated_entities(self, doc_ids: list[str]) -> set[str]:
        """Delete the entities of ``doc_ids`` that ended the batch with no relationship.

        Runs once the whole batch is merged: an entity unlinked within one chunk
        or document may still be linked by another, and a concurrent merge could
        otherwise be adding an edge to a node being deleted. Multimodal anchors
        (image/table/equation) stay — they are how figures are found. The text
        that mentioned a pruned entity stays searchable from its chunk.
        """
        rows = await self.full_entities.get_by_ids(doc_ids) if doc_ids else []
        names = sorted({n for row in rows if row for n in row.get("entity_names") or []})
        if not names:
            return set()
        graph = self.chunk_entity_relation_graph
        degrees = await graph.node_degrees_batch(names)
        unlinked = [n for n, degree in degrees.items() if degree == 0]
        nodes = await graph.get_nodes_batch(unlinked) if unlinked else {}
        isolated = {
            n for n, node in nodes.items()
            if type_key(node.get("entity_type", "")) not in _MULTIMODAL_TYPES
        }
        if not isolated:
            return set()

        await self._remove_entities(isolated)
        updated = {}
        for doc_id, row in zip(doc_ids, rows):
            if row and isolated & set(row.get("entity_names") or []):
                kept = [n for n in row["entity_names"] if n not in isolated]
                updated[doc_id] = {**row, "entity_names": kept, "count": len(kept)}
        if updated:
            await self.full_entities.upsert(updated)
        await self._insert_done()
        logger.info(f"[{self.workspace}] pruned {len(isolated)} isolated entities")
        return isolated

    async def _chunk_ids_from_storage(self, doc_id: str) -> list[str]:
        """Ask the storages which chunks carry ``doc_id``, ignoring bookkeeping.

        ``doc_status.chunks_list`` is a hand-maintained cache with two writers:
        the text pipeline overwrites it, multimodal ingestion appends to it. If
        those two interleave, ids fall out of the list — and a chunk missing
        from it would survive deletion and keep answering queries from a
        document the user deleted.

        The chunk tables record ``full_doc_id`` for text and multimodal chunks
        alike, so the database already knows the answer. Deletion unions both
        sources: the query catches what the cache lost, the cache catches
        anything buffered but not yet flushed to a table.
        """
        sources = [
            (storage, label)
            for storage, label in (
                (self.text_chunks, "text_chunks"),
                (self.chunks_vdb, "chunks_vdb"),
            )
            if storage is not None
        ]
        results = await asyncio.gather(
            *(storage.get_chunk_ids_by_doc_id(doc_id) for storage, _ in sources),
            return_exceptions=True,
        )
        found: list[str] = []
        for (_, label), result in zip(sources, results):
            if isinstance(result, Exception):
                # Never block a deletion on this: chunks_list still applies, and
                # a loud log is better than a half-deleted document.
                logger.warning(
                    f"Could not resolve chunks of {doc_id} from {label}: {result}"
                )
            elif isinstance(result, BaseException):
                raise result  # cancellation is not a lookup failure
            else:
                found.extend(result)
        return found

    async def _analyze_graph_impact(
        self, doc_id: str, chunk_ids: list[str]
    ) -> _GraphImpact:
        """Classify every entity and relation ``doc_id`` contributed to.

        ``full_entities`` / ``full_relations`` name them; each is then deleted,
        rebuilt or left alone by :func:`_remaining_sources`. Every store is read
        once in a batch — a document touches hundreds of graph elements, and one
        round trip each is what used to make deletion take minutes.
        """
        graph = self.chunk_entity_relation_graph
        doc_entities, doc_relations = await asyncio.gather(
            self.full_entities.get_by_id(doc_id),
            self.full_relations.get_by_id(doc_id),
        )
        entity_names = (doc_entities or {}).get("entity_names", [])
        relation_pairs = (doc_relations or {}).get("relation_pairs", [])
        nodes, edges = await asyncio.gather(
            graph.get_nodes_batch(entity_names),
            graph.get_edges_batch([{"src": s, "tgt": t} for s, t in relation_pairs]),
        )

        affected_nodes = [
            nodes[name]
            for name in entity_names
            if nodes.get(name) and nodes[name].get("entity_id")
        ]
        # (sorted pair, graph source_id); the edge's own source/target win
        # over the pair it was looked up by, as the graph stores them.
        affected_edges: list[tuple[tuple[str, str], str]] = []
        for src, tgt in relation_pairs:
            edge = edges.get((src, tgt))
            if not edge:
                continue
            src, tgt = edge.get("source", src), edge.get("target", tgt)
            if src and tgt and "source_id" in edge:
                affected_edges.append((tuple(sorted((src, tgt))), edge["source_id"]))

        entity_tracking, relation_tracking = await asyncio.gather(
            tracked_sources(
                self.entity_chunks,
                list(dict.fromkeys(node["entity_id"] for node in affected_nodes)),
            ),
            tracked_sources(
                self.relation_chunks,
                list(dict.fromkeys(make_relation_chunk_key(*e) for e, _ in affected_edges)),
            ),
        )

        chunk_ids_set = set(chunk_ids)
        impact = _GraphImpact()
        for node in affected_nodes:
            name = node["entity_id"]
            remaining = _remaining_sources(
                entity_tracking.get(name, []), node.get("source_id"), chunk_ids, chunk_ids_set
            )
            if remaining:
                impact.entities_to_rebuild[name] = remaining
            elif remaining is not None:
                impact.entities_to_delete.add(name)
        for edge, graph_source_id in affected_edges:
            remaining = _remaining_sources(
                relation_tracking.get(make_relation_chunk_key(*edge), []),
                graph_source_id,
                chunk_ids,
                chunk_ids_set,
            )
            if remaining:
                impact.relationships_to_rebuild[edge] = remaining
            elif remaining is not None:
                impact.relationships_to_delete.add(edge)

        logger.info(
            f"{doc_id}: entities rebuild {len(impact.entities_to_rebuild)} / delete "
            f"{len(impact.entities_to_delete)}, relations rebuild "
            f"{len(impact.relationships_to_rebuild)} / delete "
            f"{len(impact.relationships_to_delete)}"
        )
        return impact

    async def _update_chunk_tracking(self, impact: _GraphImpact) -> None:
        """Point chunk tracking of rebuilt entities/relations at their remaining chunks.

        Deleted ones are skipped: their tracking rows go with the graph elements.
        """
        now = int(time.time())

        def rows(sources: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
            return {
                key: {"chunk_ids": ids, "count": len(ids), "updated_at": now}
                for key, ids in sources.items()
            }

        writes = []
        if self.entity_chunks and impact.entities_to_rebuild:
            writes.append(self.entity_chunks.upsert(rows(impact.entities_to_rebuild)))
        if self.relation_chunks and impact.relationships_to_rebuild:
            writes.append(
                self.relation_chunks.upsert(
                    rows(
                        {
                            make_relation_chunk_key(*edge): ids
                            for edge, ids in impact.relationships_to_rebuild.items()
                        }
                    )
                )
            )
        await asyncio.gather(*writes)

    async def _purge_doc_chunks_and_kg(
        self,
        doc_id: str,
        chunk_ids: list[str],
    ) -> None:
        """Remove a document's chunks and clean up its knowledge-graph contributions.

        Used by:
            - The pipeline resume branch in ``process_document`` when a
              document whose content is already extracted is re-processed:
              chunks must be wiped and entities/relations rebuilt fresh.
            - Future deletion paths that want a focused "purge KG only"
              operation without the LLM-cache / doc_status / full_docs
              cleanup that ``adelete_by_doc_id`` also performs.

        What this method does:
            1. Reads ``full_entities`` / ``full_relations`` to identify which
               graph nodes / edges this document contributed to.
            2. For each affected entity / relation, intersects the doc's
               ``chunk_ids`` with the union of chunk-tracking entries
               (``entity_chunks`` / ``relation_chunks``) and graph
               ``source_id`` lists, then classifies it as either
               *delete-outright* (no remaining sources) or *rebuild*
               (still references chunks from other documents).
            3. Deletes the chunks themselves from ``chunks_vdb`` and
               ``text_chunks``.
            4. For *delete-outright* entries: removes the relationship /
               entity from the graph storage, vector storage, and chunk
               tracking.
            5. Calls :py:meth:`_insert_done` to persist graph changes
               before rebuilding (so the rebuild step sees a consistent
               state).
            6. Calls :func:`rebuild_knowledge_from_chunks` to rebuild any
               *rebuild* entries from their remaining chunks (so other
               documents that also contributed to the same entity /
               relation keep their data intact).
            7. Deletes the per-doc ``full_entities`` / ``full_relations``
               index rows so subsequent re-extraction starts fresh.

        Does NOT touch:
            - ``doc_status`` / ``full_docs`` records — caller manages those.
            - ``llm_response_cache`` — orthogonal to KG cleanup.
            - Any pipeline-level serialization — the caller already runs
              behind the advisory lock held in ``engine``.

        Idempotent: passing an empty ``chunk_ids`` returns immediately
        without touching storage.
        """
        if not chunk_ids:
            return
        await self._purge_graph_contributions(doc_id, chunk_ids)
        await self._drop_doc_graph_index(doc_id)

    async def _purge_graph_contributions(
        self,
        doc_id: str,
        chunk_ids: list[str],
        on_stage: Callable[[str], None] | None = None,
    ) -> None:
        """Steps 1–6 of :meth:`_purge_doc_chunks_and_kg`: everything but the per-doc index.

        Deletion runs its LLM-cache cleanup between this and
        :meth:`_drop_doc_graph_index`, so a failed delete keeps the index it
        needs to retry. ``on_stage`` is called with each step's name before
        it runs (deletion records it for retry and reports progress).
        """
        stage = on_stage or (lambda _name: None)

        # ---- 1–2. Classify affected entities/relations, update chunk tracking ----
        stage("analyze_graph_dependencies")
        try:
            impact = await self._analyze_graph_impact(doc_id, chunk_ids)
        except Exception as e:
            logger.error(
                f"[purge] Failed to analyze affected graph elements for {doc_id}: {e}"
            )
            raise Exception(f"Failed to analyze graph dependencies: {e}") from e
        stage("update_chunk_tracking")
        try:
            await self._update_chunk_tracking(impact)
        except Exception as e:
            logger.error(
                f"[purge] Failed to process graph analysis results for {doc_id}: {e}"
            )
            raise Exception(f"Failed to process graph dependencies: {e}") from e

        # ---- 3. Delete chunks themselves ----
        stage("delete_chunks")
        try:
            await self.chunks_vdb.delete(chunk_ids)
            await self.text_chunks.delete(chunk_ids)
            log_message = (
                f"[purge] {doc_id}: deleted {len(chunk_ids)} chunk(s) from storage"
            )
            logger.info(log_message)
        except Exception as e:
            logger.error(f"[purge] Failed to delete chunks for {doc_id}: {e}")
            raise Exception(f"Failed to delete document chunks: {e}") from e

        # ---- 4. Delete relationships with no remaining sources ----
        if impact.relationships_to_delete:
            stage("delete_relationships")
            try:
                await self._remove_relationships(impact.relationships_to_delete)
                logger.info(
                    f"[purge] {doc_id}: deleted {len(impact.relationships_to_delete)} relation(s)"
                )
            except Exception as e:
                logger.error(
                    f"[purge] Failed to delete relationships for {doc_id}: {e}"
                )
                raise Exception(f"Failed to delete relationships: {e}") from e

        # ---- 5. Delete entities with no remaining sources ----
        if impact.entities_to_delete:
            stage("delete_entities")
            try:
                residual_edges = await self._remove_entities(impact.entities_to_delete)
                if residual_edges:
                    # Relations still attached to entities being deleted: the
                    # relation step should have removed them, so this is drift.
                    logger.warning(
                        f"[purge] {doc_id}: cleaned {len(residual_edges)} residual "
                        f"edge(s) from VDB and chunk-tracking storage"
                    )
                logger.info(
                    f"[purge] {doc_id}: deleted {len(impact.entities_to_delete)} entity(ies)"
                )
            except Exception as e:
                logger.error(f"[purge] Failed to delete entities for {doc_id}: {e}")
                raise Exception(f"Failed to delete entities: {e}") from e

        # ---- 6. Persist pre-rebuild changes ----
        # Use plain _insert_done (no discard-on-failure): the pending buffer
        # here holds DELETES, which are not regenerable by reprocessing. On a
        # flush failure they must stay buffered for a later retry, not be
        # discarded (see _insert_done_with_cleanup docstring).
        stage("persist_pre_rebuild_changes")
        try:
            await self._insert_done()
        except Exception as e:
            logger.error(f"[purge] Failed to persist pre-rebuild changes: {e}")
            raise Exception(f"Failed to persist pre-rebuild changes: {e}") from e

        # ---- 7. Rebuild entities/relations that still have remaining sources ----
        if impact.entities_to_rebuild or impact.relationships_to_rebuild:
            stage("rebuild_knowledge_graph")
            try:
                await rebuild_knowledge_from_chunks(
                    entities_to_rebuild=impact.entities_to_rebuild,
                    relationships_to_rebuild=impact.relationships_to_rebuild,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entities_vdb=self.entities_vdb,
                    relationships_vdb=self.relationships_vdb,
                    text_chunks_storage=self.text_chunks,
                    llm_response_cache=self.llm_response_cache,
                    global_config=self._build_global_config(),
                    entity_chunks_storage=self.entity_chunks,
                    relation_chunks_storage=self.relation_chunks,
                    doc_status_storage=self.doc_status,
                    entity_resolver=self.entity_resolver,
                )
            except Exception as e:
                logger.error(f"[purge] Failed to rebuild knowledge from chunks: {e}")
                raise Exception(f"Failed to rebuild knowledge graph: {e}") from e

    async def _drop_doc_graph_index(
        self,
        doc_id: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> None:
        """Step 7 of :meth:`_purge_doc_chunks_and_kg`: drop the per-doc full_entities/full_relations rows."""
        stage = on_stage or (lambda _name: None)
        # ---- 8. Delete per-doc full_entities / full_relations index rows ----
        stage("delete_doc_graph_metadata")
        try:
            await asyncio.gather(
                self.full_entities.delete([doc_id]),
                self.full_relations.delete([doc_id]),
            )
        except Exception as e:
            logger.error(
                f"[purge] Failed to delete full_entities/full_relations rows for {doc_id}: {e}"
            )
            raise Exception(
                f"Failed to delete from full_entities/full_relations: {e}"
            ) from e
