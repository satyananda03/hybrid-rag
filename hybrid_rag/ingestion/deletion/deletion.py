
"""Document and graph deletion — mixed into :class:`IngestionRAG`.

Deleting a document is the inverse of ingesting one and touches every store:
chunk rows, the chunks/entities/relations vector tables, the Neo4j nodes and
edges, the LLM extraction cache, and doc status. Entities and relations that
other documents still reference are not dropped but *rebuilt* from the
remaining chunks (``rebuild_knowledge_from_chunks``), which is what makes this
long.

``self`` here is the :class:`IngestionRAG` instance — the storages are its fields.
"""


import asyncio
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hybrid_rag.core.base import DeletionResult, DocStatus
from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.core.utils import (
    compute_mdhash_id,
    make_relation_chunk_key,
    normalize_string_list,
    subtract_source_ids,
)
import logging
from hybrid_rag import progress

from hybrid_rag.ingestion.knowledge_graph.extraction import (
    rebuild_knowledge_from_chunks,
)

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


class _DeletionMixin:
    """Delete by document, by entity, or by relation."""

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

    async def _update_delete_retry_state(
        self,
        doc_id: str,
        doc_status_data: dict[str, Any],
        *,
        deletion_stage: str,
        doc_llm_cache_ids: list[str],
        error_message: str | None = None,
        failed: bool,
    ) -> dict[str, Any]:
        """Persist deletion retry metadata and return the updated status record."""
        metadata = doc_status_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        backup_cache_ids = normalize_string_list(
            metadata.get("deletion_llm_cache_ids", []),
            context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
        )
        retry_cache_ids = doc_llm_cache_ids or backup_cache_ids

        updated_metadata = dict(metadata)
        if retry_cache_ids:
            updated_metadata["deletion_llm_cache_ids"] = retry_cache_ids
        updated_metadata["last_deletion_attempt_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        if failed:
            updated_metadata["deletion_failed"] = True
            updated_metadata["deletion_failure_stage"] = deletion_stage
        else:
            updated_metadata.pop("deletion_failed", None)
            updated_metadata.pop("deletion_failure_stage", None)

        updated_status_data = {
            **doc_status_data,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": updated_metadata,
            "error_msg": error_message if failed else "",
        }

        await self.doc_status.upsert({doc_id: updated_status_data})
        return updated_status_data

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

        # ---- 1–2. Classify affected entities/relations, update chunk tracking ----
        try:
            impact = await self._analyze_graph_impact(doc_id, chunk_ids)
        except Exception as e:
            logger.error(
                f"[purge] Failed to analyze affected graph elements for {doc_id}: {e}"
            )
            raise Exception(f"Failed to analyze graph dependencies: {e}") from e
        try:
            await self._update_chunk_tracking(impact)
        except Exception as e:
            logger.error(
                f"[purge] Failed to process graph analysis results for {doc_id}: {e}"
            )
            raise Exception(f"Failed to process graph dependencies: {e}") from e

        # ---- 3. Delete chunks themselves ----
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
            try:
                residual_edges = await self._remove_entities(impact.entities_to_delete)
                if residual_edges:
                    logger.info(
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
        try:
            await self._insert_done()
        except Exception as e:
            logger.error(f"[purge] Failed to persist pre-rebuild changes: {e}")
            raise Exception(f"Failed to persist pre-rebuild changes: {e}") from e

        # ---- 7. Rebuild entities/relations that still have remaining sources ----
        if impact.entities_to_rebuild or impact.relationships_to_rebuild:
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
                )
            except Exception as e:
                logger.error(f"[purge] Failed to rebuild knowledge from chunks: {e}")
                raise Exception(f"Failed to rebuild knowledge graph: {e}") from e

        # ---- 8. Delete per-doc full_entities / full_relations index rows ----
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

    async def adelete_by_doc_id(
        self, doc_id: str, delete_llm_cache: bool = False
    ) -> DeletionResult:
        """Delete a document and all its related data, including chunks, graph elements.

        This method orchestrates a comprehensive deletion process for a given document ID.
        It ensures that not only the document itself but also all its derived and associated
        data across different storage layers are removed or rebuiled. If entities or relationships
        are partially affected, they will be rebuilded using LLM cached from remaining documents.

        **Concurrency:** serializing callers is not this method's job. The app's
        routes take the workspace's write turn (``workspace_write_turn``) around
        every ingest and delete, so one workspace never runs both at once.

        The upstream ``pipeline_status["busy"]`` handshake that used to sit
        here guarded nothing and was removed: it is per-process state, and the
        batch-deletion job it coordinated with (``job_name`` starting with
        ``"deleting"``) was never ported into this codebase, so its "join the
        deletion queue" branch could not be reached from any call path.

        Args:
            doc_id (str): The unique identifier of the document to be deleted.
            delete_llm_cache (bool): Whether to delete cached LLM extraction results
                associated with the document. Defaults to False.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
                - `status` (str): "success", "not_found", or "fail".
                - `doc_id` (str): The ID of the document attempted to be deleted.
                - `message` (str): A summary of the operation's result.
                - `status_code` (int): HTTP status code (e.g., 200, 404, 500).
                - `file_path` (str | None): The file path of the deleted document, if available.
        """
        deletion_operations_started = False
        deletion_fully_completed = False
        in_final_delete_stage = False
        original_exception = None
        doc_llm_cache_ids: list[str] = []
        deletion_stage = "initializing"
        progress.stage("initializing")
        doc_status_data: dict[str, Any] | None = None
        file_path: str | None = None

        log_message = f"Starting deletion process for document {doc_id}"
        logger.info(log_message)

        try:
            # 1. Get the document status and related data
            doc_status_data = await self.doc_status.get_by_id(doc_id)
            if not doc_status_data:
                logger.warning(f"Document {doc_id} not found")
                return DeletionResult(
                    status="not_found",
                    doc_id=doc_id,
                    message=f"Document {doc_id} not found.",
                    status_code=404,
                    file_path="",
                )
            file_path = doc_status_data.get("file_path")

            # Check document status and log warning for non-completed documents
            raw_status = doc_status_data.get("status")
            try:
                doc_status = DocStatus(raw_status)
            except ValueError:
                doc_status = raw_status

            if doc_status != DocStatus.PROCESSED:
                if doc_status == DocStatus.PENDING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PENDING)"
                    )
                elif doc_status == DocStatus.PROCESSING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PROCESSING)"
                    )
                elif doc_status == DocStatus.PREPROCESSED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PREPROCESSED)"
                    )
                elif doc_status == DocStatus.FAILED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: FAILED)"
                    )
                else:
                    status_text = (
                        doc_status.value
                        if isinstance(doc_status, DocStatus)
                        else str(doc_status)
                    )
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: {status_text})"
                    )
                logger.info(warning_msg)

            # 2. Get chunk IDs from document status
            metadata = doc_status_data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata_cache_ids = normalize_string_list(
                metadata.get("deletion_llm_cache_ids", []),
                context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
            )
            # Order-preserving dedup so chunk_ids stays a list and satisfies the
            # storage delete contract (``delete(ids: list[str])``).
            tracked_chunk_ids = normalize_string_list(
                doc_status_data.get("chunks_list", []),
                context=f"doc {doc_id} chunks_list",
            )
            stored_chunk_ids = await self._chunk_ids_from_storage(doc_id)
            chunk_ids = list(dict.fromkeys(tracked_chunk_ids + stored_chunk_ids))
            # Parent chunks are stored on purpose (``save_parent``) but never
            # listed in chunks_list, which counts children only; they are still
            # deleted through the full_doc_id lookup, so they are not drift.
            parent_prefix = f"{doc_id}-parent-"
            tracked_set = set(tracked_chunk_ids)
            untracked = sum(
                1
                for cid in chunk_ids
                if cid not in tracked_set and not cid.startswith(parent_prefix)
            )
            if untracked > 0:
                # Worth a warning, not a debug line: it means chunks_list drifted
                # from what was actually stored, which is how chunks leak.
                logger.warning(
                    f"Doc {doc_id}: {untracked} chunk(s) found via full_doc_id "
                    f"that chunks_list did not track"
                )

            if not chunk_ids:
                logger.warning(f"No chunks found for document {doc_id}")
                # Mark that deletion operations have started
                deletion_operations_started = True

                # A prior failed deletion may have collected LLM cache IDs before the
                # chunks were removed. If delete_llm_cache is requested and persisted IDs
                # exist, clean them up now before removing the doc/status entries.
                if delete_llm_cache and metadata_cache_ids:
                    if not self.llm_response_cache:
                        no_cache_msg = (
                            f"Cannot delete LLM cache for document {doc_id}: "
                            "cache storage is unavailable"
                        )
                        logger.error(no_cache_msg)
                        raise Exception(no_cache_msg)
                    try:
                        deletion_stage = "delete_llm_cache"
                        progress.stage("delete_llm_cache")
                        await self.llm_response_cache.delete(metadata_cache_ids)
                        remaining_cache_ids = await self._get_existing_llm_cache_ids(
                            metadata_cache_ids
                        )
                        if remaining_cache_ids:
                            raise Exception(
                                f"{len(remaining_cache_ids)} LLM cache entries still exist after delete"
                            )
                        logger.info(
                            "Cleaned up %d LLM cache entries from prior attempt for document %s",
                            len(metadata_cache_ids),
                            doc_id,
                        )
                    except Exception as cache_err:
                        raise Exception(
                            f"Failed to delete LLM cache for document {doc_id}: {cache_err}"
                        ) from cache_err

                try:
                    # Still need to delete the doc status and full doc.
                    # Delete doc_status first: if full_docs.delete fails on retry, the
                    # doc_status record is already gone so the retry finds no record and
                    # treats the document as already deleted rather than creating a zombie.
                    deletion_stage = "delete_doc_entries"
                    progress.stage("delete_doc_entries")
                    await self.doc_status.delete([doc_id])
                    await self.full_docs.delete([doc_id])
                except Exception as e:
                    logger.error(
                        f"Failed to delete document {doc_id} with no chunks: {e}"
                    )
                    raise Exception(f"Failed to delete document entry: {e}") from e

                log_message = (
                    f"Document deleted without associated chunks: {doc_id}"
                )
                logger.info(log_message)

                deletion_fully_completed = True
                return DeletionResult(
                    status="success",
                    doc_id=doc_id,
                    message=log_message,
                    status_code=200,
                    file_path=file_path,
                )

            # Mark that deletion operations have started
            deletion_operations_started = True

            if chunk_ids:
                # Always collect/persist cache IDs for chunk-backed documents, even when
                # this call does not request cache deletion. If a delete fails after the
                # chunks/graph have already been removed, a later retry may turn on
                # delete_llm_cache=True, and doc_status metadata is then the only durable
                # place left to recover the cache keys for cleanup.
                deletion_stage = "collect_llm_cache"
                progress.stage("collect_llm_cache")
                doc_llm_cache_ids = list(metadata_cache_ids)
                if not self.text_chunks:
                    logger.info(
                        "Skipping LLM cache id collection for document %s because text chunk storage is unavailable",
                        doc_id,
                    )
                else:
                    try:
                        chunk_data_list = await self.text_chunks.get_by_ids(chunk_ids)
                        seen_cache_ids: set[str] = set(doc_llm_cache_ids)
                        for chunk_data in chunk_data_list:
                            if not chunk_data or not isinstance(chunk_data, dict):
                                continue
                            cache_ids = chunk_data.get("llm_cache_list", [])
                            if not isinstance(cache_ids, list):
                                continue
                            for cache_id in cache_ids:
                                if (
                                    isinstance(cache_id, str)
                                    and cache_id
                                    and cache_id not in seen_cache_ids
                                ):
                                    doc_llm_cache_ids.append(cache_id)
                                    seen_cache_ids.add(cache_id)
                    except Exception as cache_collect_error:
                        logger.error(
                            "Failed to collect LLM cache ids for document %s: %s",
                            doc_id,
                            cache_collect_error,
                        )
                        raise Exception(
                            f"Failed to collect LLM cache ids for document {doc_id}: {cache_collect_error}"
                        ) from cache_collect_error

                if doc_llm_cache_ids:
                    try:
                        doc_status_data = await self._update_delete_retry_state(
                            doc_id,
                            doc_status_data,
                            deletion_stage=deletion_stage,
                            doc_llm_cache_ids=doc_llm_cache_ids,
                            failed=False,
                        )
                    except Exception as status_write_error:
                        logger.error(
                            "Failed to persist LLM cache IDs for document %s to retry state: %s",
                            doc_id,
                            status_write_error,
                        )
                        # Describe whether this is a fresh attempt or a retry so
                        # operators can tell whether prior partial deletions exist.
                        attempt_context = (
                            "retry — prior partial deletions may exist"
                            if metadata_cache_ids
                            else "deletion not yet started"
                        )
                        raise Exception(
                            f"Failed to persist LLM cache IDs for document {doc_id} "
                            f"({attempt_context}): {status_write_error}"
                        ) from status_write_error
                    logger.info(
                        "Collected %d LLM cache entries for document %s",
                        len(doc_llm_cache_ids),
                        doc_id,
                    )
                else:
                    logger.info("No LLM cache entries found for document %s", doc_id)

            # 4. Classify affected entities/relations and update chunk tracking
            try:
                deletion_stage = "analyze_graph_dependencies"
                progress.stage("analyze_graph_dependencies")
                impact = await self._analyze_graph_impact(doc_id, chunk_ids)
            except Exception as e:
                logger.error(f"Failed to analyze affected graph elements: {e}")
                raise Exception(f"Failed to analyze graph dependencies: {e}") from e

            try:
                deletion_stage = "update_chunk_tracking"
                progress.stage("update_chunk_tracking")
                await self._update_chunk_tracking(impact)
            except Exception as e:
                logger.error(f"Failed to process graph analysis results: {e}")
                raise Exception(f"Failed to process graph dependencies: {e}") from e

            # Data integrity is ensured by allowing only one process to hold pipeline at a time（no graph db lock is needed anymore)

            # 5. Delete chunks from storage
            if chunk_ids:
                try:
                    deletion_stage = "delete_chunks"
                    progress.stage("delete_chunks")
                    await self.chunks_vdb.delete(chunk_ids)
                    await self.text_chunks.delete(chunk_ids)

                    log_message = (
                        f"Successfully deleted {len(chunk_ids)} chunks from storage"
                    )
                    logger.info(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete chunks: {e}")
                    raise Exception(f"Failed to delete document chunks: {e}") from e

            # 6. Delete relationships that have no remaining sources
            if impact.relationships_to_delete:
                try:
                    deletion_stage = "delete_relationships"
                    progress.stage("delete_relationships")
                    await self._remove_relationships(impact.relationships_to_delete)
                    logger.info(
                        f"Successfully deleted {len(impact.relationships_to_delete)} relations"
                    )
                except Exception as e:
                    logger.error(f"Failed to delete relationships: {e}")
                    raise Exception(f"Failed to delete relationships: {e}") from e

            # 7. Delete entities that have no remaining sources
            if impact.entities_to_delete:
                try:
                    deletion_stage = "delete_entities"
                    progress.stage("delete_entities")
                    residual_edges = await self._remove_entities(impact.entities_to_delete)
                    if residual_edges:
                        # Relations still attached to entities being deleted: step 6
                        # should have removed them, so this points at drift.
                        logger.warning(
                            f"⚠️ {len(residual_edges)} edge(s) still attached to deleted "
                            f"entities; cleaned from VDB and chunk-tracking storage"
                        )
                    logger.info(
                        f"Successfully deleted {len(impact.entities_to_delete)} entities"
                    )
                except Exception as e:
                    logger.error(f"Failed to delete entities: {e}")
                    raise Exception(f"Failed to delete entities: {e}") from e

            # Persist changes to graph database before entity and relationship rebuild
            # Plain _insert_done: pending DELETES must be retained for retry on
            # failure, not discarded (see _insert_done_with_cleanup docstring).
            try:
                deletion_stage = "persist_pre_rebuild_changes"
                progress.stage("persist_pre_rebuild_changes")
                await self._insert_done()
            except Exception as e:
                logger.error(f"Failed to persist pre-rebuild changes: {e}")
                raise Exception(f"Failed to persist pre-rebuild changes: {e}") from e

            # 8. Rebuild entities and relationships from remaining chunks
            if impact.entities_to_rebuild or impact.relationships_to_rebuild:
                try:
                    deletion_stage = "rebuild_knowledge_graph"
                    progress.stage("rebuild_knowledge_graph")
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
                    )

                except Exception as e:
                    logger.error(f"Failed to rebuild knowledge from chunks: {e}")
                    raise Exception(f"Failed to rebuild knowledge graph: {e}") from e

            # 9. Delete LLM cache while the document status still exists so a failure
            # remains retryable via the same doc_id.
            log_message = f"Document {doc_id} successfully deleted"
            if delete_llm_cache and doc_llm_cache_ids:
                if not self.llm_response_cache:
                    log_message = (
                        f"Cannot delete LLM cache for document {doc_id}: "
                        "cache storage is unavailable"
                    )
                    logger.error(log_message)
                    raise Exception(log_message)
                try:
                    deletion_stage = "delete_llm_cache"
                    progress.stage("delete_llm_cache")
                    await self.llm_response_cache.delete(doc_llm_cache_ids)
                    # Some storage implementations do not raise on delete errors and
                    # instead only log internally, so confirm the cache entries are
                    # actually gone before deleting the document/status records.
                    remaining_cache_ids = await self._get_existing_llm_cache_ids(
                        doc_llm_cache_ids
                    )
                    if remaining_cache_ids:
                        doc_llm_cache_ids = remaining_cache_ids
                        raise Exception(
                            f"{len(remaining_cache_ids)} LLM cache entries still exist after delete"
                        )
                    cache_log_message = f"Successfully deleted {len(doc_llm_cache_ids)} LLM cache entries for document {doc_id}"
                    logger.info(cache_log_message)
                    log_message = cache_log_message
                except Exception as cache_delete_error:
                    log_message = (
                        f"Failed to delete LLM cache for document {doc_id}: "
                        f"{cache_delete_error}"
                    )
                    logger.error(log_message)
                    logger.error(traceback.format_exc())
                    raise Exception(log_message) from cache_delete_error

            # 10. Delete from full_entities and full_relations storage
            try:
                deletion_stage = "delete_doc_graph_metadata"
                progress.stage("delete_doc_graph_metadata")
                await asyncio.gather(
                    self.full_entities.delete([doc_id]),
                    self.full_relations.delete([doc_id]),
                )
            except Exception as e:
                logger.error(f"Failed to delete from full_entities/full_relations: {e}")
                raise Exception(
                    f"Failed to delete from full_entities/full_relations: {e}"
                ) from e

            # 11. Delete original document and status.
            # doc_status is deleted first so that if full_docs.delete fails, a retry
            # finds no doc_status record and treats the document as already gone,
            # rather than finding a doc_status that points to a missing full_docs entry.
            try:
                deletion_stage = "delete_doc_entries"
                progress.stage("delete_doc_entries")
                in_final_delete_stage = True
                await self.doc_status.delete([doc_id])
                await self.full_docs.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete document and status: {e}")
                raise Exception(f"Failed to delete document and status: {e}") from e

            deletion_fully_completed = True
            return DeletionResult(
                status="success",
                doc_id=doc_id,
                message=log_message,
                status_code=200,
                file_path=file_path,
            )

        except Exception as e:
            original_exception = e
            error_message = f"Error while deleting document {doc_id}: {e}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            try:
                # Do not attempt to write retry state if doc_status was already deleted:
                # upsert would re-create the record as a zombie. All earlier stages still
                # have doc_status intact and can safely update it, even if some chunk/graph
                # data has already been removed.
                if doc_status_data is not None and not in_final_delete_stage:
                    doc_status_data = await self._update_delete_retry_state(
                        doc_id,
                        doc_status_data,
                        deletion_stage=deletion_stage,
                        doc_llm_cache_ids=doc_llm_cache_ids,
                        error_message=error_message,
                        failed=True,
                    )
            except Exception as status_update_error:
                logger.error(
                    "Failed to update deletion retry state for document %s: %s",
                    doc_id,
                    status_update_error,
                )
                logger.error(traceback.format_exc())
                error_message = (
                    f"{error_message}. Additionally, failed to persist retry state: "
                    f"{status_update_error}. Manual cleanup may be required."
                )
            return DeletionResult(
                status="fail",
                doc_id=doc_id,
                message=error_message,
                status_code=500,
                file_path=file_path,
            )

        finally:
            # ALWAYS ensure persistence if any deletion operations were started
            if deletion_operations_started:
                # Plain _insert_done: this finally reports the deletion as
                # successful after logging a persistence error, so discarding
                # pending DELETES here would drop them with no retry and leave
                # stale vectors/KV behind. Keep them buffered for a later flush.
                try:
                    await self._insert_done()
                except Exception as persistence_error:
                    persistence_error_msg = f"Failed to persist data after deletion attempt for {doc_id}: {persistence_error}"
                    logger.error(persistence_error_msg)
                    logger.error(traceback.format_exc())

                    if deletion_fully_completed:
                        # All deletion stages succeeded; the flush error is a post-cleanup
                        # concern. Do not override the success result already returned.
                        logger.error(
                            "Post-deletion persistence flush failed for %s, "
                            "but deletion completed successfully: %s",
                            doc_id,
                            persistence_error,
                        )
                    elif original_exception is None:
                        # Deletion stages were in-flight but the try-block return was never
                        # reached; treat the persistence failure as the primary error.
                        return DeletionResult(
                            status="fail",
                            doc_id=doc_id,
                            message=f"Deletion completed but failed to persist changes: {persistence_error}",
                            status_code=500,
                            file_path=file_path,
                        )
                    # If there was an original exception, log the persistence error but
                    # don't override it — the original error result was already returned.
            else:
                logger.debug(
                    f"No deletion operations were started for document {doc_id}, skipping persistence"
                )

            logger.info("Deletion process completed for document: %s", doc_id)
