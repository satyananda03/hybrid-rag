"""Entity consolidation (Fase 3): merge duplicates that are already stored.

Not to be confused with ``entity_resolution`` (Fase 1 + 2), which runs *during
the ingest of one document* and rewrites a name **before** it is written.
Consolidation runs **after the fact**, on demand, over the whole workspace, and
repairs what resolution could not prevent:

- two documents ingested at the same time, each unaware of the other;
- documents ingested without the LLM step (Fase 2 skipped).

How: ``EntityConsolidator`` takes the entities already in the graph and reuses
the Fase 2 judge (``EntityAdjudicator``: candidate pairs → rule vetoes → LLM →
one survivor per group). Each absorbed node is then folded into its survivor
everywhere it lives: the graph, the entity and relation vector indexes, the
entity/relation → chunk links, each document's own entity/relation lists, and
the alias table.

Entry point: ``await consolidate_entities(workspace, dry_run=True)``. The default
only returns a report of the planned merges; pass ``dry_run=False`` to apply.
Run it while the workspace is not ingesting: graph locks are per process, so a run
refuses to start (or to apply) while documents are pending or in progress.

Design: docs/entity-deduplication/07-fase-3-consolidation.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from hybrid_rag.core.base import DocStatus
from hybrid_rag.core.constants import DB_SCHEMA_ENTITY_TYPES, GRAPH_FIELD_SEP
from hybrid_rag.core.source_ids import make_relation_chunk_key
from hybrid_rag.core.utils import compute_mdhash_id
from hybrid_rag.ingestion.knowledge_graph.entity_adjudication import (
    DESCRIPTION_CHARS,
    EntityAdjudicator,
    PairVerdictStore,
    Profile,
)
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import (
    _MULTIMODAL_TYPES,
    REGISTRY_DIR,
    DomainRegistry,
    load_entity_registry,
    match_key,
    type_key,
)
from hybrid_rag.storage.shared import get_storage_keyed_lock

logger = logging.getLogger(__name__)

MAX_PAIRS = 2000  # per run; the verdict store makes the next run start where this one stopped
# Database-schema entities are unique by construction (fully qualified names), so
# they have no duplicates to fold; merging two look-alike columns would corrupt them.
_UNMERGEABLE_TYPES = _MULTIMODAL_TYPES | DB_SCHEMA_ENTITY_TYPES
_IN_FLIGHT = [DocStatus.PENDING, DocStatus.PARSING, DocStatus.PROCESSING]

@dataclass(frozen=True)
class PlannedMerge:
    survivor: str
    absorbed: str
    confidence: float
    reason: str

@dataclass
class ConsolidationReport:
    workspace: str
    entities: int = 0
    pairs_judged: int = 0
    llm_calls: int = 0
    cached_verdicts: int = 0
    merges: list[PlannedMerge] = field(default_factory=list)
    applied: bool = False


class EntityConsolidator:
    """Fase 3 for the workspace of ``rag`` (an ``IngestionRAG``); every store is that workspace's."""

    def __init__(self, rag, chain=None):
        self.rag = rag
        self.adjudicator = EntityAdjudicator(
            graph=rag.chunk_entity_relation_graph,
            chain=chain,
            verdicts=PairVerdictStore(rag.workspace, rag.entity_aliases.db),
            max_pairs=MAX_PAIRS,
        )

    async def run(self, dry_run: bool = True) -> ConsolidationReport:
        """Plan the merges; apply them unless ``dry_run``."""
        await self._refuse_while_ingesting()
        report = ConsolidationReport(self.rag.workspace)
        profiles, rank = await self._profiles()
        report.entities = len(profiles)
        vectors = await self.rag.entities_vdb.get_vectors_by_ids(
            [compute_mdhash_id(name, prefix="ent-") for name in profiles]
        )
        merges = await self.adjudicator.adjudicate(
            profiles, known=[], domain=self._never_merge(),
            vectors={name: vectors.get(compute_mdhash_id(name, prefix="ent-")) for name in profiles},
            rank=rank,
        )
        report.pairs_judged = self.adjudicator.judged
        report.cached_verdicts = self.adjudicator.cached
        report.llm_calls = self.adjudicator.calls
        report.merges = sorted(
            (PlannedMerge(m.target, absorbed, m.confidence, m.reason) for absorbed, m in merges.items()),
            key=lambda m: (m.survivor, m.absorbed),
        )
        if dry_run or not report.merges:
            return report

        await self._refuse_while_ingesting()
        for planned in report.merges:
            await self.merge(planned.survivor, planned.absorbed)
        await self.rag._insert_done()
        report.applied = True
        logger.info(f"[{self.rag.workspace}] consolidation merged {len(report.merges)} entities")
        return report

    async def merge(self, survivor: str, absorbed: str, *, create_survivor: bool = False) -> None:
        """Fold ``absorbed`` into ``survivor`` in every store (flushed by the caller).

        Reuses the ingest merge for the node and each relation, and the purge
        removal for what is left of ``absorbed``; then repoints the per-document
        indexes and the alias registry. ``create_survivor`` allows a ``survivor``
        that does not exist yet, which makes this a rename (``graph_review.py``).
        """
        from hybrid_rag.ingestion.knowledge_graph.extraction import (
            _merge_edges_then_upsert,
            _merge_nodes_then_upsert,
        )

        rag, graph = self.rag, self.rag.chunk_entity_relation_graph
        edges = await graph.get_node_edges(absorbed) or []
        neighbours = {b if a == absorbed else a for a, b in edges} - {absorbed, survivor}
        namespace = f"{rag.workspace}:GraphDB" if rag.workspace else "GraphDB"
        async with get_storage_keyed_lock(sorted({survivor, absorbed, *neighbours}), namespace=namespace):
            nodes = await graph.get_nodes_batch([survivor, absorbed])
            if absorbed not in nodes or (survivor not in nodes and not create_survivor):
                logger.warning(f"consolidation: {absorbed!r} -> {survivor!r} skipped, a node is gone")
                return
            config = rag._build_global_config()
            node = nodes[absorbed]
            chunk_ids = await self._chunk_ids(rag.entity_chunks, absorbed, node.get("source_id"))
            record = {
                "entity_name": survivor,
                "entity_type": node.get("entity_type") or "UNKNOWN",
                "description": node.get("description") or "",
                "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
                "file_path": node.get("file_path") or "unknown_source",
                "timestamp": int(time.time()),
                "raw_name": GRAPH_FIELD_SEP.join([absorbed, *filter(None, (node.get("aliases") or "").split(GRAPH_FIELD_SEP))]),
            }
            await _merge_nodes_then_upsert(
                survivor, [record], graph, rag.entities_vdb, config,
                llm_response_cache=rag.llm_response_cache, entity_chunks_storage=rag.entity_chunks,
            )

            pairs = [{"src": absorbed, "tgt": n} for n in sorted(neighbours)]
            edge_props = await graph.get_edges_batch(pairs) if pairs else {}
            relation_chunks = []
            for neighbour in sorted(neighbours):
                props = edge_props.get((absorbed, neighbour)) or edge_props.get((neighbour, absorbed)) or {}
                ids = await self._chunk_ids(
                    rag.relation_chunks, make_relation_chunk_key(absorbed, neighbour), props.get("source_id")
                )
                relation_chunks += ids
                await _merge_edges_then_upsert(
                    survivor, neighbour, self._edge_records(props, absorbed, survivor, neighbour, ids),
                    graph, rag.relationships_vdb, rag.entities_vdb, config,
                    None, None, rag.llm_response_cache, [], rag.relation_chunks, rag.entity_chunks,
                )

            await rag._remove_entities({absorbed})
            await self._repoint_documents(absorbed, survivor, chunk_ids + relation_chunks)
            await rag.entity_aliases.repoint(absorbed, survivor)
        logger.info(f"consolidation: merged {absorbed!r} into {survivor!r}")

    # --- helpers --------------------------------------------------------------------

    async def _refuse_while_ingesting(self) -> None:
        busy = await self.rag.doc_status.get_docs_by_statuses(_IN_FLIGHT)
        if busy:
            raise RuntimeError(
                f"workspace {self.rag.workspace!r} has documents being ingested "
                f"({', '.join(sorted(busy)[:5])}{' ...' if len(busy) > 5 else ''}); consolidate when idle"
            )

    async def _profiles(self) -> tuple[dict[str, Profile], dict[str, tuple[int, int]]]:
        """Every resolvable stored entity, and its rank (source chunks, then relations)."""
        graph = self.rag.chunk_entity_relation_graph
        nodes = [n for n in await graph.get_all_nodes()
                 if n.get("id") and match_key(n["id"])
                 and type_key(n.get("entity_type", "")) not in _UNMERGEABLE_TYPES]
        names = [n["id"] for n in nodes]
        chunks = await self.rag.entity_chunks.get_by_ids(names) if names else []
        degrees = await graph.node_degrees_batch(names) if names else {}
        profiles, rank = {}, {}
        for node, row in zip(nodes, chunks):
            name = node["id"]
            aliases = tuple(a for a in (node.get("aliases") or "").split(GRAPH_FIELD_SEP) if a)
            profiles[name] = Profile(name, type_key(node.get("entity_type", "")),
                                     (node.get("description") or "")[:DESCRIPTION_CHARS], aliases)
            sources = (row or {}).get("chunk_ids") or (node.get("source_id") or "").split(GRAPH_FIELD_SEP)
            rank[name] = (len([s for s in sources if s]), degrees.get(name, 0))
        return profiles, rank

    @staticmethod
    def _never_merge() -> DomainRegistry:
        """never_merge of every YAML registry: documents of one workspace may use different ones."""
        pairs = frozenset().union(*(
            load_entity_registry(path.stem).never_merge for path in sorted(REGISTRY_DIR.glob("*.yaml"))
        ))
        return DomainRegistry("all", {}, pairs)

    @staticmethod
    async def _chunk_ids(storage, key: str, fallback: str | None) -> list[str]:
        """The full chunk list (the graph's ``source_id`` may be truncated to a limit)."""
        row = await storage.get_by_id(key) if storage else None
        ids = (row or {}).get("chunk_ids") or (fallback or "").split(GRAPH_FIELD_SEP)
        return [i for i in ids if i]

    @staticmethod
    def _edge_records(props: dict, absorbed: str, survivor: str, neighbour: str, chunk_ids: list[str]) -> list[dict]:
        """The absorbed edge as ingest records; one per ontology triple so its types and
        directions survive (extra records weigh 0, so the weight is counted once)."""
        base = {
            "description": props.get("description") or "",
            "keywords": props.get("keywords") or "",
            "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
            "file_path": props.get("file_path") or "unknown_source",
            "timestamp": int(time.time()),
        }
        triples = [t.split("|") for t in props.get("relation_triples") or [] if t.count("|") == 2]
        if not triples:
            return [{**base, "src_id": survivor, "tgt_id": neighbour, "weight": float(props.get("weight") or 1.0)}]
        records = []
        for i, (src, relation_type, tgt) in enumerate(triples):
            src, tgt = (survivor if src == absorbed else src), (survivor if tgt == absorbed else tgt)
            records.append({**base, "src_id": src, "tgt_id": tgt, "relation_type": relation_type,
                            "weight": float(props.get("weight") or 1.0) if i == 0 else 0.0})
        return records

    async def _repoint_documents(self, absorbed: str, survivor: str, chunk_ids: list[str]) -> None:
        """Per-document entity/relation indexes: ``absorbed`` becomes ``survivor``.

        Deleting a document reads these, so a stale name here would leave a zombie.
        """
        rag = self.rag
        rows = await rag.text_chunks.get_by_ids(sorted(set(chunk_ids))) if chunk_ids else []
        doc_ids = sorted({r["full_doc_id"] for r in rows if isinstance(r, dict) and r.get("full_doc_id")})
        if not doc_ids:
            return
        entities = await rag.full_entities.get_by_ids(doc_ids)
        relations = await rag.full_relations.get_by_ids(doc_ids)
        entity_rows, relation_rows = {}, {}
        for doc_id, entry, rel in zip(doc_ids, entities, relations):
            names = (entry or {}).get("entity_names") or []
            if absorbed in names:
                renamed = list(dict.fromkeys(survivor if n == absorbed else n for n in names))
                entity_rows[doc_id] = {**entry, "entity_names": renamed, "count": len(renamed)}
            pairs = (rel or {}).get("relation_pairs") or []
            if any(absorbed in pair for pair in pairs):
                seen, renamed_pairs = set(), []
                for pair in pairs:
                    src, tgt = (survivor if n == absorbed else n for n in pair)
                    key = tuple(sorted((src, tgt)))
                    if src != tgt and key not in seen:
                        seen.add(key)
                        renamed_pairs.append([src, tgt])
                relation_rows[doc_id] = {**rel, "relation_pairs": renamed_pairs, "count": len(renamed_pairs)}
        if entity_rows:
            await rag.full_entities.upsert(entity_rows)
        if relation_rows:
            await rag.full_relations.upsert(relation_rows)


async def consolidate_entities(workspace: str, *, dry_run: bool = True) -> ConsolidationReport:
    """Merge duplicate entities already stored in ``workspace`` (see module doc).

    ``dry_run=True`` (default) only reports the planned merges. The host must have
    configured the LLM and embedding providers, as for ingest.
    """
    from hybrid_rag.ingestion import workspace_pool

    async with workspace_pool.lease(workspace) as rag:
        return await EntityConsolidator(rag).run(dry_run=dry_run)
