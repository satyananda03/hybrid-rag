"""Batched graph merge: read everything once, merge in memory, write once.

``merge_nodes_and_edges`` used to call Neo4j and the chunk-tracking KV per entity
and per relation — ±6 sequential round-trips per relation, serialised further by
hub entities whose keyed lock every one of their relations needs. This module gives
the unchanged per-item merge functions (``_merge_nodes_then_upsert`` /
``_merge_edges_then_upsert``) storage stand-ins that answer reads from one prefetched
snapshot plus their own pending writes, then flushes every write with the storage's
batch API. The merge logic is untouched; only the I/O pattern changes.

The caller must hold the graph keyed lock for every name in the snapshot for the
whole session — that is what keeps the snapshot current (see ``merge_nodes_and_edges``).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable

from hybrid_rag.core.utils import make_relation_chunk_key


def _edge_key(src: str, tgt: str) -> tuple[str, str]:
    return (src, tgt) if src <= tgt else (tgt, src)


class GraphOverlay:
    """The subset of ``BaseGraphStorage`` the merge functions call, served from memory.

    Writes mimic Neo4j's ``SET n += props`` (merged onto what is there) so a read after
    a write returns what the database would. Graph edges are undirected here exactly
    as in the Cypher (``-[r:DIRECTED]-``); the first upsert's direction is kept for
    edges that do not exist yet, which is the direction ``MERGE`` would create.
    """

    def __init__(self, nodes: dict[str, dict], edges: dict[tuple[str, str], dict]):
        self._nodes = {k: dict(v) for k, v in nodes.items()}
        self._edges = {k: dict(v) for k, v in edges.items()}
        self._edge_dir: dict[tuple[str, str], tuple[str, str]] = {}
        # dicts as ordered sets: rows are written in the order they were merged
        self._dirty_nodes: dict[str, None] = {}
        self._dirty_edges: dict[tuple[str, str], None] = {}

    async def get_node(self, node_id: str) -> dict | None:
        node = self._nodes.get(node_id)
        return dict(node) if node is not None else None

    async def has_edge(self, src: str, tgt: str) -> bool:
        return _edge_key(src, tgt) in self._edges

    async def get_edge(self, src: str, tgt: str) -> dict | None:
        edge = self._edges.get(_edge_key(src, tgt))
        return dict(edge) if edge is not None else None

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        self._nodes[node_id] = {**self._nodes.get(node_id, {}), **node_data}
        self._dirty_nodes[node_id] = None

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None:
        key = _edge_key(src, tgt)
        self._edge_dir.setdefault(key, (src, tgt))
        self._edges[key] = {**self._edges.get(key, {}), **edge_data}
        self._dirty_edges[key] = None

    def pending_nodes(self) -> list[tuple[str, dict]]:
        return [(n, self._nodes[n]) for n in self._dirty_nodes]

    def pending_edges(self) -> list[tuple[str, str, dict]]:
        return [
            (*self._edge_dir.get(k, k), self._edges[k]) for k in self._dirty_edges
        ]


class KVOverlay:
    """``get_by_id`` / ``upsert`` over a prefetched snapshot of a chunk-tracking KV."""

    def __init__(self, rows: dict[str, dict | None]):
        self._rows = {k: v for k, v in rows.items() if v is not None}
        self._written: dict[str, dict] = {}

    async def get_by_id(self, key: str) -> dict | None:
        row = self._written.get(key, self._rows.get(key))
        return dict(row) if row is not None else None

    async def upsert(self, data: dict[str, dict]) -> None:
        for key, value in data.items():
            self._written[key] = dict(value)

    def pending(self) -> dict[str, dict]:
        return self._written


class MergeSession:
    """One document's merge: prefetch → in-memory merge → one batched flush."""

    def __init__(self, graph, entity_chunks, relation_chunks):
        self._graph = graph
        self._entity_chunks = entity_chunks
        self._relation_chunks = relation_chunks
        self.graph: GraphOverlay | None = None
        self.entity_chunks: KVOverlay | None = None
        self.relation_chunks: KVOverlay | None = None
        self._locks: dict[str, asyncio.Lock] = {}

    async def prefetch(
        self, node_names: Iterable[str], edge_pairs: Iterable[tuple[str, str]]
    ) -> None:
        pairs = list(dict.fromkeys(_edge_key(s, t) for s, t in edge_pairs if s != t))
        # Relation endpoints too: a node may exist without being extracted in this
        # document (the multimodal side writes "<name> (table)" before the merge and
        # the merge meets it only as a belongs_to endpoint). Missing from the snapshot
        # it would be recreated as UNKNOWN and overwrite the real node on flush.
        names = list(dict.fromkeys([*node_names, *(n for pair in pairs for n in pair)]))
        rel_keys = [make_relation_chunk_key(s, t) for s, t in pairs]

        async def kv_rows(storage, keys):
            if storage is None or not keys:
                return {}
            return dict(zip(keys, await storage.get_by_ids(keys)))

        nodes, edges, ent_rows, rel_rows = await asyncio.gather(
            self._graph.get_nodes_batch(names) if names else _const({}),
            self._graph.get_edges_batch([{"src": s, "tgt": t} for s, t in pairs])
            if pairs
            else _const({}),
            kv_rows(self._entity_chunks, names),
            kv_rows(self._relation_chunks, rel_keys),
        )
        self.graph = GraphOverlay(
            nodes, {_edge_key(s, t): props for (s, t), props in edges.items()}
        )
        self.entity_chunks = KVOverlay(ent_rows)
        self.relation_chunks = KVOverlay(rel_rows)

    async def flush(self) -> None:
        """Write every pending change: nodes before edges (edges MATCH their nodes)."""

        async def graph_writes():
            await self._graph.upsert_nodes_batch(self.graph.pending_nodes())
            await self._graph.upsert_edges_batch(self.graph.pending_edges())

        writes = [graph_writes()]
        if self._entity_chunks is not None and self.entity_chunks.pending():
            writes.append(self._entity_chunks.upsert(self.entity_chunks.pending()))
        if self._relation_chunks is not None and self.relation_chunks.pending():
            writes.append(self._relation_chunks.upsert(self.relation_chunks.pending()))
        await asyncio.gather(*writes)

    @asynccontextmanager
    async def lock(self, keys: list[str]) -> AsyncIterator[None]:
        """In-process keyed lock for tasks of this one merge.

        Cross-document exclusion comes from the storage keyed lock the caller holds;
        this only stops two tasks of the same document from interleaving on a shared
        entity (a hub) while one of them awaits an LLM summary.
        """
        held = [self._locks.setdefault(k, asyncio.Lock()) for k in sorted(set(keys))]
        acquired = []
        try:
            for lock in held:
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()


async def _const(value: Any) -> Any:
    return value
