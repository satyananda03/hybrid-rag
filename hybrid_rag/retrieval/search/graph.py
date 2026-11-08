"""Graph search: entities and relations, found by vector and expanded through the graph.

* local  — low-level keywords → entity vectors → those nodes (+ degree) → their edges,
  sorted by (degree, weight);
* global — high-level keywords → relation vectors → those edges (vector order) → their
  endpoint nodes.

Local and global results are interleaved round-robin. The result is entities and
relations, not chunks: their chunks are collected later through ``source_id``
(:func:`hybrid_rag.retrieval.context._merge_all_chunks`).
"""

import asyncio
import logging
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from hybrid_rag.core.base import BaseGraphStorage, BaseVectorStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphSearch:
    graph: BaseGraphStorage
    entities_vdb: BaseVectorStorage
    relationships_vdb: BaseVectorStorage
    top_k: int

    async def search(
        self,
        ll_keywords: str,
        hl_keywords: str,
        ll_embedding: Any = None,
        hl_embedding: Any = None,
    ) -> tuple[list[dict], list[dict]]:
        """(entities, relations) for the keywords; an empty keyword string skips its side."""
        (local_entities, local_relations), (global_relations, global_entities) = await asyncio.gather(
            self._local(ll_keywords, ll_embedding),
            self._global(hl_keywords, hl_embedding),
        )
        entities = _round_robin(local_entities, global_entities, key=lambda e: e.get("entity_name"))
        relations = _round_robin(local_relations, global_relations, key=_relation_key)
        logger.info(f"Raw graph search results: {len(entities)} entities, {len(relations)} relations")
        return entities, relations

    async def _local(self, keywords: str, embedding: Any) -> tuple[list[dict], list[dict]]:
        """Entities by vector, in vector order, and the edges around them."""
        if not keywords:
            return [], []
        logger.info(
            f"Query nodes: {keywords} (top_k:{self.top_k}, cosine:{self.entities_vdb.cosine_better_than_threshold})"
        )
        results = await self.entities_vdb.query(keywords, top_k=self.top_k, query_embedding=embedding)
        if not results:
            return [], []

        node_ids = [r["entity_name"] for r in results]
        nodes_dict, degrees_dict = await asyncio.gather(
            self.graph.get_nodes_batch(node_ids),
            self.graph.node_degrees_batch(node_ids),
        )
        node_datas = [nodes_dict.get(nid) for nid in node_ids]
        node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]
        if not all(n is not None for n in node_datas):
            logger.warning("Some nodes are missing, maybe the storage is damaged")

        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d, "created_at": k.get("created_at")}
            for k, n, d in zip(results, node_datas, node_degrees)
            if n is not None
        ]
        relations = await self._edges_around(node_datas)
        logger.info(f"Local query: {len(node_datas)} entites, {len(relations)} relations")
        return node_datas, relations

    async def _edges_around(self, node_datas: list[dict]) -> list[dict]:
        """Every edge of the nodes, once, sorted by (degree, weight) descending."""
        node_names = [dp["entity_name"] for dp in node_datas]
        batch_edges_dict = await self.graph.get_nodes_edges_batch(node_names)

        all_edges = []
        seen = set()
        for node_name in node_names:
            for e in batch_edges_dict.get(node_name, []):
                sorted_edge = tuple(sorted(e))
                if sorted_edge not in seen:
                    seen.add(sorted_edge)
                    all_edges.append(sorted_edge)

        edge_data_dict, edge_degrees_dict = await asyncio.gather(
            self.graph.get_edges_batch([{"src": e[0], "tgt": e[1]} for e in all_edges]),
            self.graph.edge_degrees_batch(all_edges),
        )

        all_edges_data = []
        for pair in all_edges:
            edge_props = edge_data_dict.get(pair)
            if edge_props is not None:
                _default_weight(pair, edge_props)
                all_edges_data.append(
                    {"src_tgt": pair, "rank": edge_degrees_dict.get(pair, 0), **edge_props}
                )
        return sorted(all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True)

    async def _global(self, keywords: str, embedding: Any) -> tuple[list[dict], list[dict]]:
        """Relations by vector, in vector order, and their endpoint entities."""
        if not keywords:
            return [], []
        logger.info(
            f"Query edges: {keywords} (top_k:{self.top_k}, cosine:{self.relationships_vdb.cosine_better_than_threshold})"
        )
        results = await self.relationships_vdb.query(keywords, top_k=self.top_k, query_embedding=embedding)
        if not results:
            return [], []

        edge_data_dict = await self.graph.get_edges_batch(
            [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
        )
        edge_datas = []
        for k in results:
            pair = (k["src_id"], k["tgt_id"])
            edge_props = edge_data_dict.get(pair)
            if edge_props is not None:
                _default_weight(pair, edge_props)
                edge_datas.append(
                    {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "created_at": k.get("created_at", None), **edge_props}
                )

        entities = await self._endpoints(edge_datas)
        logger.info(f"Global query: {len(entities)} entites, {len(edge_datas)} relations")
        return edge_datas, entities

    async def _endpoints(self, edge_datas: list[dict]) -> list[dict]:
        """The relations' endpoint entities, once each, in order of first appearance."""
        entity_names = list(dict.fromkeys(name for e in edge_datas for name in (e["src_id"], e["tgt_id"])))
        nodes_dict = await self.graph.get_nodes_batch(entity_names)

        node_datas = []
        for entity_name in entity_names:
            node = nodes_dict.get(entity_name)
            if node is None:
                logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
                continue
            node_datas.append({**node, "entity_name": entity_name})
        return node_datas


def _default_weight(pair: tuple[str, str], edge_props: dict) -> None:
    if "weight" not in edge_props:
        logger.warning(f"Edge {pair} missing 'weight' attribute, using default value 1.0")
        edge_props["weight"] = 1.0


def _relation_key(relation: dict) -> tuple[str, ...]:
    """Local relations carry ``src_tgt``, global ones ``src_id``/``tgt_id``."""
    pair = relation["src_tgt"] if "src_tgt" in relation else (relation.get("src_id"), relation.get("tgt_id"))
    return tuple(sorted(pair))


def _round_robin(first: list[dict], second: list[dict], key: Callable[[dict], Hashable]) -> list[dict]:
    """``first[0], second[0], first[1], ...``, skipping items whose key is empty or seen."""
    merged, seen = [], set()
    for i in range(max(len(first), len(second))):
        for items in (first, second):
            if i < len(items):
                k = key(items[i])
                if k and k not in seen:
                    seen.add(k)
                    merged.append(items[i])
    return merged
