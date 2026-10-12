"""Neo4j graph storage: every workspace in one graph, isolated by a ``workspace`` property."""
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar
from neo4j import AsyncManagedTransaction  # type: ignore
from neo4j import exceptions as neo4jExceptions  # type: ignore
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from hybrid_rag.config import settings
from hybrid_rag.core.base import BaseGraphStorage
from hybrid_rag.core.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from hybrid_rag.core.utils import validate_workspace
from hybrid_rag.storage.neo4j.connection import get_neo4j_driver
from hybrid_rag.storage.shared import get_data_init_lock

logger = logging.getLogger(__name__)
logging.getLogger("neo4j").setLevel(logging.ERROR)

LABEL = "HybridRagEntity"
KEY_CONSTRAINT = "hybrid_rag_entity_key"
FULLTEXT_INDEX = "hybrid_rag_entity_id_fulltext"
WS_NODE = f"(n:{LABEL} {{workspace: $workspace}}) WHERE n.entity_id IS NOT NULL"
DEFAULT_EDGE = {"weight": 1.0, "source_id": None, "description": None, "keywords": None}

READ_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((
        neo4jExceptions.ServiceUnavailable,
        neo4jExceptions.TransientError,
        neo4jExceptions.SessionExpired,
        ConnectionResetError,
        OSError,
        AttributeError,
    )),
    reraise=True,
)
WRITE_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((
        neo4jExceptions.ServiceUnavailable,
        neo4jExceptions.TransientError,
        neo4jExceptions.WriteServiceUnavailable,
        neo4jExceptions.ClientError,
        neo4jExceptions.SessionExpired,
        ConnectionResetError,
        OSError,
    )),
)

async def create_schema() -> None:
    """The key constraint and full-text index, global rather than per workspace.

    Called by :mod:`hybrid_rag.db_init` only; the runtime runs no DDL. Idempotent.
    """
    async with get_neo4j_driver().session(database=settings.neo4j_database or None) as session:
        await (await session.run(
            f"CREATE CONSTRAINT {KEY_CONSTRAINT} IF NOT EXISTS "
            f"FOR (n:{LABEL}) REQUIRE (n.workspace, n.entity_id) IS UNIQUE"
        )).consume()
        await (await session.run(
            f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS "
            f"FOR (n:{LABEL}) ON EACH [n.entity_id] "
            "OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk', `fulltext.eventually_consistent`: true}}"
        )).consume()


def _node_props(node) -> dict[str, Any]:
    """Node properties as callers have always seen them: without the isolation key."""
    props = dict(node)
    props.pop("workspace", None)
    return props


def _edge_props(props: dict) -> dict:
    """Edge properties, with any missing ``DEFAULT_EDGE`` key appended."""
    return {**props, **{k: v for k, v in DEFAULT_EDGE.items() if k not in props}}


@dataclass
class Neo4JStorage(BaseGraphStorage):
    # Lucene query-syntax reserved characters. The full-text query parser
    # interprets these (e.g. '-' as NOT) before the analyzer runs, so a raw
    # 'tb-' becomes a NOT clause and silently matches nothing.
    _LUCENE_RESERVED: ClassVar[re.Pattern[str]] = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')
    _CJK: ClassVar[re.Pattern[str]] = re.compile(
        r"[㐀-䶿一-鿿豈-﫿]|[\U00020000-\U0002fa1f]"
    )

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=(workspace or "").strip() or "base",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        validate_workspace(self.workspace)
        self._driver = None

    async def initialize(self):
        async with get_data_init_lock():
            # One driver for every workspace (connection.py); never build one here.
            self._driver = get_neo4j_driver()
            self._DATABASE = settings.neo4j_database or None

    async def finalize(self):
        """Drop this instance's reference to the shared driver.

        The driver itself is closed once at shutdown by
        ``hybrid_rag.storage.neo4j.connection.close_neo4j_driver``: one finished
        instance must not cut the connection of every other workspace.
        """
        self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.finalize()

    async def index_done_callback(self) -> None:
        pass  # Neo4j persists every transaction itself

    # ── Query helpers ──────────────────────────────────────────────────────
    async def _read(self, cypher: str, **params) -> list:
        """Run a read query with ``$workspace`` bound to this instance's workspace."""
        params.setdefault("workspace", self.workspace)
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            result = await session.run(cypher, params)
            return [record async for record in result]

    async def _write(self, cypher: str, **params) -> None:
        """Run a write query in a managed (retried-by-driver) transaction."""
        params.setdefault("workspace", self.workspace)

        async def work(tx: AsyncManagedTransaction):
            await (await tx.run(cypher, params)).consume()

        async with self._driver.session(database=self._DATABASE) as session:
            await session.execute_write(work)

    # ── Reads ──────────────────────────────────────────────────────────────
    @READ_RETRY
    async def has_node(self, node_id: str) -> bool:
        records = await self._read(
            f"MATCH (n:{LABEL} {{workspace: $workspace, entity_id: $entity_id}}) RETURN count(n) > 0 AS node_exists",
            entity_id=node_id,
        )
        return records[0]["node_exists"]

    @READ_RETRY
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        records = await self._read(
            f"MATCH (a:{LABEL} {{workspace: $workspace, entity_id: $src}})"
            f"-[r]-(b:{LABEL} {{workspace: $workspace, entity_id: $tgt}}) RETURN count(r) > 0 AS edge_exists",
            src=source_node_id, tgt=target_node_id,
        )
        return records[0]["edge_exists"]

    @READ_RETRY
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        records = await self._read(
            f"MATCH (n:{LABEL} {{workspace: $workspace, entity_id: $entity_id}}) RETURN n", entity_id=node_id
        )
        return _node_props(records[0]["n"]) if records else None

    @READ_RETRY
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        records = await self._read(
            f"UNWIND $node_ids AS id MATCH (n:{LABEL} {{workspace: $workspace, entity_id: id}}) "
            "RETURN n.entity_id AS entity_id, n",
            node_ids=node_ids,
        )
        return {r["entity_id"]: _node_props(r["n"]) for r in records}

    @READ_RETRY
    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        records = await self._read(
            f"UNWIND $node_ids AS id MATCH (n:{LABEL} {{workspace: $workspace, entity_id: id}}) "
            "RETURN n.entity_id AS entity_id, count { (n)--() } AS degree",
            node_ids=node_ids,
        )
        degrees = {r["entity_id"]: r["degree"] for r in records}
        for nid in node_ids:
            if nid not in degrees:
                logger.warning(f"[{self.workspace}] No node found with label '{nid}'")
                degrees[nid] = 0
        return degrees

    @READ_RETRY
    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        degrees = await self.node_degrees_batch(list({n for pair in edge_pairs for n in pair}))
        return {(src, tgt): degrees.get(src, 0) + degrees.get(tgt, 0) for src, tgt in edge_pairs}

    @READ_RETRY
    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        records = await self._read(
            f"MATCH (a:{LABEL} {{workspace: $workspace, entity_id: $src}})"
            f"-[r]-(b:{LABEL} {{workspace: $workspace, entity_id: $tgt}}) RETURN properties(r) AS p LIMIT 1",
            src=source_node_id, tgt=target_node_id,
        )
        if not records:
            return None
        props = dict(records[0]["p"])
        for key in DEFAULT_EDGE.keys() - props.keys():
            logger.warning(
                f"[{self.workspace}] Edge between {source_node_id} and {target_node_id} "
                f"missing {key}, using default: {DEFAULT_EDGE[key]}"
            )
        return _edge_props(props)

    @READ_RETRY
    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        records = await self._read(
            f"UNWIND $pairs AS pair "
            f"MATCH (a:{LABEL} {{workspace: $workspace, entity_id: pair.src}})"
            f"-[r:DIRECTED]-(b:{LABEL} {{workspace: $workspace, entity_id: pair.tgt}}) "
            "RETURN pair.src AS src, pair.tgt AS tgt, collect(properties(r)) AS edges",
            pairs=pairs,
        )
        return {(r["src"], r["tgt"]): _edge_props(r["edges"][0]) for r in records if r["edges"]}

    @READ_RETRY
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        records = await self._read(
            f"MATCH (n:{LABEL} {{workspace: $workspace, entity_id: $entity_id}}) "
            f"OPTIONAL MATCH (n)-[r]-(m:{LABEL} {{workspace: $workspace}}) WHERE m.entity_id IS NOT NULL "
            "RETURN n.entity_id AS src, m.entity_id AS tgt",
            entity_id=source_node_id,
        )
        return [(r["src"], r["tgt"]) for r in records if r["src"] and r["tgt"]]

    @READ_RETRY
    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Each node's edges, oriented as stored: (node, other) outgoing, (other, node) incoming."""
        records = await self._read(
            f"UNWIND $node_ids AS id "
            f"MATCH (n:{LABEL} {{workspace: $workspace, entity_id: id}}) "
            f"OPTIONAL MATCH (n)-[r]-(m:{LABEL} {{workspace: $workspace}}) "
            "RETURN id AS queried, n.entity_id AS node, m.entity_id AS other, startNode(r).entity_id AS start",
            node_ids=node_ids,
        )
        edges: dict[str, list[tuple[str, str]]] = {nid: [] for nid in node_ids}
        for r in records:
            if r["node"] and r["other"]:
                edges[r["queried"]].append(
                    (r["node"], r["other"]) if r["start"] == r["node"] else (r["other"], r["node"])
                )
        return edges

    # ── Writes ─────────────────────────────────────────────────────────────
    # `n.workspace = $workspace` after `+=`: a stray "workspace" key in the data can
    # never move a node to another workspace.
    @WRITE_RETRY
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        if "entity_id" not in node_data:
            raise ValueError("Neo4j: node properties must contain an 'entity_id' field")
        await self._write(
            f"MERGE (n:{LABEL} {{workspace: $workspace, entity_id: $entity_id}}) "
            "SET n += $properties, n.workspace = $workspace",
            entity_id=node_id, properties=node_data,
        )

    @WRITE_RETRY
    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        if not nodes:
            return
        if any("entity_id" not in data for _, data in nodes):
            raise ValueError("Neo4j: node properties must contain an 'entity_id' field")
        await self._write(
            f"UNWIND $nodes AS row "
            f"MERGE (n:{LABEL} {{workspace: $workspace, entity_id: row.entity_id}}) "
            "SET n += row.props, n.workspace = $workspace",
            nodes=[{"entity_id": node_id, "props": data} for node_id, data in nodes],
        )

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        await self.upsert_edges_batch([(source_node_id, target_node_id, edge_data)])  # retried there

    @WRITE_RETRY
    async def upsert_edges_batch(self, edges: list[tuple[str, str, dict[str, str]]]) -> None:
        if not edges:
            return
        await self._write(
            f"UNWIND $edges AS row "
            f"MATCH (source:{LABEL} {{workspace: $workspace, entity_id: row.src}}) "
            f"WITH source, row "
            f"MATCH (target:{LABEL} {{workspace: $workspace, entity_id: row.tgt}}) "
            "MERGE (source)-[r:DIRECTED]-(target) SET r += row.props",
            edges=[{"src": src, "tgt": tgt, "props": data} for src, tgt, data in edges],
        )

    @WRITE_RETRY
    async def remove_nodes(self, nodes: list[str]):
        """Delete the nodes, with their edges, in one transaction."""
        if nodes:
            await self._write(
                f"UNWIND $entity_ids AS entity_id "
                f"MATCH (n:{LABEL} {{workspace: $workspace, entity_id: entity_id}}) DETACH DELETE n",
                entity_ids=list(nodes),
            )

    @WRITE_RETRY
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete the edges, in either direction, in one transaction."""
        if edges:
            # DISTINCT: an undirected match yields a self-loop twice.
            await self._write(
                f"UNWIND $pairs AS pair "
                f"MATCH (a:{LABEL} {{workspace: $workspace, entity_id: pair[0]}})"
                f"-[r]-(b:{LABEL} {{workspace: $workspace, entity_id: pair[1]}}) "
                "WITH DISTINCT r DELETE r",
                pairs=[[s, t] for s, t in edges],
            )

    # ── Whole-workspace reads (graph API, repair script) ───────────────────
    async def get_all_nodes(self) -> list[dict]:
        records = await self._read(f"MATCH {WS_NODE} RETURN n")
        return [{**_node_props(r["n"]), "id": r["n"].get("entity_id")} for r in records]

    async def get_all_edges(self) -> list[dict]:
        records = await self._read(
            f"MATCH (a:{LABEL} {{workspace: $workspace}})-[r]-(b:{LABEL} {{workspace: $workspace}}) "
            "WHERE a.entity_id IS NOT NULL "
            "RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties"
        )
        return [{**r["properties"], "source": r["source"], "target": r["target"]} for r in records]

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Entity names by degree, most connected first."""
        records = await self._read(
            f"MATCH {WS_NODE} OPTIONAL MATCH (n)-[r]-() "
            "WITH n.entity_id AS label, count(r) AS degree "
            "ORDER BY degree DESC, label ASC LIMIT $limit RETURN label",
            limit=limit,
        )
        return [r["label"] for r in records]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Entity names matching ``query``: full-text index first, CONTAINS scan as fallback."""
        query_strip = query.strip()
        if not query_strip:
            return []
        # Strip Lucene reserved characters before the full-text parser sees them; the
        # CASE scoring below still uses the raw text.
        sanitized = " ".join(self._LUCENE_RESERVED.sub(" ", query_strip).split())
        if not sanitized:
            return []  # only reserved characters (e.g. "---"): no full-graph CONTAINS scan
        is_cjk = bool(self._CJK.search(query_strip))
        params = {"query_strip": query_strip, "query_lower": query_strip.lower(), "limit": limit}
        if is_cjk:  # no wildcard: it does not work with the CJK analyzer
            search, label, bonus = sanitized, "label", (
                "WHEN label = $query_strip THEN score + 1000 "
                "WHEN label CONTAINS $query_strip THEN score + 500"
            )
        else:
            search, label, bonus = f"{sanitized}*", "toLower(label)", (
                "WHEN toLower(label) = $query_lower THEN score + 1000 "
                "WHEN toLower(label) STARTS WITH $query_lower THEN score + 500 "
                "WHEN toLower(label) CONTAINS ' ' + $query_lower OR toLower(label) CONTAINS '_' + $query_lower "
                "THEN score + 50"
            )
        try:
            records = await self._read(
                "CALL db.index.fulltext.queryNodes($index_name, $search_query) YIELD node, score "
                "WITH node, score WHERE node.workspace = $workspace "
                "WITH node.entity_id AS label, score "
                f"WITH label, CASE {bonus} ELSE score END AS final_score "
                "RETURN label ORDER BY final_score DESC, label ASC LIMIT $limit",
                index_name=FULLTEXT_INDEX, search_query=search, **params,
            )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Full-text search failed with error: {e}. Falling back to slower, non-indexed search."
            )
            needle = "$query_strip" if is_cjk else "$query_lower"
            records = await self._read(
                f"MATCH {WS_NODE} WITH n.entity_id AS label "
                f"WHERE {label} CONTAINS {needle} "
                f"WITH label, CASE WHEN {label} = {needle} THEN 1000 "
                f"WHEN {label} STARTS WITH {needle} THEN 500 ELSE 100 - size(label) END AS score "
                "ORDER BY score DESC, label ASC LIMIT $limit RETURN label",
                **params,
            )
        return [r["label"] for r in records]

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
        workspace: str = None,
    ) -> KnowledgeGraph:
        """Connected subgraph around ``node_label`` (``"*"``: the most connected nodes).

        ``is_truncated`` is set when ``max_nodes`` (capped by ``max_graph_nodes``) cut it.
        """
        cap = self.global_config.get("max_graph_nodes", 1000)
        max_nodes = cap if max_nodes is None else min(max_nodes, cap)
        ws = (workspace or "").strip() or self.workspace
        result = KnowledgeGraph()
        try:
            if node_label == "*":
                total = (await self._read(f"MATCH {WS_NODE} RETURN count(n) AS total", workspace=ws))[0]["total"]
                if total > max_nodes:
                    result.is_truncated = True
                    logger.info(f"[{self.workspace}] Graph truncated: {total} nodes found, limited to {max_nodes}")
                records = await self._read(
                    f"MATCH {WS_NODE} OPTIONAL MATCH (n)-[r]-() "
                    "WITH n, COALESCE(count(r), 0) AS degree ORDER BY degree DESC LIMIT $max_nodes "
                    "WITH collect({node: n}) AS filtered_nodes "
                    "UNWIND filtered_nodes AS node_info "
                    "WITH collect(node_info.node) AS kept_nodes, filtered_nodes "
                    f"OPTIONAL MATCH (a:{LABEL} {{workspace: $workspace}})-[r]-(b) "
                    "WHERE a IN kept_nodes AND b IN kept_nodes "
                    "RETURN filtered_nodes AS node_info, collect(DISTINCT r) AS relationships",
                    workspace=ws, max_nodes=max_nodes,
                )
            else:
                subgraph = (
                    f"MATCH (start:{LABEL} {{workspace: $workspace, entity_id: $entity_id}}) "
                    "CALL apoc.path.subgraphAll(start, {relationshipFilter: '', labelFilter: $label_filter, "
                    "minLevel: 0, maxLevel: $max_depth, bfs: true__LIMIT__}) YIELD nodes, relationships "
                )
                params = {"workspace": ws, "entity_id": node_label, "max_depth": max_depth, "label_filter": LABEL}
                records = await self._read(
                    subgraph.replace("__LIMIT__", "")
                    + "UNWIND nodes AS node WITH collect({node: node}) AS node_info, relationships, size(nodes) AS total "
                    "RETURN node_info, relationships, total",
                    **params,
                )
                if not records:
                    return result
                if records[0]["total"] > max_nodes:
                    result.is_truncated = True
                    logger.info(
                        f"[{self.workspace}] Graph truncated: {records[0]['total']} nodes found, "
                        f"breadth-first search limited to {max_nodes}"
                    )
                    records = await self._read(
                        subgraph.replace("__LIMIT__", ", limit: $max_nodes")
                        + "UNWIND nodes AS node WITH collect({node: node}) AS node_info, relationships "
                        "RETURN node_info, relationships",
                        max_nodes=max_nodes, **params,
                    )
        except neo4jExceptions.ClientError as e:
            logger.warning(f"[{self.workspace}] APOC plugin error: {e}")
            if node_label == "*":
                return result
            logger.warning(f"[{self.workspace}] Neo4j: falling back to basic Cypher recursive search...")
            return await self._bfs_subgraph(node_label, max_depth, max_nodes)

        if records:
            seen_nodes, seen_edges = set(), set()
            for info in records[0]["node_info"]:
                node = info["node"]
                if node.id not in seen_nodes:
                    seen_nodes.add(node.id)
                    result.nodes.append(KnowledgeGraphNode(
                        id=f"{node.id}", labels=[node.get("entity_id")], properties=_node_props(node)
                    ))
            for rel in records[0]["relationships"]:
                if rel.id not in seen_edges:
                    seen_edges.add(rel.id)
                    result.edges.append(KnowledgeGraphEdge(
                        id=f"{rel.id}", type=rel.type, source=f"{rel.start_node.id}",
                        target=f"{rel.end_node.id}", properties=dict(rel),
                    ))
            logger.info(
                f"[{self.workspace}] Subgraph query successful | Node count: {len(result.nodes)} | "
                f"Edge count: {len(result.edges)}"
            )
        return result

    async def _bfs_subgraph(self, node_label: str, max_depth: int, max_nodes: int) -> KnowledgeGraph:
        """``get_knowledge_graph`` without APOC: breadth-first search in plain Cypher."""
        result = KnowledgeGraph()
        records = await self._read(
            f"MATCH (n:{LABEL} {{workspace: $workspace, entity_id: $entity_id}}) RETURN n", entity_id=node_label
        )
        if not records:
            return result
        start = records[0]["n"]
        queue = deque([(KnowledgeGraphNode(
            id=f"{start.get('entity_id')}", labels=[start.get("entity_id")], properties=_node_props(start)
        ), 0)])
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()
        visited_pairs: set[tuple[str, str]] = set()

        while queue and len(visited_nodes) < max_nodes:
            node, depth = queue.popleft()
            if node.id in visited_nodes or depth > max_depth:
                continue
            result.nodes.append(node)
            visited_nodes.add(node.id)
            if len(visited_nodes) >= max_nodes:
                result.is_truncated = True
                logger.info(f"[{self.workspace}] Graph truncated: breadth-first search limited to: {max_nodes} nodes")
                break

            neighbours = (await self._read(
                f"MATCH (a:{LABEL} {{workspace: $workspace, entity_id: $entity_id}})-[r]-(b) "
                "RETURN r, b, id(r) AS edge_id",
                entity_id=node.id,
            ))[:1000]  # max neighbour nodes handled per node
            for record in neighbours:
                edge_id = str(record["edge_id"])
                target_id = record["b"].get("entity_id")
                if edge_id in visited_edges:
                    continue
                if not target_id:
                    logger.warning(f"[{self.workspace}] Skipping edge {edge_id} due to missing entity_id on target node")
                    continue
                # (A,B) and (B,A) are one undirected edge
                pair = tuple(sorted([node.id, target_id]))
                if pair not in visited_pairs and (target_id in visited_nodes or depth < max_depth):
                    result.edges.append(KnowledgeGraphEdge(
                        id=edge_id, type=record["r"].type, source=f"{node.id}", target=f"{target_id}",
                        properties=dict(record["r"]),
                    ))
                    visited_edges.add(edge_id)
                    visited_pairs.add(pair)
                if target_id not in visited_nodes and depth < max_depth:
                    queue.append((KnowledgeGraphNode(
                        id=f"{target_id}", labels=[target_id], properties=_node_props(record["b"])
                    ), depth + 1))

        logger.info(
            f"[{self.workspace}] BFS subgraph query successful | Node count: {len(result.nodes)} | "
            f"Edge count: {len(result.edges)}"
        )
        return result


async def list_workspaces() -> list[str]:
    """Workspaces that own at least one entity, sorted by name."""
    async with get_neo4j_driver().session(
        database=settings.neo4j_database or None, default_access_mode="READ"
    ) as session:
        result = await session.run(
            f"MATCH (n:{LABEL}) WHERE n.entity_id IS NOT NULL "
            "RETURN DISTINCT n.workspace AS workspace ORDER BY workspace"
        )
        return [record["workspace"] async for record in result]
