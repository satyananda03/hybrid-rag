"""Characterization tests for document deletion and the resume-path purge.

Every storage call goes into one log, so these pin what a delete (or a resume
purge) does to the stores, in which order, which stages it reports, and what it
returns. Doc ``doc-1`` owns chunks c1 and c2; c3 belongs to another document.
Entity A lives only in doc-1 (deleted), B also in c3 (rebuilt), C only in c3
(untouched); relation A-B is doc-1 only (deleted), B-C also in c3 (rebuilt).
"""

import asyncio

import pytest

from hybrid_rag import progress
from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.core.source_ids import make_relation_chunk_key
from hybrid_rag.ingestion.deletion import deletion as deletion_module
from hybrid_rag.ingestion.deletion.deletion import _DeletionMixin
from hybrid_rag.ingestion.knowledge_graph import purge as purge_module
from hybrid_rag.ingestion.knowledge_graph.purge import _PurgeMixin


def _norm(value):
    """Order-insensitive view of call arguments built from sets."""
    if isinstance(value, dict):
        return {str(k): _norm(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        items = [_norm(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (list, set)) else items
    return value


class FakeKV:
    def __init__(self, name, log, rows=None, by_doc=None):
        self.name, self.log, self.rows, self.by_doc = name, log, dict(rows or {}), by_doc or {}

    async def get_by_id(self, id):
        return self.rows.get(id)

    async def get_by_ids(self, ids):
        return [self.rows.get(i) for i in ids]

    async def get_chunk_ids_by_doc_id(self, doc_id):
        return [c for c in self.by_doc.get(doc_id, []) if c in self.rows]

    async def upsert(self, data):
        self.log.append((self.name, "upsert", _norm(data)))
        self.rows.update(data)

    async def delete(self, ids):
        self.log.append((self.name, "delete", _norm(ids)))
        for i in ids:
            self.rows.pop(i, None)


class FakeGraph:
    def __init__(self, log, nodes, edges):
        self.log, self.nodes, self.edges = log, nodes, edges

    async def get_nodes_batch(self, names):
        return {n: self.nodes[n] for n in names if n in self.nodes}

    async def get_edges_batch(self, pairs):
        out = {}
        for p in pairs:
            edge = self.edges.get(tuple(sorted((p["src"], p["tgt"]))))
            if edge:
                out[(p["src"], p["tgt"])] = edge
        return out

    async def get_nodes_edges_batch(self, names):
        return {n: [e for e in self.edges if n in e] for n in names}

    async def remove_nodes(self, names):
        self.log.append(("graph", "remove_nodes", _norm(names)))
        for n in names:
            self.nodes.pop(n, None)

    async def remove_edges(self, edges):
        self.log.append(("graph", "remove_edges", _norm(edges)))
        for e in edges:
            self.edges.pop(tuple(sorted(e)), None)


class FakeRag(_PurgeMixin, _DeletionMixin):
    def __init__(self):
        self.log = []
        sep = GRAPH_FIELD_SEP
        chunks = {
            "c1": {"full_doc_id": "doc-1", "llm_cache_list": ["k1"]},
            "c2": {"full_doc_id": "doc-1", "llm_cache_list": ["k2"]},
            "c3": {"full_doc_id": "doc-2", "llm_cache_list": ["k3"]},
        }
        by_doc = {"doc-1": ["c1", "c2"], "doc-2": ["c3"]}
        self.text_chunks = FakeKV("text_chunks", self.log, chunks, by_doc)
        self.chunks_vdb = FakeKV("chunks_vdb", self.log, chunks, by_doc)
        self.entities_vdb = FakeKV("entities_vdb", self.log)
        self.relationships_vdb = FakeKV("relationships_vdb", self.log)
        self.entity_chunks = FakeKV("entity_chunks", self.log, {
            "A": {"chunk_ids": ["c1", "c2"]},
            "B": {"chunk_ids": ["c2", "c3"]},
        })
        self.relation_chunks = FakeKV("relation_chunks", self.log, {
            make_relation_chunk_key("A", "B"): {"chunk_ids": ["c2"]},
            make_relation_chunk_key("B", "C"): {"chunk_ids": ["c2", "c3"]},
        })
        self.full_entities = FakeKV("full_entities", self.log, {"doc-1": {"entity_names": ["A", "B"]}})
        self.full_relations = FakeKV("full_relations", self.log, {"doc-1": {"relation_pairs": [["A", "B"], ["B", "C"]]}})
        self.full_docs = FakeKV("full_docs", self.log, {"doc-1": {"content": "x"}})
        self.doc_status = FakeKV("doc_status", self.log, {"doc-1": {
            "status": "processed", "file_path": "a.md", "chunks_list": ["c1", "c2"], "metadata": {},
        }})
        self.llm_response_cache = FakeKV("llm_response_cache", self.log, {"k1": {}, "k2": {}, "k3": {}})
        self.chunk_entity_relation_graph = FakeGraph(
            self.log,
            nodes={
                "A": {"entity_id": "A", "source_id": f"c1{sep}c2"},
                "B": {"entity_id": "B", "source_id": f"c2{sep}c3"},
                "C": {"entity_id": "C", "source_id": "c3"},
            },
            edges={
                ("A", "B"): {"source": "A", "target": "B", "source_id": "c2"},
                ("B", "C"): {"source": "B", "target": "C", "source_id": f"c2{sep}c3"},
            },
        )

    async def _insert_done(self):
        self.log.append(("rag", "_insert_done", None))

    def _build_global_config(self):
        return {"cfg": True}


@pytest.fixture
def harness(monkeypatch):
    rag = FakeRag()
    stages, rebuilds = [], []

    async def fake_rebuild(**kwargs):
        rebuilds.append(kwargs)
        rag.log.append(("rag", "rebuild_knowledge_from_chunks", _norm({
            "entities": kwargs["entities_to_rebuild"],
            "relationships": {"|".join(k): v for k, v in kwargs["relationships_to_rebuild"].items()},
        })))
        if getattr(rag, "fail_rebuild", False):
            raise RuntimeError("boom")

    for module in (deletion_module, purge_module):
        if hasattr(module, "rebuild_knowledge_from_chunks"):
            monkeypatch.setattr(module, "rebuild_knowledge_from_chunks", fake_rebuild)
    monkeypatch.setattr(progress, "stage", stages.append)
    return rag, stages, rebuilds


AB = make_relation_chunk_key("A", "B")
BC = make_relation_chunk_key("B", "C")

# What both paths do to the graph and chunks, in order (steps 1–5 of the purge).
GRAPH_PURGE = [
    ("entity_chunks", "upsert", {"B": {"chunk_ids": ["c3"], "count": 1, "updated_at": "<t>"}}),
    ("relation_chunks", "upsert", {BC: {"chunk_ids": ["c3"], "count": 1, "updated_at": "<t>"}}),
    ("chunks_vdb", "delete", ["c1", "c2"]),
    ("text_chunks", "delete", ["c1", "c2"]),
    ("relationships_vdb", "delete", "<rel ids>"),
    ("graph", "remove_edges", [["A", "B"]]),
    ("relation_chunks", "delete", [AB]),
    ("graph", "remove_nodes", ["A"]),
    ("entities_vdb", "delete", "<ent ids>"),
    ("entity_chunks", "delete", ["A"]),
    ("rag", "_insert_done", None),
    ("rag", "rebuild_knowledge_from_chunks", {
        "entities": {"B": ["c3"]}, "relationships": {"B|C": ["c3"]},
    }),
]


def _comparable(log):
    """Mask values that are hashes or clocks so the expected log stays readable."""
    out = []
    for store, method, args in log:
        if store in ("entities_vdb", "relationships_vdb") and method == "delete":
            args = "<ent ids>" if store == "entities_vdb" else "<rel ids>"
        if method == "upsert" and store in ("entity_chunks", "relation_chunks"):
            args = {k: {**v, "updated_at": "<t>"} for k, v in args.items()}
        if store == "doc_status" and method == "upsert":
            args = {k: {**v, "updated_at": "<t>", "metadata": {
                mk: ("<t>" if mk == "last_deletion_attempt_at" else mv) for mk, mv in v["metadata"].items()
            }} for k, v in args.items()}
        out.append((store, method, args))
    return out


def test_delete_document_store_calls_stages_and_result(harness):
    rag, stages, rebuilds = harness
    result = asyncio.run(rag.adelete_by_doc_id("doc-1", delete_llm_cache=True))

    assert (result.status, result.status_code, result.file_path) == ("success", 200, "a.md")
    assert result.message == "Successfully deleted 2 LLM cache entries for document doc-1"
    retry_state = {"doc-1": {
        "chunks_list": ["c1", "c2"], "error_msg": "", "file_path": "a.md",
        "metadata": {"deletion_llm_cache_ids": ["k1", "k2"], "last_deletion_attempt_at": "<t>"},
        "status": "processed", "updated_at": "<t>",
    }}
    assert _comparable(rag.log) == [
        ("doc_status", "upsert", retry_state),
        *GRAPH_PURGE,
        ("llm_response_cache", "delete", ["k1", "k2"]),
        ("full_entities", "delete", ["doc-1"]),
        ("full_relations", "delete", ["doc-1"]),
        ("doc_status", "delete", ["doc-1"]),
        ("full_docs", "delete", ["doc-1"]),
        ("rag", "_insert_done", None),
    ]
    assert stages == [
        "initializing", "collect_llm_cache", "analyze_graph_dependencies", "update_chunk_tracking",
        "delete_chunks", "delete_relationships", "delete_entities", "persist_pre_rebuild_changes",
        "rebuild_knowledge_graph", "delete_llm_cache", "delete_doc_graph_metadata", "delete_doc_entries",
    ]
    [kwargs] = rebuilds
    assert kwargs["knowledge_graph_inst"] is rag.chunk_entity_relation_graph
    assert kwargs["llm_response_cache"] is rag.llm_response_cache
    assert kwargs["text_chunks_storage"] is rag.text_chunks
    assert kwargs["entity_chunks_storage"] is rag.entity_chunks
    assert kwargs["relation_chunks_storage"] is rag.relation_chunks
    assert kwargs["doc_status_storage"] is rag.doc_status
    assert kwargs["global_config"] == {"cfg": True}
    assert set(rag.chunk_entity_relation_graph.nodes) == {"B", "C"}
    assert "doc-1" not in rag.doc_status.rows


def test_delete_failure_records_stage_for_retry(harness):
    rag, stages, _ = harness
    rag.fail_rebuild = True
    result = asyncio.run(rag.adelete_by_doc_id("doc-1", delete_llm_cache=True))

    assert (result.status, result.status_code) == ("fail", 500)
    assert result.message == "Error while deleting document doc-1: Failed to rebuild knowledge graph: boom"
    meta = rag.doc_status.rows["doc-1"]["metadata"]
    assert meta["deletion_failed"] is True
    assert meta["deletion_failure_stage"] == "rebuild_knowledge_graph"
    assert meta["deletion_llm_cache_ids"] == ["k1", "k2"]
    assert stages[-1] == "rebuild_knowledge_graph"
    # Nothing past the rebuild ran: cache, per-doc index and the document rows are kept for the retry.
    assert set(rag.llm_response_cache.rows) == {"k1", "k2", "k3"}
    assert "doc-1" in rag.full_entities.rows and "doc-1" in rag.full_docs.rows


def test_resume_purge_store_calls(harness):
    rag, stages, _ = harness
    asyncio.run(rag._purge_doc_chunks_and_kg("doc-1", ["c1", "c2"]))

    assert _comparable(rag.log) == [
        *GRAPH_PURGE,
        ("full_entities", "delete", ["doc-1"]),
        ("full_relations", "delete", ["doc-1"]),
    ]
    assert stages == []  # the resume path reports no deletion stages
    assert "doc-1" in rag.doc_status.rows and "doc-1" in rag.full_docs.rows
