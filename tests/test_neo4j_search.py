"""Pure logic of the Neo4j vector/BM25 storage: no database, network or model."""

import asyncio
import datetime

import pytest

from hybrid_rag.storage.neo4j import bm25, schema, vector
from hybrid_rag.storage.neo4j.vector import Neo4jVectorStorage, _PendingVectorDoc, _sorted_pair


def test_score_and_distance_match_pgvector_cosine():
    # Neo4j cosine score is (1 + cos) / 2; pgvector's <=> was 1 - cos.
    for cos in (-1.0, 0.0, 0.2, 0.9, 1.0):
        score, dist = (1 + cos) / 2, 1 - cos
        assert vector.distance(score) == pytest.approx(dist)
        assert vector.score_floor(dist) == pytest.approx(score)


def test_bm25_query_requires_workspace_and_matches_any_word():
    q = bm25.bm25_query("ws.v1-a", "pasal 25 spip")
    key = schema.ws_key("ws.v1-a")
    assert q == f"+ws_key:{key} +content_cleaned:(pasal 25 spip)"
    assert key.isalnum() and key != schema.ws_key("ws.v1-b")


def test_bm25_skips_the_database_for_a_stopword_only_query(monkeypatch):
    async def no_db(*a, **kw):
        raise AssertionError("an empty Lucene clause must never reach Neo4j")

    monkeypatch.setattr(bm25, "read", no_db)
    monkeypatch.setattr(bm25, "get_stopwords", lambda: frozenset({"yang", "dan"}))
    assert asyncio.run(bm25.Neo4jBM25Storage("w").query("yang, dan?", top_k=5)) == []


def test_relation_hits_have_sorted_endpoints():
    assert _sorted_pair({"src_id": "b", "tgt_id": "a", "created_at": 1}) == {
        "src_id": "a", "tgt_id": "b", "created_at": 1,
    }


def test_unmatched_vectors_wait_until_ttl():
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    storage = object.__new__(Neo4jVectorStorage)  # only the buffer is exercised
    storage.workspace, storage.namespace = "w", "entities"
    storage._pending_vector_docs = {
        "written": _PendingVectorDoc(item={}, created_at=now),
        "waiting": _PendingVectorDoc(item={}, created_at=now),
        "expired": _PendingVectorDoc(item={}, created_at=now - datetime.timedelta(hours=2)),
    }
    storage._keep_unmatched({"written"})
    assert list(storage._pending_vector_docs) == ["waiting"]
