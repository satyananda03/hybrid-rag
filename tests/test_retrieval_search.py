"""Pure logic of the retrieval search stage: RRF fusion and the graph round-robin."""

import pytest

from hybrid_rag.retrieval.search.fusion import weighted_rrf
from hybrid_rag.retrieval.search.graph import _relation_key, _round_robin


def test_weighted_rrf_sums_arms_and_breaks_ties_by_id():
    dense = [{"id": "a", "src": "dense"}, {"id": "c"}]
    sparse = [{"id": "a", "src": "sparse"}, {"id": "b"}]
    fused = weighted_rrf([(dense, 1.0), (sparse, 1.0)], limit=10, k=60)

    assert [h["id"] for h in fused] == ["a", "b", "c"]  # b and c tie at 1/62: by id
    assert fused[0]["rrf_score"] == pytest.approx(2 / 61)
    assert fused[0]["src"] == "dense"  # fields of the first arm that returned it


def test_weighted_rrf_weights_limit_and_empty_arm():
    fused = weighted_rrf([([{"id": "a"}], 0.3), ([{"id": "b"}], 0.7), ([], 1.0)], limit=1, k=60)
    assert [(h["id"], h["rrf_score"]) for h in fused] == [("b", 0.7 / 61)]


def test_round_robin_interleaves_and_skips_empty_or_seen_keys():
    first = [{"k": "a"}, {"k": ""}, {"k": "c"}]
    second = [{"k": "a"}, {"k": "b"}]
    assert [x["k"] for x in _round_robin(first, second, key=lambda x: x["k"])] == ["a", "b", "c"]


def test_relation_key_is_the_same_for_local_and_global_shapes():
    assert _relation_key({"src_tgt": ("b", "a")}) == _relation_key({"src_id": "a", "tgt_id": "b"})
