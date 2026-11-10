"""Rank fusion of search arms whose hits share an ``id``."""

RRF_K = 60.0  # the usual RRF constant (Neo4j hybrid-search guide, Cormack et al.)

def weighted_rrf(arms: list[tuple[list[dict], float]], limit: int, k: float = RRF_K) -> list[dict]:
    """Weighted Reciprocal Rank Fusion of ``(hits, weight)`` arms, each ranked best first.

    A hit scores ``weight / (k + rank)`` per arm it appears in (rank from 1), summed into
    ``rrf_score``. Best first, ties broken by ``id`` so the order is reproducible. A hit
    keeps the fields of the first arm that returned it.
    """
    fused: dict[str, dict] = {}
    for hits, weight in arms:
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit["id"], {**hit, "rrf_score": 0.0})
            entry["rrf_score"] += weight / (k + rank)
    return sorted(fused.values(), key=lambda h: (-h["rrf_score"], h["id"]))[:limit]