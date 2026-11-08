"""Stage 1 of retrieval — search. Entry point: :func:`hybrid_search`.

    hybrid.py  hybrid_search(): dense + sparse + graph, concurrently
    graph.py   GraphSearch: entities / relations by vector, expanded through the graph
    fusion.py  weighted_rrf(): fuses the dense and sparse chunk rankings
"""

from hybrid_rag.retrieval.search.hybrid import hybrid_search

__all__ = ["hybrid_search"]
