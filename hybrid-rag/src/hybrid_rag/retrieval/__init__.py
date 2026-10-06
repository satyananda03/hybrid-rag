"""Retrieval — a query in, a context string out. Never an answer.

``pipeline.py`` is the outer boundary (``retrieve_context``). Inside:

    rag.py        RetrievalRAG: the read-side storages, ``aretrieve``
    query.py      keyword extraction + the 4-stage orchestration
    search.py     stage 1: entities, relations and hybrid (vector + BM25) chunks
    context.py    stages 2-4: truncate, merge chunks, render the context
    rerank.py     the rerank pass inside stage 3
    keywords/     the keyword-extraction prompt, schema and cached chain
    types.py      QueryParam / QueryResult

There is one mode, ``mix``: knowledge graph and vector search together.
"""


# Public API, resolved on first access so importing a submodule stays cheap.
_EXPORTS = {
    "RetrievalRAG": "hybrid_rag.retrieval.rag",
    "build_rag": "hybrid_rag.retrieval.factory",
    "workspace_pool": "hybrid_rag.retrieval.workspace",
    "retrieve_context": "hybrid_rag.retrieval.pipeline",
    "retrieve_context_without_relations": "hybrid_rag.retrieval.pipeline",
    "RetrievalResult": "hybrid_rag.retrieval.pipeline",
    "QueryParam": "hybrid_rag.retrieval.types",
    "QueryResult": "hybrid_rag.retrieval.types",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)
