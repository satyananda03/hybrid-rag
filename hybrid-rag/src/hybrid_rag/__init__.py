"""hybrid-rag — knowledge-graph + vector RAG over PostgreSQL and Neo4j.

Two entry points sharing one storage layer:

    ingestion/    documents in: parse → chunk → caption → extract graph → store; deletion
    retrieval/    a query in, a context string out (``mix``: graph + vector + BM25)
    core/         shared types, hashing, LLM cache, chunk schema, prompt loading
    storage/      PostgreSQL (KV, vector, doc status) and Neo4j (graph)
    config.py     settings, read from the environment / ``.env``
    providers.py  what the host plugs in: LLM, embeddings, rerank, S3, progress

Data is isolated per ``workspace``; one instance per workspace is pooled by
:mod:`hybrid_rag.workspace_pool`. The package never imports its host: call
:func:`hybrid_rag.providers.configure` once, then use ``hybrid_rag.ingestion``
and ``hybrid_rag.retrieval``.
"""
