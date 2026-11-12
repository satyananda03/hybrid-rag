"""Stage 1 of retrieval: run every search, return raw candidates.

Three searches, run concurrently:

* dense  — chunk vectors           (``Neo4jVectorStorage.query``)
* sparse — chunk BM25              (``Neo4jBM25Storage.query``)
* graph  — entities and relations  (:class:`.graph.GraphSearch`)

Dense and sparse return the same chunks and are fused with weighted RRF
(:func:`.fusion.weighted_rrf`). Graph returns entities and relations, whose chunks
:mod:`hybrid_rag.retrieval.context` collects and merges with these. Nothing is budgeted
or formatted here.
"""
import asyncio
import logging
from typing import Any
from hybrid_rag.config import settings
from hybrid_rag.core.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from hybrid_rag.retrieval.search.fusion import weighted_rrf
from hybrid_rag.retrieval.search.graph import GraphSearch
from hybrid_rag.retrieval.types import QueryParam
from hybrid_rag.storage.neo4j import Neo4jBM25Storage

logger = logging.getLogger(__name__)

async def hybrid_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage | None = None,
    chunks_bm25: Neo4jBM25Storage | None = None,
) -> dict[str, Any]:
    """Raw entities, relations and chunks for the query; chunk search runs when ``chunks_vdb`` is given."""
    pick_method = text_chunks_db.global_config.get("kg_chunk_pick_method", settings.kg_chunk_pick_method)
    # The query vector also serves context.py's VECTOR chunk pick.
    embeddings = await _embed_queries(text_chunks_db.embedding_func, {
        "query": query if (pick_method == "VECTOR" or chunks_vdb) else "",
        "ll": ll_keywords,
        "hl": hl_keywords,
    })
    query_embedding = embeddings.get("query")

    graph = GraphSearch(knowledge_graph_inst, entities_vdb, relationships_vdb, top_k=query_param.top_k)
    vector_chunks, (entities, relations) = await asyncio.gather(
        _search_chunks(query, query_embedding, chunks_vdb, chunks_bm25),
        graph.search(ll_keywords, hl_keywords, embeddings.get("ll"), embeddings.get("hl")),
    )

    # chunk_id -> where it came from ("C": chunk search) and its rank there.
    chunk_tracking = {
        chunk["chunk_id"]: {"source": "C", "frequency": 1, "order": i}
        for i, chunk in enumerate(vector_chunks, start=1)
    }
    logger.info(
        f"Raw search results: {len(entities)} entities, {len(relations)} relations, "
        f"{len(vector_chunks)} vector chunks"
    )
    return {
        "final_entities": entities,
        "final_relations": relations,
        "vector_chunks": vector_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }


async def _embed_queries(embedding_func, texts: dict[str, str]) -> dict[str, Any]:
    """Every non-empty text embedded in one call, by purpose.

    ``{}`` when that call fails: each search then embeds its own text.
    """
    texts = {purpose: text for purpose, text in texts.items() if text}
    if not embedding_func or not texts:
        return {}
    try:
        vectors = await embedding_func(list(texts.values()), context="query")
    except Exception as e:
        logger.warning(f"Failed to batch pre-compute embeddings: {e}")
        return {}
    return dict(zip(texts, vectors))


async def _search_chunks(
    query: str,
    query_embedding: Any,
    chunks_vdb: BaseVectorStorage | None,
    chunks_bm25: Neo4jBM25Storage | None,
) -> list[dict]:
    """Dense and sparse chunk search fused with weighted RRF; ``[]`` if a search fails."""
    if chunks_vdb is None:
        return []
    candidates = settings.hybrid_search_chunk_candidates
    try:
        dense, sparse = await asyncio.gather(
            chunks_vdb.query(query, top_k=candidates, query_embedding=query_embedding),
            chunks_bm25.query(query, top_k=candidates),
        )
    except Exception as e:
        logger.error(f"Chunk search failed: {e}")
        return []

    fused = weighted_rrf(
        [(dense, settings.hybrid_search_w_semantic), (sparse, settings.hybrid_search_w_keyword)],
        limit=candidates,
    )
    logger.info(
        f"Chunk search: {len(fused)} chunks (dense {len(dense)}, sparse {len(sparse)}, "
        f"candidates {candidates}, cosine {chunks_vdb.cosine_better_than_threshold})"
    )
    return [
        {
            "content": hit["content"],
            "created_at": hit.get("created_at"),
            "file_path": hit.get("file_path", "unknown_source"),
            "source_type": "vector",
            "chunk_id": hit["id"],
        }
        for hit in fused
    ]
