"""Reranking — the last filter between raw retrieval and the built context.

Two functions, called from :mod:`hybrid_rag.retrieval.context` while it
merges chunks:

* :func:`apply_rerank_if_enabled` — the rerank call itself, a no-op when no
  rerank model is configured or ``enable_rerank`` is off.
* :func:`process_chunks_unified` — the pipeline around it: rerank, drop chunks
  below ``min_rerank_score``, cut to ``chunk_top_k``, then stamp each chunk with a ``DC<n>`` id the context template cites.

The rerank model itself lives outside this module: ``factory.py`` injects the
host's rerank provider as ``global_config["rerank_model_func"]``.
Nothing here builds a provider client.
"""
from typing import TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from hybrid_rag.retrieval.types import QueryParam

async def apply_rerank_if_enabled(
    query: str,
    retrieved_docs: list[dict],
    global_config: dict,
    enable_rerank: bool = True,
    top_n: int = None,
) -> list[dict]:
    """
    Apply reranking to retrieved documents if rerank is enabled.

    Args:
        query: The search query
        retrieved_docs: List of retrieved documents
        global_config: Global configuration containing rerank settings
        enable_rerank: Whether to enable reranking from query parameter
        top_n: Number of top documents to return after reranking

    Returns:
        Reranked documents if rerank is enabled, otherwise original documents
    """
    if not enable_rerank or not retrieved_docs:
        return retrieved_docs

    rerank_func = global_config.get("rerank_model_func")
    if not rerank_func:
        logger.warning(
            "Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters."
        )
        return retrieved_docs

    try:
        # Extract document content for reranking
        document_texts = []
        for doc in retrieved_docs:
            # Try multiple possible content fields
            content = (
                doc.get("content")
                or doc.get("text")
                or doc.get("chunk_content")
                or doc.get("document")
                or str(doc)
            )
            document_texts.append(content)

        # Call the new rerank function that returns index-based results
        top_n_clamped = min(top_n, len(document_texts)) if top_n is not None else None
        rerank_results = await rerank_func(
            query=query,
            documents=document_texts,
            top_n=top_n_clamped,
        )

        # Process rerank results based on return format
        if rerank_results and len(rerank_results) > 0:
            # Check if results are in the new index-based format
            if isinstance(rerank_results[0], dict) and "index" in rerank_results[0]:
                # New format: [{"index": 0, "relevance_score": 0.85}, ...]
                reranked_docs = []
                for result in rerank_results:
                    index = result["index"]
                    relevance_score = result["relevance_score"]

                    # Get original document and add rerank score
                    if 0 <= index < len(retrieved_docs):
                        doc = retrieved_docs[index].copy()
                        doc["rerank_score"] = relevance_score
                        reranked_docs.append(doc)

                logger.info(
                    f"Successfully reranked: {len(reranked_docs)} chunks from {len(retrieved_docs)} original chunks"
                )
                return reranked_docs
            else:
                # Legacy format: assume it's already reranked documents
                logger.info(f"Using legacy rerank format: {len(rerank_results)} chunks")
                return rerank_results[:top_n] if top_n else rerank_results
        else:
            logger.warning("Rerank returned empty results, using original chunks")
            return retrieved_docs

    except Exception as e:
        logger.error(f"Error during reranking: {e}, using original chunks")
        return retrieved_docs


async def process_chunks_unified(
    query: str,
    unique_chunks: list[dict],
    query_param: "QueryParam",
    global_config: dict,
) -> list[dict]:
    """
    Unified processing for text chunks: reranking and chunk_top_k limiting.

    Chunks are original source text, so they are never truncated by tokens —
    ``chunk_top_k`` is the only cap.

    Args:
        query: Search query for reranking
        chunks: List of text chunks to process
        query_param: Query parameters containing configuration
        global_config: Global configuration dictionary

    Returns:
        Processed and filtered list of text chunks
    """
    if not unique_chunks:
        return []

    origin_count = len(unique_chunks)

    # 1. Apply reranking if enabled and query is provided
    if query_param.enable_rerank and query and unique_chunks:
        rerank_top_k = (
            min(query_param.chunk_top_k, len(unique_chunks))
            if query_param.chunk_top_k is not None
            else len(unique_chunks)
        )
        unique_chunks = await apply_rerank_if_enabled(
            query=query,
            retrieved_docs=unique_chunks,
            global_config=global_config,
            enable_rerank=query_param.enable_rerank,
            top_n=rerank_top_k,
        )

    # 2. Filter by minimum rerank score if reranking is enabled
    if query_param.enable_rerank and unique_chunks:
        min_rerank_score = global_config.get("min_rerank_score", 0.5)
        if min_rerank_score > 0.0:
            original_count = len(unique_chunks)

            # Filter chunks with score below threshold
            filtered_chunks = []
            for chunk in unique_chunks:
                rerank_score = chunk.get(
                    "rerank_score", 1.0
                )  # Default to 1.0 if no score
                if rerank_score >= min_rerank_score:
                    filtered_chunks.append(chunk)

            unique_chunks = filtered_chunks
            filtered_count = original_count - len(unique_chunks)

            if filtered_count > 0:
                logger.info(
                    f"Rerank filtering: {len(unique_chunks)} chunks remained (min rerank score: {min_rerank_score})"
                )
            if not unique_chunks:
                return []

    # 3. Apply chunk_top_k limiting if specified
    if query_param.chunk_top_k is not None and query_param.chunk_top_k > 0:
        if len(unique_chunks) > query_param.chunk_top_k:
            unique_chunks = unique_chunks[: query_param.chunk_top_k]
        logger.debug(
            f"Kept chunk_top-k: {len(unique_chunks)} chunks (deduplicated original: {origin_count})"
        )

    # 4. add id field to each chunk
    final_chunks = []
    for i, chunk in enumerate(unique_chunks):
        chunk_with_id = chunk.copy()
        chunk_with_id["id"] = f"DC{i + 1}"
        final_chunks.append(chunk_with_id)

    return final_chunks
