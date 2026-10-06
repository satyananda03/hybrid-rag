"""Retrieval entry point (``mix``: knowledge graph + vector) and its 4 stages.

* :func:`kg_query` — extracts high/low-level keywords, then runs
  :func:`_build_query_context`.
* :func:`_build_query_context` — the four stages, in order:
  search (:mod:`.search`) → truncate → merge chunks → build context string
  (all three in :mod:`.context`).

Nothing here generates an answer — retrieval stops at the context string.
"""


import re
from typing import Any

from hybrid_rag.core.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from hybrid_rag.retrieval.types import QueryParam, QueryResult
# Every prompt that reaches a model lives in retrieval/keywords/keywords.yaml and is wired
# into chains by keywords.chain. What is left in TEMPLATES is the two context templates
# that serialize retrieved entities/relations/chunks — formatting, not
# instructions.
from hybrid_rag.retrieval.keywords.chain import keywords_chain
from hybrid_rag.retrieval.keywords.schema import KeywordsOutput
from hybrid_rag.core.llm_cache import CACHE_KV, CACHE_MODE, CACHE_QUERYPARAM
from hybrid_rag.retrieval.context import (
    _apply_token_truncation,
    _build_context_str,
    _merge_all_chunks,
)
from hybrid_rag.retrieval.search import (
    _perform_kg_search,
)
import logging

logger = logging.getLogger(__name__)


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    chunks_vdb: BaseVectorStorage = None,
) -> QueryResult | None:
    """
    Execute knowledge graph query and return unified QueryResult object.

    Args:
        query: Query string
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_db: Text chunks storage
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        chunks_vdb: Document chunks vector database

    Returns:
        QueryResult | None: ``context`` holds the assembled context string and
        ``raw_data`` the structured entities / relationships / chunks behind it.
        None when no relevant context could be constructed for the query.
    """
    if not query:
        return None

    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if ll_keywords == [] and hl_keywords:
        # Query tematik tanpa entitas konkret (mis. "domain proses bisnis aplikasi data") sah
        # menghasilkan LL kosong; tanpa fallback ini pencarian entitas dilewati sama sekali.
        logger.info("low_level_keywords kosong; memakai high_level_keywords untuk pencarian entitas")
        ll_keywords = list(hl_keywords)
    if hl_keywords == []:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            logger.warning(f"Forced low_level_keywords to origin query: {query}")
            ll_keywords = [query]
        else:
            return None

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build query context (unified interface)
    context_result = await _build_query_context(
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    if context_result is None:
        logger.info("[kg_query] No query context could be built; returning no-result.")
        return None

    return context_result

def _normalize_keyword_list(raw_values: Any, field_name: str) -> list[str]:
    """Normalize keyword payloads into a clean list of strings.

    When the field is a plain string (e.g. LLM returned CSV), split on
    newlines/commas/semicolons. List-shaped payloads are preserved per-item so
    multi-word phrases that legitimately contain commas are not broken apart.
    """

    if raw_values is None:
        return []

    if isinstance(raw_values, str):
        raw_values = [
            part.strip()
            for part in re.split(r"[\n,;]+", raw_values)
            if part and part.strip()
        ]

    if not isinstance(raw_values, list):
        logger.warning(
            "Keyword extraction field '%s' is not a list: %r",
            field_name,
            raw_values,
        )
        return []

    normalized: list[str] = []
    for idx, value in enumerate(raw_values):
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                normalized.append(cleaned)
            continue

        logger.warning(
            "Keyword extraction field '%s' contains non-string element at index %d: %r",
            field_name,
            idx,
            value,
        )

    return normalized

async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """
    result: KeywordsOutput | None = await keywords_chain().ainvoke(
        {"query": text},
        config={
            "configurable": {
                CACHE_KV: hashing_kv,
                # "mix" is part of the stored cache key; keep it so existing
                # keyword entries stay addressable.
                CACHE_MODE: "mix",
                CACHE_QUERYPARAM: {
                    "mode": "mix",
                    "top_k": param.top_k,
                    "chunk_top_k": param.chunk_top_k,
                    "max_entity_tokens": param.max_entity_tokens,
                    "max_relation_tokens": param.max_relation_tokens,
                    "enable_rerank": param.enable_rerank,
                },
            }
        },
    )
    if result is None:
        logger.warning("Keyword extraction returned no structured output")
        return [], []

    return (
        _normalize_keyword_list(result.high_level_keywords, "high_level_keywords"),
        _normalize_keyword_list(result.low_level_keywords, "low_level_keywords"),
    )

async def _build_query_context(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
) -> QueryResult | None:
    """
    Main query context building function using the new 4-stage architecture:
    1. Search -> 2. Truncate -> 3. Merge chunks -> 4. Build LLM context

    Returns a QueryResult holding both the context string and raw_data.
    """

    if not query:
        logger.warning("Query is empty, skipping context building")
        return None

    # Stage 1: Pure search
    search_result = await _perform_kg_search(
        query,
        ll_keywords,
        hl_keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    if not search_result["final_entities"] and not search_result["final_relations"]:
        if not search_result["chunk_tracking"]:
            return None

    # Stage 2: Apply token truncation for LLM efficiency
    truncation_result = await _apply_token_truncation(
        search_result,
        query_param,
        text_chunks_db.global_config,
    )

    # Stage 3: Merge chunks using filtered entities/relations
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result["filtered_entities"],
        filtered_relations=truncation_result["filtered_relations"],
        vector_chunks=search_result["vector_chunks"],
        query=query,
        knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=chunks_vdb,
        chunk_tracking=search_result["chunk_tracking"],
        query_embedding=search_result["query_embedding"],
    )

    if (
        not merged_chunks
        and not truncation_result["entities_context"]
        and not truncation_result["relations_context"]
    ):
        return None

    # Stage 4: Build final LLM context with dynamic token processing
    # _build_context_str now always returns tuple[str, dict]
    context, raw_data = await _build_context_str(
        entities_context=truncation_result["entities_context"],
        relations_context=truncation_result["relations_context"],
        merged_chunks=merged_chunks,
        query=query,
        query_param=query_param,
        global_config=text_chunks_db.global_config,
        chunk_tracking=search_result["chunk_tracking"],
        entity_id_to_original=truncation_result["entity_id_to_original"],
        relation_id_to_original=truncation_result["relation_id_to_original"],
        text_chunks_db=text_chunks_db,
    )

    # Convert keywords strings to lists and add complete metadata to raw_data
    hl_keywords_list = hl_keywords.split(", ") if hl_keywords else []
    ll_keywords_list = ll_keywords.split(", ") if ll_keywords else []

    # Add complete metadata to raw_data (preserve existing metadata including query_mode)
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    # Update keywords while preserving existing metadata
    raw_data["metadata"]["keywords"] = {
        "high_level": hl_keywords_list,
        "low_level": ll_keywords_list,
    }
    raw_data["metadata"]["processing_info"] = {
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "entities_after_truncation": len(
            truncation_result.get("filtered_entities", [])
        ),
        "relations_after_truncation": len(
            truncation_result.get("filtered_relations", [])
        ),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }

    logger.debug(
        f"[_build_query_context] Context length: {len(context) if context else 0}"
    )
    logger.debug(
        f"[_build_query_context] Raw data entities: {len(raw_data.get('data', {}).get('entities', []))}, relationships: {len(raw_data.get('data', {}).get('relationships', []))}, chunks: {len(raw_data.get('data', {}).get('chunks', []))}"
    )

    return QueryResult(context=context, raw_data=raw_data)

