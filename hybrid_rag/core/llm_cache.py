
"""LLM response caching as a step inside the chain, not a layer around it.

:func:`cached_chain` returns a Runnable with the same contract as the chain it
wraps — give it inputs, get the chain's output back — except it consults
``llm_response_cache`` first and writes the result back on a miss. It handles
both shapes in use here: a Pydantic model (``llm.with_structured_output(Model)``)
and plain text (``llm | StrOutputParser()``).

The cache storage is per-RAG-instance while chains are built once at import, so
the storage and its bookkeeping arrive per call through ``configurable``::

    await chain.ainvoke(
        inputs,
        config={"configurable": {
            CACHE_KV: rag.llm_response_cache,
            CACHE_CHUNK_ID: chunk_key,
            CACHE_KEYS: keys_collected_here,
        }},
    )

Omit ``CACHE_KV`` (or pass ``None``) and the chain runs uncached.

Every key the chain reads or writes is appended to the ``CACHE_KEYS`` list, so
callers can record them on the chunk (``llm_cache_list``) and the delete path
can clean them up. Skipping that leaves cache rows behind after a document is
deleted, so pass the collector wherever a ``chunk_id`` exists.
"""


import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel, ValidationError

from hybrid_rag.core.utils import compute_args_hash

logger = logging.getLogger(__name__)


def generate_cache_key(mode: str, cache_type: str, hash_value: str) -> str:
    """Generate a flattened cache key in the format {mode}:{cache_type}:{hash}

    Args:
        mode: Cache mode (e.g., 'default', 'local', 'global')
        cache_type: Type of cache (e.g., 'extract', 'query', 'keywords')
        hash_value: Hash value from compute_args_hash

    Returns:
        str: Flattened cache key
    """
    return f"{mode}:{cache_type}:{hash_value}"


async def handle_cache(
    hashing_kv,
    args_hash,
    prompt,
    mode="default",
    cache_type="unknown",
) -> tuple[str, int] | None:
    """Generic cache handling function with flattened cache keys

    Returns:
        tuple[str, int] | None: (content, create_time) if cache hit, None if cache miss
    """
    if hashing_kv is None:
        return None

    if mode != "default":  # handle cache for all type of query
        if not hashing_kv.global_config.get("enable_llm_cache"):
            return None
    else:  # handle cache for entity extraction
        if not hashing_kv.global_config.get("enable_llm_cache_for_entity_extract"):
            return None

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(mode, cache_type, args_hash)
    cache_entry = await hashing_kv.get_by_id(flattened_key)
    if cache_entry:
        logger.debug(f"Flattened cache hit(key:{flattened_key})")
        content = cache_entry["return"]
        timestamp = cache_entry.get("create_time", 0)
        return content, timestamp

    logger.debug(f"Cache missed(mode:{mode} type:{cache_type})")
    return None


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    mode: str = "default"
    cache_type: str = "query"
    chunk_id: str | None = None
    queryparam: dict | None = None


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """Save data to cache using flattened key structure.

    Args:
        hashing_kv: The key-value storage for caching
        cache_data: The cache data to save
    """
    # Skip if storage is None or content is a streaming response
    if hashing_kv is None or not cache_data.content:
        return

    # If content is a streaming response, don't cache it
    if hasattr(cache_data.content, "__aiter__"):
        logger.debug("Streaming response detected, skipping cache")
        return

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(
        cache_data.mode, cache_data.cache_type, cache_data.args_hash
    )

    # Check if we already have identical content cached
    existing_cache = await hashing_kv.get_by_id(flattened_key)
    if existing_cache:
        existing_content = existing_cache.get("return")
        if existing_content == cache_data.content:
            logger.warning(
                f"Cache duplication detected for {flattened_key}, skipping update"
            )
            return

    # Create cache entry with flattened structure
    cache_entry = {
        "return": cache_data.content,
        "cache_type": cache_data.cache_type,
        "chunk_id": cache_data.chunk_id if cache_data.chunk_id is not None else None,
        "original_prompt": cache_data.prompt,
        "queryparam": cache_data.queryparam
        if cache_data.queryparam is not None
        else None,
    }

    logger.info(f" == LLM cache == saving: {flattened_key}")

    # Save using flattened key
    await hashing_kv.upsert({flattened_key: cache_entry})


# configurable keys
CACHE_KV = "llm_cache"
"""``BaseKVStorage`` holding cached responses. Absent/None disables caching."""

CACHE_CHUNK_ID = "cache_chunk_id"
"""Chunk this call belongs to; stored on the entry so deletion can find it."""

CACHE_KEYS = "cache_keys"
"""Mutable list; every key read or written is appended to it."""

CACHE_MODE = "cache_mode"
"""Overrides the build-time mode (query paths key their cache by query mode)."""

CACHE_QUERYPARAM = "cache_queryparam"
"""Query parameters recorded alongside a query-mode entry, for observability."""


def _output_fingerprint(output_model: type[BaseModel] | None) -> str:
    """Stable digest of the chain's output contract.

    Folded into the cache key so that changing the contract retires the old
    entries instead of replaying responses that no longer fit it. Text chains
    have no schema to hash, so they get a fixed sentinel.
    """
    if output_model is None:
        return "text"
    return compute_args_hash(
        json.dumps(output_model.model_json_schema(), sort_keys=True)
    )


def _cache_enabled(kv: Any, mode: str) -> bool:
    """Mirror the gate :func:`handle_cache` applies on read.

    Writes have to honour the same switch, or a disabled cache would still fill
    up with entries nothing will ever read.
    """
    config = getattr(kv, "global_config", None) or {}
    key = "enable_llm_cache_for_entity_extract" if mode == "default" else "enable_llm_cache"
    return bool(config.get(key))


def cached_chain(
    prompt: ChatPromptTemplate,
    llm: Runnable,
    *,
    cache_type: str,
    model_id: str,
    output_model: type[BaseModel] | None = None,
    mode: str = "default",
) -> Runnable:
    """Wrap ``prompt | llm`` with a cache lookup.

    Args:
        prompt: Template rendered from the invoke inputs.
        llm: The runnable invoked with the rendered messages — either a model
            bound via ``with_structured_output`` or one piped into
            ``StrOutputParser``.
        cache_type: Bucket recorded on the entry (``extract`` / ``keywords`` /
            ``vlm`` / ``summary``), and part of the cache key.
        model_id: Included in the cache key so switching models does not serve
            another model's answers.
        output_model: The model ``llm`` is bound to. Cached bodies are
            revalidated against it, so an entry written by an older contract is
            discarded rather than trusted. Leave as None for a text chain.
        mode: Default cache mode; ``CACHE_MODE`` overrides it per call.
    """
    schema_key = _output_fingerprint(output_model)
    default_mode = mode

    async def _invoke(inputs: dict[str, Any], config: RunnableConfig | None = None):
        configurable = (config or {}).get("configurable") or {}
        kv = configurable.get(CACHE_KV)
        messages = prompt.format_messages(**inputs)

        if kv is None:
            return await llm.ainvoke(messages, config)

        call_mode = configurable.get(CACHE_MODE) or default_mode
        prompt_text = "\n".join(str(message.content) for message in messages)
        args_hash = compute_args_hash(
            prompt_text,
            "\n<schema>\n",
            schema_key,
            "\n<model>\n",
            model_id,
        )
        cache_key = generate_cache_key(call_mode, cache_type, args_hash)
        collector = configurable.get(CACHE_KEYS)

        cached = await handle_cache(
            kv, args_hash, prompt_text, call_mode, cache_type=cache_type
        )
        if cached is not None:
            hit, usable = cached[0], True
            if output_model is not None:
                try:
                    hit = output_model.model_validate_json(cached[0])
                except ValidationError as e:
                    # Written before the contract changed, or by another writer.
                    # Re-running costs a call; trusting it would corrupt the
                    # graph.
                    usable = False
                    logger.warning(
                        f"Discarding unusable {cache_type} cache entry "
                        f"{cache_key}: {e}"
                    )
            if usable:
                if collector is not None:
                    collector.append(cache_key)
                return hit

        result = await llm.ainvoke(messages, config)
        if result is None:
            return None

        if _cache_enabled(kv, call_mode):
            await save_to_cache(
                kv,
                CacheData(
                    args_hash=args_hash,
                    content=result if output_model is None
                    else result.model_dump_json(),
                    prompt=prompt_text,
                    mode=call_mode,
                    cache_type=cache_type,
                    chunk_id=configurable.get(CACHE_CHUNK_ID),
                    queryparam=configurable.get(CACHE_QUERYPARAM),
                ),
            )
            if collector is not None:
                collector.append(cache_key)
        return result

    return RunnableLambda(_invoke, name=f"cached_{cache_type}")
