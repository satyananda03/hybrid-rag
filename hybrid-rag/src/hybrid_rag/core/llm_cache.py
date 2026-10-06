
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
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel, ValidationError

from hybrid_rag.core.utils import (
    CacheData,
    compute_args_hash,
    generate_cache_key,
    handle_cache,
    save_to_cache,
)
import logging

logger = logging.getLogger(__name__)


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
