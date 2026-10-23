import logging
from collections.abc import Awaitable, Callable, Generator
from typing import Any

import numpy as np

from hybrid_rag.config import settings
from hybrid_rag.core.embedding import EmbeddingFunc

logger = logging.getLogger(__name__)

_hooks: dict[str, Any] = {
    "get_llm": None,
    "get_vlm": None,
    "get_embeddings": None,
    "get_reranker": None,
}

def configure(**hooks: Any) -> None:
    """Install the host's hooks; unknown names are rejected."""
    unknown = set(hooks) - set(_hooks)
    if unknown:
        raise TypeError(f"unknown provider hook(s): {', '.join(sorted(unknown))}")
    _hooks.update({name: value for name, value in hooks.items() if value is not None})

def _require(name: str) -> Any:
    hook = _hooks[name]
    if hook is None:
        raise RuntimeError(f"hybrid_rag.providers.configure({name}=...) has not been called")
    return hook

# Model families whose endpoint rejects the field outright (Bedrock: "This model doesn't
# support the temperature field"), so it is never sent to them.
_NO_TEMPERATURE_FAMILIES = ("grok",)


def _accepts_temperature(model_id: str | None) -> bool:
    return not any(family in (model_id or "").lower() for family in _NO_TEMPERATURE_FAMILIES)


def _temperature_kwargs(model_id: str | None, temperature: float | None) -> dict[str, Any]:
    """An explicit None for families that reject the field: leaving it out lets the hook's own default (0) through."""
    if not _accepts_temperature(model_id):
        return {"temperature": None}
    return {} if temperature is None else {"temperature": temperature}


def get_llm(
    model_id: str,
    *,
    base_url: str | None,
    api_key: str | None,
    timeout: float | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    extra = _temperature_kwargs(model_id, temperature)
    return _require("get_llm")(
        model_id,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=settings.max_retries,
        max_tokens=max_tokens,
        **extra,
    )

def get_vlm(model_id: str, *, base_url: str | None, api_key: str | None, timeout: float | None = None) -> Any:
    """A vision-capable chat model, uncapped: a ``max_tokens`` cap truncates the structured output."""
    hook = _hooks["get_vlm"] or _require("get_llm")
    return hook(
        model_id, base_url=base_url, api_key=api_key, timeout=timeout, max_retries=settings.max_retries, max_tokens=None,
        **_temperature_kwargs(model_id, None),
    )

def get_embeddings(model_id: str, *, base_url: str | None, api_key: str | None, timeout: float | None = None) -> Any:
    return _require("get_embeddings")(model_id, base_url=base_url, api_key=api_key, timeout=timeout, max_retries=settings.max_retries)

def get_reranker(
    model_id: str, *, base_url: str | None, api_key: str | None, timeout: float | None = None
) -> Callable[..., Any] | None:
    """The rerank function, or None when the host installed no ``get_reranker``."""
    hook = _hooks["get_reranker"]
    if hook is None:
        return None
    return hook(model_id, base_url=base_url, api_key=api_key, timeout=timeout, max_retries=settings.max_retries)


def embed_by_index_sync(create: Callable[..., Any], texts: list[str], batch_size: int) -> list[list[float]]:
    """:func:`embed_by_index` for a synchronous ``create``."""
    requests = _index_paired_requests(texts, batch_size)
    try:
        batch = next(requests)
        while True:
            batch = requests.send(create(input=batch))
    except StopIteration as done:
        return done.value


async def embed_by_index(
    create: Callable[..., Awaitable[Any]], texts: list[str], batch_size: int
) -> list[list[float]]:
    """Embed ``texts`` through an OpenAI-compatible ``create(input=...)``, pairing each vector by its ``index``.

    An OpenAI-compatible endpoint is not bound to answer in input order. A caching
    proxy given a batch that mixes cached and uncached inputs returns some inputs
    twice and others not at all, each still labelled with its true ``index``; a
    client pairing by position (``OpenAIEmbeddings``) then stores one text's vector
    under another. Here every vector goes to the input its ``index`` names, and an
    input left without one is asked for again on its own.
    """
    requests = _index_paired_requests(texts, batch_size)
    try:
        batch = next(requests)
        while True:
            batch = requests.send(await create(input=batch))
    except StopIteration as done:
        return done.value


def _index_paired_requests(texts: list[str], batch_size: int) -> Generator[list[str], Any, list[list[float]]]:
    """The pairing shared by both drivers: yields each request's ``input``, is sent its response.

    Written once so the sync and async paths cannot drift apart; the drivers only
    differ in whether they await ``create``.
    """
    vectors: dict[int, list[float]] = {}
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for item in _embedding_data((yield batch)):
            if 0 <= item["index"] < len(batch):
                vectors.setdefault(start + item["index"], item["embedding"])
    missing = [i for i in range(len(texts)) if i not in vectors]
    if missing:
        logger.warning(f"embedding endpoint left {len(missing)}/{len(texts)} inputs without a vector; re-asking one by one")
    for i in missing:
        data = _embedding_data((yield [texts[i]]))
        if len(data) != 1:
            raise RuntimeError(f"embedding endpoint returned {len(data)} vectors for one input")
        vectors[i] = data[0]["embedding"]
    return [vectors[i] for i in range(len(texts))]


def _embedding_data(response: Any) -> list[dict]:
    return (response if isinstance(response, dict) else response.model_dump())["data"]


def build_embedding_func() -> EmbeddingFunc:
    """Adapt the host's LangChain embedding model to the storages' ndarray contract.

    Token truncation, retries and the cached client stay with the client the host's
    ``get_embeddings`` builds; this only converts to float32 arrays.
    """
    embeddings = get_embeddings(
        model_id=settings.embedding_model,
        base_url=settings.embedding_binding_host,
        api_key=settings.embedding_binding_api_key,
        timeout=settings.embedding_timeout,
    )

    async def embed(texts: list[str]) -> np.ndarray:
        return np.array(await embeddings.aembed_documents(texts), dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=settings.embedding_token_limit,
        func=embed,
        model_name=settings.embedding_model,
    )
