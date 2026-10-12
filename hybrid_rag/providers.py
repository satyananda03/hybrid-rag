"""What the host application plugs in: how to reach the models.

The package builds no model client of its own. The host calls :func:`configure`
once at import time, before any instance is built, with one factory per kind of
model. All three take the same connection arguments, every one of them keyword-only
after ``model_id``, and the library fills them from ``.env``:

* ``get_llm(model_id, *, base_url, api_key, timeout, max_retries, max_tokens[, temperature])``
  — a LangChain chat model. Extraction, contextualize, summaries, VLM and keyword
  chains all go through it. ``temperature`` is passed only where the library sets
  it; otherwise use the provider's default.
* ``get_embeddings(model_id, *, base_url, api_key, timeout, max_retries)`` — a
  LangChain ``Embeddings``.
* ``get_reranker(model_id, *, base_url, api_key, timeout, max_retries)`` — an async
  ``rerank(query, documents, top_n)`` returning ``[{"index", "relevance_score"}]``.
  Optional: without it, or without ``RERANK_MODEL`` / ``RERANK_BINDING_HOST``,
  retrieval runs unreranked.

``timeout`` is seconds per call (None: no limit). ``max_retries`` is ``MAX_RETRIES``:
retrying a model call is the client's job, since only it knows which errors are
safe to repeat, so the factory must hand it to the client and the library adds no
retry loop of its own.

A missing required hook fails with a message naming it. Everything else is
configured elsewhere: S3 and the databases by environment variables
(:mod:`hybrid_rag.config`), progress reporting by :mod:`hybrid_rag.progress`.
"""

from typing import Any, Callable
from hybrid_rag.config import settings

_hooks: dict[str, Any] = {
    "get_llm": None,
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

def get_llm(
    model_id: str,
    *,
    base_url: str | None,
    api_key: str | None,
    timeout: float | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    extra = {} if temperature is None else {"temperature": temperature}
    return _require("get_llm")(
        model_id,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=settings.max_retries,
        max_tokens=max_tokens,
        **extra,
    )


def get_embeddings(
    model_id: str, *, base_url: str | None, api_key: str | None, timeout: float | None = None
) -> Any:
    return _require("get_embeddings")(
        model_id, base_url=base_url, api_key=api_key, timeout=timeout, max_retries=settings.max_retries
    )

def get_reranker(
    model_id: str, *, base_url: str | None, api_key: str | None, timeout: float | None = None
) -> Callable[..., Any] | None:
    """The rerank function, or None when the host installed no ``get_reranker``."""
    hook = _hooks["get_reranker"]
    if hook is None:
        return None
    return hook(model_id, base_url=base_url, api_key=api_key, timeout=timeout, max_retries=settings.max_retries)