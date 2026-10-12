"""What the host application plugs in: model clients.

The package builds no provider client of its own. The host calls
:func:`configure` once at import time, before any instance is built:

* ``get_llm(model_id, *, max_tokens, base_url, api_key, timeout, temperature)``
  — a LangChain chat model; extraction, contextualize, summaries, VLM and
  keyword chains all go through it.
* ``get_embeddings(model_id, base_url, api_key)`` — a LangChain ``Embeddings``.
* ``rerank(query, documents, top_n=...)`` — ``[{"index", "relevance_score"}]``;
  leave unset to retrieve without reranking.

A missing required hook fails with a message naming it. Everything else is
configured elsewhere: S3 and the databases by environment variables
(:mod:`hybrid_rag.config`), progress reporting by :mod:`hybrid_rag.progress`.
"""

from typing import Any, Callable


_hooks: dict[str, Any] = {
    "get_llm": None,
    "get_embeddings": None,
    "rerank": None,
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


def get_llm(*args: Any, **kwargs: Any) -> Any:
    return _require("get_llm")(*args, **kwargs)


def get_embeddings(*args: Any, **kwargs: Any) -> Any:
    return _require("get_embeddings")(*args, **kwargs)


def rerank_func() -> Callable[..., Any] | None:
    """The configured rerank function, or None when reranking is off."""
    return _hooks["rerank"]
