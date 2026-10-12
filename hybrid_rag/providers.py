"""What the host application plugs in: model clients, object storage, progress.

The package builds no provider client of its own. The host calls
:func:`configure` once at import time, before any instance is built:

* ``get_llm(model_id, *, max_tokens, base_url, api_key, timeout, temperature)``
  — a LangChain chat model; extraction, contextualize, summaries, VLM and
  keyword chains all go through it.
* ``get_embeddings(model_id, base_url, api_key)`` — a LangChain ``Embeddings``.
* ``rerank(query, documents, top_n=...)`` — ``[{"index", "relevance_score"}]``;
  leave unset to retrieve without reranking.
* ``s3`` — object with ``upload_figures_to_s3``, ``upload_fileobj``,
  ``delete_object`` and ``require_bucket`` (Textract input and figure crops).
* ``progress`` — object with ``stage``, ``skip``, ``activity`` and the
  ``current_doc_id`` / ``current_workspace`` context variables; optional, the
  default reports nothing.

Package code reads them through the module attributes below (``progress.stage``,
``s3.upload_fileobj``), so a missing hook fails with a message naming it.
"""

from contextvars import ContextVar
from typing import Any, Callable


class _Silent:
    """Default progress reporter: drops every event."""

    current_doc_id: ContextVar[str | None] = ContextVar("hybrid_rag_doc_id", default=None)
    current_workspace: ContextVar[str | None] = ContextVar("hybrid_rag_workspace", default=None)

    def stage(self, *args: Any, **kwargs: Any) -> None:
        pass

    def skip(self, *args: Any, **kwargs: Any) -> None:
        pass

    def activity(self, *args: Any, **kwargs: Any) -> None:
        pass


_hooks: dict[str, Any] = {
    "get_llm": None,
    "get_embeddings": None,
    "rerank": None,
    "s3": None,
    "progress": _Silent(),
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


class _Proxy:
    """Attribute access forwards to the hook installed under ``name``."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        return getattr(_require(self._name), attr)


s3: Any = _Proxy("s3")
progress: Any = _Proxy("progress")
