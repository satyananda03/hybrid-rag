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