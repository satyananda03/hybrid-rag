from typing import Any, Callable
import numpy as np
from hybrid_rag.config import settings
from hybrid_rag.core.embedding import EmbeddingFunc

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

def get_vlm(model_id: str, *, base_url: str | None, api_key: str | None, timeout: float | None = None) -> Any:
    """A vision-capable chat model, uncapped: a ``max_tokens`` cap truncates the structured output."""
    hook = _hooks["get_vlm"] or _require("get_llm")
    return hook(model_id, base_url=base_url, api_key=api_key, timeout=timeout, max_retries=settings.max_retries, max_tokens=None)

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
