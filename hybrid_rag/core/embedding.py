"""The embedding function every vector storage uses."""

import numpy as np
from hybrid_rag import providers
from hybrid_rag.config import settings
from hybrid_rag.core.utils import EmbeddingFunc

def build_embedding_func() -> EmbeddingFunc:
    """Adapt the host's LangChain embedding model to the storages' ndarray contract.

    Token truncation, retries and the cached client stay with the client the host's
    ``get_embeddings`` builds; this only converts to float32 arrays.
    """
    embeddings = providers.get_embeddings(
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