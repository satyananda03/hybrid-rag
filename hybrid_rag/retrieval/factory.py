"""Build a ready-to-use :class:`RetrievalRAG` for one workspace.

Workspace isolation lives in ``workspace`` — it is forwarded to every storage, so
two instances built with different workspaces share the database but never each
other's data. Building one instance per workspace is what
:mod:`hybrid_rag.retrieval.workspace` does.

Models reach their clients through :mod:`hybrid_rag.providers`: the keyword LLM
in ``keywords/chain.py``, rerank and embedding here.
"""

from hybrid_rag import providers
from hybrid_rag.config import settings
from hybrid_rag.core.embedding import build_embedding_func
from hybrid_rag.retrieval.rag import RetrievalRAG


async def build_rag(workspace: str) -> RetrievalRAG:
    """Construct a ``RetrievalRAG`` for ``workspace`` and bring its storages up.

    The caller owns the returned instance and is responsible for
    ``await rag.finalize_storages()`` on shutdown.
    """
    settings.require("keyword", "embedding")
    rag = RetrievalRAG(
        workspace=workspace,
        enable_llm_cache=settings.enable_llm_cache,
        embedding_func=build_embedding_func(),
        # None disables reranking entirely.
        rerank_model_func=(
            providers.get_reranker(
                model_id=settings.rerank_model,
                base_url=settings.rerank_binding_host,
                api_key=settings.rerank_binding_api_key,
                timeout=settings.rerank_timeout,
            )
            if settings.rerank_model and settings.rerank_binding_host
            else None
        ),
    )
    await rag.initialize_storages()
    return rag
