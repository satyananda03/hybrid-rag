"""Build a ready-to-use :class:`IngestionRAG` for one workspace.

Workspace isolation lives in ``workspace`` — it is forwarded to every storage, so
two instances built with different workspaces share the database but never each
other's data. Building one instance per workspace is what
:mod:`hybrid_rag.ingestion.workspace` does.

Models reach their clients through :mod:`hybrid_rag.providers`; nothing here
builds a provider client of its own.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from hybrid_rag.config import settings
from hybrid_rag.core.embedding import build_embedding_func
from hybrid_rag.ingestion.rag import IngestionRAG

logger = logging.getLogger(__name__)


async def build_rag(workspace: str) -> IngestionRAG:
    """Construct an ``IngestionRAG`` for ``workspace`` and bring its storages up.

    The caller owns the returned instance and is responsible for
    ``await rag.finalize_storages()`` on shutdown.
    """
    settings.require("extract", "contextualize", "embedding")
    rag = IngestionRAG(
        workspace=workspace,
        llm_model_max_async=settings.extract_llm_max_async,
        enable_llm_cache=settings.enable_llm_cache,
        embedding_func=build_embedding_func(),
    )
    await rag.initialize_storages()
    return rag


@asynccontextmanager
async def build_pipeline(workspace: str) -> AsyncIterator[IngestionRAG]:
    """Async context manager yielding a ready ``IngestionRAG``, finalized on exit."""
    rag = await build_rag(workspace)
    try:
        yield rag
    finally:
        try:
            await rag.finalize_storages()
        except Exception:
            logger.debug("finalize_storages raised on shutdown", exc_info=True)
