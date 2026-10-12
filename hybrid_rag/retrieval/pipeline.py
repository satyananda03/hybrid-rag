"""Retrieval entry point.

Retrieval takes a text query and stops at the context. Nothing here generates
an answer — the caller assembles its own prompt from what comes back, so the
generator model, its prompt and its history live outside this module.

Attachments on the *question* (an image to ask about, a table to compare
against) are also the caller's job: describe them however you like and fold the
description into the query string. Multimodal input only reaches this module
through ingestion, never through retrieval.

``retrieve_context`` leases the workspace's :class:`RetrievalRAG` from the read
pool for the duration of the query, so retrieval reads exactly the documents
that workspace ingested — see :mod:`hybrid_rag.retrieval.workspace`.
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hybrid_rag.retrieval.types import QueryParam
from hybrid_rag.retrieval.rag import RetrievalRAG
from hybrid_rag.retrieval.workspace import workspace_pool
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    context: str
    """Raw retrieved context (chunks, entities, relations)."""
    images: List[Dict[str, str]] = field(default_factory=list)
    """Images of the retrieved chunks, in context order: ``{"path", "reference_id"}``,
    so each can be shown next to its citation ``[reference_id]``. ``reference_id`` is
    absent when ``QueryParam.include_citations`` is off."""
    raw_data: Optional[Dict[str, Any]] = None
    """Structured entities / relationships / chunks, when requested."""


async def _retrieve(
    rag: RetrievalRAG,
    query: str,
    param: QueryParam,
    include_raw_data: bool,
) -> RetrievalResult:
    """Run retrieval and unpack the result."""
    result = await rag.aretrieve(query, param=param)

    context = result.get("context", "") or ""
    raw_data = (
        {k: v for k, v in result.items() if k != "context"}
        if include_raw_data
        else None
    )

    images = [
        {"path": chunk["image_path"], **({"reference_id": chunk["reference_id"]} if chunk.get("reference_id") else {})}
        for chunk in (result.get("data") or {}).get("chunks") or []
        if chunk.get("image_path")
    ]
    logger.info(f"[retrieve] context_len={len(context)} images_found={len(images)}")
    return RetrievalResult(context=context, images=images, raw_data=raw_data)


async def retrieve_context(
    query: str,
    *,
    workspace: str,
    param: Optional[QueryParam] = None,
    include_raw_data: bool = False,
) -> RetrievalResult:
    """Retrieve context for ``query`` from ``workspace``'s documents only.

    ``workspace`` is required, never defaulted: a shared fallback instance would
    silently cross the workspace boundary. Per-query options (``return_parent``,
    ``include_citations``, budgets) all live on :class:`QueryParam`.
    """
    param = param or QueryParam()
    async with workspace_pool.lease(workspace) as rag:
        return await _retrieve(rag, query, param, include_raw_data)


async def retrieve_context_without_relations(
    query: str,
    *,
    workspace: str,
    param: Optional[QueryParam] = None,
) -> List[Dict[str, Any]]:
    """Retrieval graf+vektor tetap jalan (mode ``mix`` memilih chunk lewat KG); yang dikembalikan
    hanya ``data.chunks`` berperingkat — tanpa blok entitas/relasi dan tanpa ``context`` teks."""
    result = await retrieve_context(query, workspace=workspace, param=param, include_raw_data=True)
    return list(((result.raw_data or {}).get("data") or {}).get("chunks") or [])
