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


import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from hybrid_rag.retrieval.types import QueryParam
from hybrid_rag.retrieval.rag import RetrievalRAG
from hybrid_rag.retrieval.workspace import workspace_pool
import logging

logger = logging.getLogger(__name__)

# Chunks cite their images as "Image Path: /abs/path.jpg"; pull those out of the
# retrieved context so a caller can attach the actual files to its response.
IMAGE_PATH_PATTERN = re.compile(
    r"Image Path:\s*([^\r\n]+?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif)(?:\?[^\r\n]*)?)"
)


@dataclass
class RetrievalResult:
    context: str
    """Raw retrieved context (chunks, entities, relations)."""
    image_paths: List[str] = field(default_factory=list)
    """Image paths found in the retrieved chunks."""
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

    image_paths = IMAGE_PATH_PATTERN.findall(context)
    logger.info(f"[retrieve] context_len={len(context)} images_found={len(image_paths)}")
    return RetrievalResult(context=context, image_paths=image_paths, raw_data=raw_data)


async def retrieve_context(
    query: str,
    *,
    workspace: str,
    param: Optional[QueryParam] = None,
    include_raw_data: bool = False,
    return_parent: bool = False,
) -> RetrievalResult:
    """Retrieve context for ``query`` from ``workspace``'s documents only.

    ``workspace`` is required, never defaulted: a shared fallback instance would
    silently cross the workspace boundary.

    ``return_parent`` menukar isi tiap chunk dengan teks parent-nya — potongan
    2500 karakter yang menaungi child pemenang rerank, untuk pemanggil yang butuh
    bukti utuh (Inspector/Judge) dan bukan sekadar kalimat yang paling mirip.
    Pencarian tetap berjalan di atas child; sitasi tetap menunjuk child. Hanya
    berlaku untuk dokumen yang di-ingest dengan ``save_parent=True``; sisanya
    mengembalikan teks child apa adanya.
    """
    param = param or QueryParam()
    if return_parent and not param.return_parent:
        # replace, bukan mutasi: ``param`` milik pemanggil dan sering dipakai ulang.
        param = replace(param, return_parent=True)
    async with workspace_pool.lease(workspace) as rag:
        return await _retrieve(rag, query, param, include_raw_data)


async def retrieve_context_without_relations(
    query: str,
    *,
    workspace: str,
    param: Optional[QueryParam] = None,
    return_parent: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieval graf+vektor tetap jalan (mode ``mix`` memilih chunk lewat KG); yang dikembalikan
    hanya ``data.chunks`` berperingkat — tanpa blok entitas/relasi dan tanpa ``context`` teks.

    ``return_parent`` seperti di :func:`retrieve_context`: isi chunk diganti teks
    parent, sitasi tetap milik child."""
    result = await retrieve_context(query, workspace=workspace, param=param,
                                    include_raw_data=True, return_parent=return_parent)
    return list(((result.raw_data or {}).get("data") or {}).get("chunks") or [])
