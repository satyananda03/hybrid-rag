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
``retrieve_context_cross_workspaces`` runs that per workspace and ranks the chunks
of all of them as one list.
"""
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import zip_longest
from typing import Any, Dict, List, Optional
from hybrid_rag.config import settings
from hybrid_rag.retrieval.context import _references_block, _render_chunks, _source
from hybrid_rag.retrieval.context_template import TEMPLATES
from hybrid_rag.retrieval.query import extract_keywords_only
from hybrid_rag.retrieval.types import QueryParam
from hybrid_rag.retrieval.rag import RetrievalRAG
from hybrid_rag.retrieval.utils import generate_reference_list_from_chunks
from hybrid_rag.retrieval.workspace import workspace_pool
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    context: str
    """Raw retrieved context (chunks, entities, relations)."""
    images: List[Dict[str, str]] = field(default_factory=list)
    """Images of the retrieved chunks, in context order: ``{"path", "reference_id"}``,
    so each can be shown next to its citation ``[reference_id]``. Image chunks come
    first, then each slide of a slide deck once. ``reference_id`` is
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

    images = _images((result.get("data") or {}).get("chunks") or [])
    logger.info(f"[retrieve] context_len={len(context)} images_found={len(images)}")
    return RetrievalResult(context=context, images=images, raw_data=raw_data)


def _images(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """``RetrievalResult.images`` of ``chunks``, in context order."""
    def cited(path: str, chunk: dict) -> Dict[str, str]:
        return {"path": path, **({"reference_id": chunk["reference_id"]} if chunk.get("reference_id") else {})}

    images = [cited(chunk["image_path"], chunk) for chunk in chunks if chunk.get("image_path")]
    # Gambar slide: satu slide bisa terbawa banyak chunk, jadi sekali saja, dengan sitasi chunk pertamanya.
    slides: Dict[str, Dict[str, str]] = {}
    for chunk in chunks:
        for path in chunk.get("images") or []:
            slides.setdefault(path, cited(path, chunk))
    return images + list(slides.values())

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


async def retrieve_context_cross_workspaces(
    query: str,
    *,
    workspaces: Sequence[str],
    labels: Optional[Mapping[str, str]] = None,
    param: Optional[QueryParam] = None,
) -> RetrievalResult:
    """Retrieve chunks from several workspaces and rank them as one list (chunks only, no KG block).

    Each workspace runs :func:`retrieve_context_without_relations` on its own, so isolation, KG chunk
    selection, BM25 and rerank are exactly those of a single-workspace query. The results are then
    ordered by rerank score: the reranker scores each (query, chunk) pair independently, so this is the
    order one rerank over every candidate would give, without a second rerank call. The final cut is
    the single-workspace one (``chunk_top_k``, or ``parent_top_k`` with ``return_parent``).

    Keywords are extracted once and passed to every workspace (``QueryParam.keywords``), so all of them
    are searched with the same reading of the query. The extraction is cached in the alphabetically
    first workspace, so the same query hits the cache whatever order the workspaces come in.

    ``labels`` maps a workspace to the text appended to each source after a comma (default: the workspace name),
    so a citation tells which workspace its document belongs to.
    """
    if not workspaces:
        raise ValueError("workspaces must not be empty")
    param = param or QueryParam()
    labels = labels or {}
    if param.keywords is None:
        async with workspace_pool.lease(min(workspaces)) as rag:
            keywords = await extract_keywords_only(
                query.strip(), param, rag.text_chunks.global_config, rag.llm_response_cache
            )
        param = replace(param, keywords=keywords)
    results = await asyncio.gather(
        *(retrieve_context_without_relations(query, workspace=workspace, param=param) for workspace in workspaces)
    )
    limit = settings.parent_top_k if param.return_parent else param.chunk_top_k
    chunks = _rank_across(results, [labels.get(workspace, workspace) for workspace in workspaces])[:limit]
    if not chunks:
        return RetrievalResult(context="")

    sources: Dict[str, str] = {}
    if param.include_citations:
        references, chunks = generate_reference_list_from_chunks(chunks, param.citation_style)
        ref_labels: Dict[str, Dict[str, None]] = {}
        for chunk in chunks:
            ref_labels.setdefault(chunk["reference_id"], {}).update(dict.fromkeys(chunk["labels"]))
        sources = {
            ref["reference_id"]: f"{_source(ref)}, {', '.join(ref_labels[ref['reference_id']])}"
            for ref in references
        }
    context = TEMPLATES["chunk_only_context"].format(
        text_chunks_str=_render_chunks(chunks, sources), references_block=_references_block(sources)
    )
    logger.info(f"[retrieve_cross] workspaces={len(workspaces)} chunks={len(chunks)} context_len={len(context)}")
    return RetrievalResult(context=context, images=_images(chunks))


def _rank_across(results: Sequence[List[Dict[str, Any]]], labels: Sequence[str]) -> List[Dict[str, Any]]:
    """Merge per-workspace ranked chunks into one list; each chunk gets the ``labels`` it was found under.

    The base order is round-robin (every workspace's first chunk, then every second, ...). When every
    chunk carries a rerank score, a stable sort by score refines it; when any lacks one (rerank off or
    failed), round-robin stands, since retrieval scores of different workspaces are not comparable.
    The same text of the same file found in several workspaces (one file uploaded to each) is kept
    once, at its best rank, with all its labels.
    """
    ranked = [
        {**chunk, "labels": [label]}
        for row in zip_longest(*results)
        for chunk, label in zip(row, labels)
        if chunk is not None
    ]
    if all("rerank_score" in chunk for chunk in ranked):
        ranked.sort(key=lambda chunk: -chunk["rerank_score"])
    merged: Dict[tuple, Dict[str, Any]] = {}
    for chunk in ranked:
        key = (chunk.get("file_path"), chunk["content"])
        if key in merged:
            merged[key]["labels"] += chunk["labels"]
        else:
            merged[key] = chunk
    return list(merged.values())
