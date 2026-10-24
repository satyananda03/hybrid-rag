import bisect
from typing import Any

from hybrid_rag.config import settings
from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.core.tokenizer import Tokenizer
from hybrid_rag.ingestion.chunking.structure import chunker_for
from hybrid_rag.ingestion.token_limit import _normalized_child_offsets
import logging

logger = logging.getLogger(__name__)


CHILD_CHUNK_SIZE = settings.child_chunk_size
CHILD_CHUNK_OVERLAP = settings.child_chunk_overlap

def _locate(haystack: str, needle: str, search_from: int) -> tuple[int, int] | None:
    """Span ``needle`` di ``haystack`` sejak ``search_from``; verbatim dulu, lalu abaikan spasi."""
    pos = haystack.find(needle, search_from)
    if pos >= 0:
        return pos, pos + len(needle)
    return _normalized_child_offsets(haystack, needle, search_from)


def _page_range(page_offsets: list[dict[str, Any]], starts: list[int], span: tuple[int, int]) -> dict[str, Any]:
    """Halaman awal/akhir untuk satu span karakter; ``starts`` = ``[o["start"]]`` terurut.

    Plus ``images`` — gambar slide di rentang itu — bila offset-nya membawa ``image``.
    """
    first = max(0, bisect.bisect_right(starts, span[0]) - 1)
    last = max(0, bisect.bisect_right(starts, max(span[0], span[1] - 1)) - 1)
    pages = {"page_start": page_offsets[first]["page"], "page_end": page_offsets[last]["page"]}
    images = list(dict.fromkeys(o["image"] for o in page_offsets[first:last + 1] if o.get("image")))
    return {**pages, "images": images} if images else pages


def chunking_parent_child(
    tokenizer: Tokenizer,
    content: str,
    *,
    chunk_options: dict[str, Any] | None = None,
    parent_max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split content into parent chunks along its structure, then sub-split each parent into children.

    Parents come from the document's chunker (``structure.chunker_for``): one section
    per parent, cut between paragraphs only when the section is over
    ``parent_max_tokens``. Children are LangChain's RecursiveCharacterTextSplitter
    over each parent's text.

    ``chunk_options`` is what ``separate_content`` stored for ``content``:
    ``page_offsets`` (``[{"page", "start"}]``) only add ``sidecar={"page_start",
    "page_end"}`` to each chunk without touching its ``content``, so embedding and
    retrieval see no difference; ``heading_offsets`` and ``layout`` pick the parent
    boundaries. A child whose position is not found (it should never be: the splitter
    only strips) is left without a ``sidecar``.

    Returns:
        parent_chunks: list of ``{tokens, content, chunk_order_index[, heading][, sidecar]}``
        child_chunks:  list of ``{tokens, content, chunk_order_index, parent_index[, sidecar]}``

    ``heading`` (``{"level", "heading", "parent_headings"}``) is the parent's section
    path, read by entity extraction; children carry none, so what retrieval shows is
    unchanged. A child points at its parent through ``parent_index``; that is the
    only link between the two levels. Do not return a list-position map —
    ``enforce_chunk_token_limit_before_embedding`` splits children after this step,
    so any map built here is stale before it is used.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunker = chunker_for(tokenizer, chunk_options, parent_max_tokens)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_chunks: list[dict[str, Any]] = []
    child_chunks: list[dict[str, Any]] = []

    child_global_index = 0
    page_offsets = (chunk_options or {}).get("page_offsets") or []
    page_starts = [o["start"] for o in page_offsets]
    unlocated = 0

    for parent_idx, parent_span in enumerate(chunker.split(content)):
        parent_text_stripped = content[parent_span.start:parent_span.end]
        parent: dict[str, Any] = {
            "tokens": len(tokenizer.encode(parent_text_stripped)),
            "content": parent_text_stripped,
            "chunk_order_index": parent_idx,
        }
        if parent_span.headings:
            *outer, own = parent_span.headings
            parent["heading"] = {
                "level": own.level,
                "heading": own.text,
                "parent_headings": [h.text for h in outer],
            }
        if page_starts:
            # Rentang halaman parent melebar melewati tiap child-nya; inilah
            # yang dipakai laporan untuk merender halaman asli saat bukti
            # yang dikembalikan adalah teks parent.
            parent["sidecar"] = _page_range(page_offsets, page_starts, (parent_span.start, parent_span.end))
        parent_chunks.append(parent)
        child_cursor = 0

        # Split parent into children
        child_texts = child_splitter.split_text(parent_text_stripped)

        for child_text in child_texts:
            child_text_stripped = child_text.strip()
            if not child_text_stripped:
                continue

            child_tokens = len(tokenizer.encode(child_text_stripped))
            child: dict[str, Any] = {
                "tokens": child_tokens,
                "content": child_text_stripped,
                "chunk_order_index": child_global_index,
                "parent_index": parent_idx,
            }
            local = _locate(parent_text_stripped, child_text_stripped, child_cursor)
            if local is not None:
                child_cursor = local[0] + 1
                span = (parent_span.start + local[0], parent_span.start + local[1])
                if page_starts:
                    child["sidecar"] = _page_range(page_offsets, page_starts, span)
            else:
                unlocated += 1
            child_chunks.append(child)
            child_global_index += 1

    if unlocated:
        logger.warning(f"[parent-child] {unlocated} child tanpa posisi sumber; halaman tidak tercatat untuk mereka")

    logger.info(
        f"[parent-child] Split into {len(parent_chunks)} parents, "
        f"{len(child_chunks)} children "
        f"({type(chunker).__name__}, parent_max_tokens={chunker.max_tokens}, child_size={CHILD_CHUNK_SIZE})"
    )

    return parent_chunks, child_chunks

def redirect_source_ids(
    chunk_results: list,
    parent_to_child_map: dict[str, list[str]],
) -> None:
    """In-place replace source_id in extraction results from parent → child keys.

    After entity/relation extraction runs on parent chunks, each entity's
    source_id points to a parent chunk_key. This function redirects those
    to the child chunk_keys so that graph retrieval returns child chunks.

    Args:
        chunk_results: List of (maybe_nodes, maybe_edges) from _process_extract_entities.
        parent_to_child_map: ``{parent_chunk_key: [child_chunk_keys]}``, dibangun
            di worker dari ``parent_index`` tiap child (lihat pipeline.py).
    """
    redirected_count = 0

    for maybe_nodes, maybe_edges in chunk_results:
        for _entity_name, entities in maybe_nodes.items():
            for entity in entities:
                source_id = entity.get("source_id", "")
                if source_id in parent_to_child_map:
                    entity["source_id"] = GRAPH_FIELD_SEP.join(
                        parent_to_child_map[source_id]
                    )
                    redirected_count += 1

        for _edge_key, edges in maybe_edges.items():
            for edge in edges:
                source_id = edge.get("source_id", "")
                if source_id in parent_to_child_map:
                    edge["source_id"] = GRAPH_FIELD_SEP.join(
                        parent_to_child_map[source_id]
                    )
                    redirected_count += 1

    logger.info(f"[parent-child] Redirected {redirected_count} source_ids")