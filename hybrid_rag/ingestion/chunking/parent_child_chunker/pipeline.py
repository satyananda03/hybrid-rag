"""Connects a layout chunker's parents to their children. The only module that knows how the levels fit together."""
import bisect
import logging
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hybrid_rag.config import settings
from hybrid_rag.core.tokenizer import Tokenizer
from hybrid_rag.ingestion.token_limit import _normalized_child_offsets
from hybrid_rag.ingestion.chunking.parent_child_chunker.selector import chunker_for

logger = logging.getLogger(__name__)

CHILD_CHUNK_SIZE = settings.child_chunk_size
CHILD_CHUNK_OVERLAP = settings.child_chunk_overlap


class ParentChildChunker:
    """Splits a document into parents along its layout, then each parent into children.

    Parents come from the layout chunker (``chunker_for``); this class only adds
    the child level and the page ranges. One instance can split any number of documents.
    """

    def __init__(self, tokenizer: Tokenizer, *, parent_max_tokens: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.parent_max_tokens = parent_max_tokens
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(
        self,
        content: str,
        chunk_options: dict[str, Any] | None = None,
        document_name: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parent and child chunks of ``content``, in document order.

        One section per parent, cut only when the section is over ``parent_max_tokens``.
        Children are LangChain's RecursiveCharacterTextSplitter over each parent's text.

        ``chunk_options`` is what ``separate_content`` stored for ``content``:
        ``page_offsets`` (``[{"page", "start"}]``) only add ``sidecar={"page_start",
        "page_end"}`` to each chunk without touching its ``content``, so embedding and
        retrieval see no difference; ``heading_offsets`` and ``layout`` pick the parent
        boundaries. A child whose position is not found (it should never be: the splitter
        only strips) is left without a ``sidecar``.

        ``document_name`` is the file the document came from. When the chunker gives a
        parent a citation (``citation_of``), it is stored as ``sidecar["citation"]`` on the
        parent and its children and put in front of each child's text: search reads only
        the text, and a workspace holds several regulations that all have a Pasal 5.

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
        chunker = chunker_for(self.tokenizer, content, chunk_options, self.parent_max_tokens, document_name)

        parent_chunks: list[dict[str, Any]] = []
        child_chunks: list[dict[str, Any]] = []

        child_global_index = 0
        page_offsets = (chunk_options or {}).get("page_offsets") or []
        page_starts = [o["start"] for o in page_offsets]
        unlocated = 0

        for parent_idx, parent_span in enumerate(chunker.split(content)):
            parent_text_stripped = content[parent_span.start:parent_span.end]
            parent: dict[str, Any] = {
                "tokens": len(self.tokenizer.encode(parent_text_stripped)),
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
            citation = chunker.citation_of(parent_span)
            if citation:
                parent.setdefault("sidecar", {})["citation"] = citation
            parent_chunks.append(parent)
            child_cursor = 0

            # Split parent into children
            child_texts = self._child_splitter.split_text(parent_text_stripped)

            for child_text in child_texts:
                child_text_stripped = child_text.strip()
                if not child_text_stripped:
                    continue

                child_content = f"{citation}: {child_text_stripped}" if citation else child_text_stripped
                child: dict[str, Any] = {
                    "tokens": len(self.tokenizer.encode(child_content)),
                    "content": child_content,
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
                if citation:
                    child.setdefault("sidecar", {})["citation"] = citation
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
