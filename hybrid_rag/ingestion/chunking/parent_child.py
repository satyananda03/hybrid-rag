"""Parent-child chunking: parents follow the document's structure, children split each parent.

Parents port the merge rule of docling's HybridChunker (``merge_peers``) onto the flat text
``separate_content`` produces. A *section* is a heading plus the text under it, up to
the next heading. Sections are never merged with each other and never overlap: a
section that fits ``max_tokens`` becomes one parent, a larger one is cut as docling's
``_split_by_doc_items`` cuts it: paragraphs packed greedily, then lines, sentences and
words for a paragraph that alone is too large.

Headings come from the two standard content-list signals, whichever parser produced
the list: ``text_level`` (persisted as ``heading_offsets``) and markdown ``#`` lines in
the text. What differs between documents is only what starts a new section regardless
of headings; that is what the layout chunkers define.

``ParentChildChunker`` is the entry point: it takes the parents from the document's
layout chunker (``chunker_for``) and splits each into children, with their page ranges.
"""

import bisect
import logging
import re
from typing import Any, NamedTuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag.config import settings
from hybrid_rag.core.chunk_schema import is_sheet
from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.core.tokenizer import Tokenizer
from hybrid_rag.ingestion.token_limit import _normalized_child_offsets

logger = logging.getLogger(__name__)

PARENT_MAX_TOKENS = settings.parent_chunk_size
CHILD_CHUNK_SIZE = settings.child_chunk_size
CHILD_CHUNK_OVERLAP = settings.child_chunk_overlap
_MARKDOWN_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(\S.*)$")

class Heading(NamedTuple):
    start: int
    end: int
    level: int
    text: str


class ParentSpan(NamedTuple):
    """``content[start:end]`` is the parent's text; ``headings`` its section path, outermost first."""

    start: int
    end: int
    headings: tuple[Heading, ...]


class DocumentChunker:
    """Prose that flows across pages: sections end only at headings.

    PDFs from any parser, .md, .txt and prebuilt content lists.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_tokens: int | None = None,
        *,
        page_offsets: list[dict[str, Any]] | None = None,
        heading_offsets: list[dict[str, int]] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens or PARENT_MAX_TOKENS
        self.page_offsets = page_offsets or []
        self.heading_offsets = heading_offsets or []
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator="end",
            length_function=self._tokens,
        )

    def hard_breaks(self) -> list[int]:
        """Positions that start a new section with an empty heading path, headings or not."""
        return []

    def split(self, content: str) -> list[ParentSpan]:
        """Parent spans in document order, whitespace-trimmed, covering all of ``content``'s text."""
        parents = (
            parent
            for section in self._sections(content, self._headings(content))
            for parent in self._fill(content, section)
        )
        return [span for span in (_trimmed(content, parent) for parent in parents) if span.start < span.end]

    def _tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _headings(self, content: str) -> dict[int, Heading]:
        """Headings by start position; a ``text_level`` heading wins over a ``#`` line at the same spot."""
        found = {
            match.start(): Heading(match.start(), match.end(), len(match.group(1)), match.group(2).strip())
            for match in _MARKDOWN_HEADING.finditer(content)
        }
        for offset in self.heading_offsets:
            start, end = offset["start"], offset["end"]
            found[start] = Heading(start, end, int(offset["level"]), content[start:end].lstrip("#").strip())
        return found

    def _sections(self, content: str, headings: dict[int, Heading]) -> list[ParentSpan]:
        """``content`` cut at every heading and hard break.

        A heading with no text of its own (``# BAB II`` right before ``## A``) leads the
        next section instead of standing alone; at a hard break it closes the previous
        section instead. Consecutive sections with the same heading path are one section,
        as docling compares heading paths, not heading occurrences.
        """
        breaks = set(self.hard_breaks())
        cuts = sorted({0, len(content), *headings, *breaks})

        sections: list[ParentSpan] = []
        path: tuple[Heading, ...] = ()
        lead: int | None = None  # start of headings still waiting for their text
        for start, end in zip(cuts, cuts[1:]):
            if start in breaks:
                path = ()
                if lead is not None:
                    _close_last(sections, lead, start)
                    lead = None
            heading = headings.get(start)
            if heading:
                path = tuple(h for h in path if h.level < heading.level) + (heading,)
            if not content[heading.end if heading else start:end].strip():
                lead = start if lead is None else lead
                continue
            continues_previous = (
                sections and lead is None and start not in breaks
                and _titles(sections[-1].headings) == _titles(path)
            )
            if continues_previous:
                sections[-1] = sections[-1]._replace(end=end)
            else:
                sections.append(ParentSpan(start if lead is None else lead, end, path))
            lead = None
        if lead is not None:
            _close_last(sections, lead, len(content))
        return sections

    def _fill(self, content: str, section: ParentSpan) -> list[ParentSpan]:
        """The section as one parent, or, over ``max_tokens``, cut at the coarsest separator that fits.

        Paragraphs (for Textract, one layout block each) are packed greedily, as docling's
        ``_split_by_doc_items`` packs doc items; a paragraph that alone is over ``max_tokens``
        falls back to lines, sentences and words.
        """
        if self._tokens(content[section.start:section.end]) <= self.max_tokens:
            return [section]
        parents: list[ParentSpan] = []
        cursor = section.start
        for piece in self._splitter.split_text(content[section.start:section.end]):
            start = content.find(piece, cursor, section.end)
            if start < 0:
                # Never drop text silently: the splitter only ever returns trimmed substrings.
                raise ValueError(f"Split piece not found in the source text after offset {cursor}")
            cursor = start + len(piece)
            parents.append(ParentSpan(start, cursor, section.headings))
        return parents


class SlideChunker(DocumentChunker):
    """VLM slide decks: every slide is exactly one parent, whatever its length or headings.

    A topic split across two parents makes retrieval return only half of it, so a slide
    is never cut, not at the ``#`` lines the VLM writes nor at ``max_tokens``. Its first
    heading, if any, is its section path.
    """

    def hard_breaks(self) -> list[int]:
        return [offset["start"] for offset in self.page_offsets]

    def _sections(self, content: str, headings: dict[int, Heading]) -> list[ParentSpan]:
        cuts = sorted({0, len(content), *self.hard_breaks()})
        return [
            ParentSpan(start, end, tuple(headings[pos] for pos in sorted(headings) if start <= pos < end)[:1])
            for start, end in zip(cuts, cuts[1:])
        ]

    def _fill(self, content: str, section: ParentSpan) -> list[ParentSpan]:
        return [section]


class SheetChunker(DocumentChunker):
    """Spreadsheets: nothing crosses into the next sheet.

    A sheet is a context its author chose (one province, one topic): a sheet A label
    next to a sheet B URL makes the LLM attach B's URL to A.
    """

    def hard_breaks(self) -> list[int]:
        return [offset["start"] for offset in self.page_offsets if is_sheet(offset["page"])]

#: Layout recorded by the parser router (``_with_layout``) → chunker.
CHUNKERS: dict[str, type[DocumentChunker]] = {
    "document": DocumentChunker,
    "slides": SlideChunker,
    "sheet": SheetChunker,
}

def chunker_for(
    tokenizer: Tokenizer,
    chunk_options: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> DocumentChunker:
    """The chunker for a document's stored ``chunk_options`` (see ``separate_content``).

    Documents enqueued before the layout was recorded fall back on their pages: sheet
    names mean a spreadsheet, slide images a slide deck, anything else a document.
    """
    options = chunk_options or {}
    page_offsets = options.get("page_offsets") or []
    layout = options.get("layout") or _layout_from_pages(page_offsets)
    chunker_cls = CHUNKERS.get(layout)
    if chunker_cls is None:
        logger.warning(f"[chunking] layout {layout!r} tidak dikenal; dipotong sebagai dokumen")
        chunker_cls = DocumentChunker
    return chunker_cls(
        tokenizer,
        max_tokens,
        page_offsets=page_offsets,
        heading_offsets=options.get("heading_offsets"),
    )


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
        self, content: str, chunk_options: dict[str, Any] | None = None
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
        chunker = chunker_for(self.tokenizer, chunk_options, self.parent_max_tokens)

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
            parent_chunks.append(parent)
            child_cursor = 0

            # Split parent into children
            child_texts = self._child_splitter.split_text(parent_text_stripped)

            for child_text in child_texts:
                child_text_stripped = child_text.strip()
                if not child_text_stripped:
                    continue

                child_tokens = len(self.tokenizer.encode(child_text_stripped))
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


def _layout_from_pages(page_offsets: list[dict[str, Any]]) -> str:
    if any(is_sheet(offset["page"]) for offset in page_offsets):
        return "sheet"
    if any(offset.get("image") for offset in page_offsets):
        return "slides"
    return "document"


def _titles(path: tuple[Heading, ...]) -> tuple[str, ...]:
    return tuple(heading.text for heading in path)


def _close_last(sections: list[ParentSpan], lead: int, end: int) -> None:
    """Headings with no text before a break or the end close the last section, or stand alone."""
    if sections:
        sections[-1] = sections[-1]._replace(end=end)
    else:
        sections.append(ParentSpan(lead, end, ()))


def _trimmed(content: str, span: ParentSpan) -> ParentSpan:
    text = content[span.start:span.end]
    return span._replace(
        start=span.start + len(text) - len(text.lstrip()),
        end=span.end - (len(text) - len(text.rstrip())),
    )


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
