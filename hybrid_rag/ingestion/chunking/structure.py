"""Parent boundaries that follow the document's structure.

Ports the merge rule of docling's HybridChunker (``merge_peers``) onto the flat text
``separate_content`` produces. A *section* is a heading plus the text under it, up to
the next heading. Sections are never merged with each other and never overlap: a
section that fits ``max_tokens`` becomes one parent, a larger one is cut as docling's
``_split_by_doc_items`` cuts it: paragraphs packed greedily, then lines, sentences and
words for a paragraph that alone is too large.

Headings come from the two standard content-list signals, whichever parser produced
the list: ``text_level`` (persisted as ``heading_offsets``) and markdown ``#`` lines in
the text. What differs between documents is only what starts a new section regardless
of headings; that is what the subclasses define.
"""

import logging
import re
from typing import Any, NamedTuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag.config import settings
from hybrid_rag.core.chunk_schema import is_sheet
from hybrid_rag.core.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

PARENT_MAX_TOKENS = settings.parent_max_tokens

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
