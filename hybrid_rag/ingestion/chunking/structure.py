"""Parent boundaries that follow the document's structure.

Ports the merge rule of docling's HybridChunker (``merge_peers``) onto the flat text
``separate_content`` produces. A *section* is a heading plus the text under it, up to
the next heading. Sections are never merged with each other and never overlap: a
section that fits ``max_tokens`` becomes one parent, a larger one is cut between its
paragraphs.

One rule goes beyond docling: when a section has to be cut, the cut backs off to the
cleanest paragraph boundary inside the parent being filled (see ``_cut_quality``).
Parsers emit text per block and per page, so a block that ends without ``.?!`` may be
a sentence running on into the next page, and one ending in ``:`` is a list intro.
Cutting there would hand the LLM half a sentence, or a list without its intro, so
such boundaries are only cut when the parent offers no other.

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
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_SENTENCE_END = re.compile(r"[.?!…][\"'”’)\]]*$")
_LIST_ITEM = re.compile(r"([a-z]{1,2}[.)]|\d+[.)]|[-•*])\s")


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
        self._sentence_splitter = RecursiveCharacterTextSplitter(
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
        headings = self._headings(content)
        heading_ends = {heading.end for heading in headings.values()}
        parents = (
            parent
            for section in self._sections(content, headings)
            for parent in self._fill(content, section, heading_ends)
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

    def _fill(self, content: str, section: ParentSpan, heading_ends: set[int]) -> list[ParentSpan]:
        """The section as one parent, or, over ``max_tokens``, runs of its paragraphs.

        Each run takes as many paragraphs as fit, then backs off to its cleanest
        boundary; a paragraph that alone is over ``max_tokens`` is cut on its own.
        """
        if self._tokens(content[section.start:section.end]) <= self.max_tokens:
            return [section]
        paragraphs = _paragraphs(content, section.start, section.end)
        parents: list[ParentSpan] = []
        first = 0
        while first < len(paragraphs):
            if self._tokens(content[slice(*paragraphs[first])]) > self.max_tokens:
                parents += [ParentSpan(a, b, section.headings) for a, b in self._pieces(content, *paragraphs[first])]
                first += 1
                continue
            last = first
            while last + 1 < len(paragraphs) and self._tokens(
                content[paragraphs[first][0]:paragraphs[last + 1][1]]
            ) <= self.max_tokens:
                last += 1
            if last + 1 < len(paragraphs):
                last = max(
                    range(first, last + 1),
                    key=lambda k: (_cut_quality(content, paragraphs[k], paragraphs[k + 1], heading_ends), k),
                )
            parents.append(ParentSpan(paragraphs[first][0], paragraphs[last][1], section.headings))
            first = last + 1
        return parents

    def _pieces(self, content: str, start: int, end: int) -> list[tuple[int, int]]:
        """One paragraph over ``max_tokens``, cut at the coarsest separator that fits (line, sentence, word)."""
        spans: list[tuple[int, int]] = []
        cursor = start
        for piece in self._sentence_splitter.split_text(content[start:end]):
            pos = content.find(piece, cursor, end)
            if pos < 0:
                # Never drop text silently: the splitter only ever returns trimmed substrings.
                raise ValueError(f"Split piece not found in the source text after offset {cursor}")
            spans.append((pos, pos + len(piece)))
            cursor = pos + len(piece)
        return spans


class SlideChunker(DocumentChunker):
    """VLM slide decks: every slide is its own section, titled or not."""

    def hard_breaks(self) -> list[int]:
        return [offset["start"] for offset in self.page_offsets]


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


def _paragraphs(content: str, start: int, end: int) -> list[tuple[int, int]]:
    """Non-blank paragraphs of ``content[start:end]`` — for Textract, one layout block each."""
    paragraphs: list[tuple[int, int]] = []
    pos = start
    for paragraph_break in [*_PARAGRAPH_BREAK.finditer(content, start, end), None]:
        stop = paragraph_break.start() if paragraph_break else end
        if content[pos:stop].strip():
            paragraphs.append((pos, stop))
        if paragraph_break:
            pos = paragraph_break.end()
    return paragraphs


def _cut_quality(content: str, before: tuple[int, int], after: tuple[int, int], heading_ends: set[int]) -> int:
    """How clean a cut between two paragraphs is; higher is cleaner.

    2 after a finished sentence; 1 between plain blocks (list items, a title and its
    text); 0 after a list intro (``:``) or before a lowercase run-on, a sentence broken
    by a page; -1 right after a heading, which would leave the heading without its text.
    """
    text = content[slice(*before)].rstrip()
    following = content[slice(*after)].lstrip()
    if before[0] + len(text) in heading_ends:
        return -1
    if _SENTENCE_END.search(text):
        return 2
    if text.endswith(":") or (following[:1].islower() and not _LIST_ITEM.match(following)):
        return 0
    return 1


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
