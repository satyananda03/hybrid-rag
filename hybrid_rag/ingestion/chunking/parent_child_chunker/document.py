"""The default layout chunker, and the rule the other layouts build on."""
import re
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hybrid_rag.config import settings
from hybrid_rag.core.tokenizer import Tokenizer
from hybrid_rag.ingestion.chunking.parent_child_chunker.spans import Heading, ParentSpan

PARENT_MAX_TOKENS = settings.parent_chunk_size
_MARKDOWN_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(\S.*)$")


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

    def citation_of(self, parent: ParentSpan) -> str:
        """How the document cites ``parent``, stored on its chunks; empty when only the file name can."""
        return ""

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
