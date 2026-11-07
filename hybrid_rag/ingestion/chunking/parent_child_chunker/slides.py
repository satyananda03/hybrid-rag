"""The layout chunker for slide decks."""
from hybrid_rag.ingestion.chunking.parent_child_chunker.document import DocumentChunker
from hybrid_rag.ingestion.chunking.parent_child_chunker.spans import Heading, ParentSpan


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
