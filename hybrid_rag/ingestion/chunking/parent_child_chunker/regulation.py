"""The layout chunker for regulations; what a regulation is comes from ``legal_chunker``."""
from typing import Any
from hybrid_rag.ingestion.chunking.legal_chunker.pipeline import Regulation
from hybrid_rag.ingestion.chunking.parent_child_chunker.document import DocumentChunker
from hybrid_rag.ingestion.chunking.parent_child_chunker.spans import Heading, ParentSpan


class IndonesianPolicyChunker(DocumentChunker):
    """Regulations: one parent per Pasal, whatever pages it crosses, each citable by its Pasal.

    Only the articles change. The text before them and the signature block, Penjelasan and
    Lampiran after them are cut as in any document, and are cited by the file name.

    ponytail: BAB/Bagian/Paragraf lines stay at the tail of the Pasal before them. Move them
    to the Pasal they introduce if that hurts retrieval.
    """

    def __init__(self, *args: Any, regulation: Regulation, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._regulation = regulation
        self._citations = {article.start: regulation.citation(article) for article in regulation.articles}

    def hard_breaks(self) -> list[int]:
        return [self._regulation.end]

    def citation_of(self, parent: ParentSpan) -> str:
        return self._citations.get(parent.headings[-1].start, "") if parent.headings else ""

    def _headings(self, content: str) -> dict[int, Heading]:
        """Inside the articles only a Pasal cuts; outside them, the document's own headings."""
        body_start = self._regulation.articles[0].start
        outside = {
            position: heading
            for position, heading in super()._headings(content).items()
            if not body_start <= position < self._regulation.end
        }
        return outside | {
            article.start: Heading(article.start, article.end, 1, self._citations[article.start])
            for article in self._regulation.articles
        }
