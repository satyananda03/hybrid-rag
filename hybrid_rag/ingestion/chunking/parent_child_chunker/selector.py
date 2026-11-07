"""Picks the layout chunker for a document."""
import logging
from typing import Any
from hybrid_rag.core.chunk_schema import is_sheet
from hybrid_rag.core.tokenizer import Tokenizer
from hybrid_rag.ingestion.chunking.legal_chunker.pipeline import analyze_regulation
from hybrid_rag.ingestion.chunking.parent_child_chunker.document import DocumentChunker
from hybrid_rag.ingestion.chunking.parent_child_chunker.regulation import IndonesianPolicyChunker
from hybrid_rag.ingestion.chunking.parent_child_chunker.sheet import SheetChunker
from hybrid_rag.ingestion.chunking.parent_child_chunker.slides import SlideChunker

logger = logging.getLogger(__name__)

#: Layout recorded by the parser router (``_with_layout``) → chunker.
CHUNKERS: dict[str, type[DocumentChunker]] = {
    "document": DocumentChunker,
    "slides": SlideChunker,
    "sheet": SheetChunker,
}

def chunker_for(
    tokenizer: Tokenizer,
    content: str,
    chunk_options: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    document_name: str | None = None,
) -> DocumentChunker:
    """The chunker for ``content`` and its stored ``chunk_options`` (see ``separate_content``).

    Documents enqueued before the layout was recorded fall back on their pages: sheet
    names mean a spreadsheet, slide images a slide deck, anything else a document. A
    document that follows the regulation pattern gets ``IndonesianPolicyChunker``: only
    its text can tell, so this is decided here and not in the parser router.
    """
    options = chunk_options or {}
    page_offsets = options.get("page_offsets") or []
    layout = options.get("layout") or _layout_from_pages(page_offsets)
    chunker_cls = CHUNKERS.get(layout)
    if chunker_cls is None:
        logger.warning(f"[chunking] layout {layout!r} tidak dikenal; dipotong sebagai dokumen")
        chunker_cls = DocumentChunker
    layout_options = {"page_offsets": page_offsets, "heading_offsets": options.get("heading_offsets")}
    if chunker_cls is DocumentChunker and (regulation := analyze_regulation(content, document_name)):
        logger.info(f"[chunking] pola peraturan: {len(regulation.articles)} pasal, nama={regulation.name!r}")
        return IndonesianPolicyChunker(tokenizer, max_tokens, **layout_options, regulation=regulation)
    return chunker_cls(tokenizer, max_tokens, **layout_options)


def _layout_from_pages(page_offsets: list[dict[str, Any]]) -> str:
    if any(is_sheet(offset["page"]) for offset in page_offsets):
        return "sheet"
    if any(offset.get("image") for offset in page_offsets):
        return "slides"
    return "document"
