"""The layout chunker for spreadsheets."""
from hybrid_rag.core.chunk_schema import is_sheet
from hybrid_rag.ingestion.chunking.parent_child_chunker.document import DocumentChunker


class SheetChunker(DocumentChunker):
    """Spreadsheets: nothing crosses into the next sheet.

    A sheet is a context its author chose (one province, one topic): a sheet A label
    next to a sheet B URL makes the LLM attach B's URL to A.
    """

    def hard_breaks(self) -> list[int]:
        return [offset["start"] for offset in self.page_offsets if is_sheet(offset["page"])]
