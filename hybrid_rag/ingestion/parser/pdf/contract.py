"""What every PDF document parser returns: the content list the chunker and pipeline rely on.

PDFs (and office files converted to PDF) that are not slide decks go through one parser,
chosen in ``parser/router.py``. Amazon Textract (``amazon_textract.parse_pdf``) is the
current one; LlamaParse or LiteParse plug in by adding ``parse_pdf`` in their own package
with this same signature and output. Chunkers follow the document layout, not the parser,
so nothing downstream changes when the parser does.

Rules a parser keeps, because the chunker and the multimodal pipeline read them:

* Items are in reading order; page headers, footers and page numbers are left out.
* One ``text`` item per layout block (paragraph, list item, heading).
* A heading is a ``text`` item with ``text_level`` (1 = document title, 2 = section
  header), or a markdown ``#`` line in its text.
* Images are saved under ``tmp_dir``; their ``img_path`` is absolute.
* ``layout`` is not set here: the router stamps it.
"""
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict


class TextItem(TypedDict):
    type: Literal["text"]
    text: str
    page_idx: int
    text_level: NotRequired[int]


class ImageItem(TypedDict):
    type: Literal["image"]
    img_path: str
    image_caption: list[str] | None
    image_footnote: list[str] | None
    page_idx: int


class TableItem(TypedDict):
    type: Literal["table"]
    table_body: str
    """The table as markdown."""
    table_caption: list[str] | None
    table_footnote: list[str] | None
    page_idx: int


ContentItem = TextItem | ImageItem | TableItem


class PdfDocumentParser(Protocol):
    async def __call__(self, path: Path, tmp_dir: Path) -> list[ContentItem]:
        """Parse the PDF at ``path`` into a content list; working files go under ``tmp_dir``."""
        ...
