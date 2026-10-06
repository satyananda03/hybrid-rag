import asyncio
import json
import shutil
import tempfile
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional
import logging
from hybrid_rag.providers import progress

from hybrid_rag.ingestion.parser.amazon_textract import reconstruct_content_list
from hybrid_rag.ingestion.parser import async_amazon_textract
from hybrid_rag.ingestion.parser.office_to_pdf import (
    SUPPORTED_EXTENSIONS as OFFICE_EXTENSIONS,
    convert_office_to_pdf,
)

logger = logging.getLogger(__name__)


TEXT_EXTENSIONS = {"md", "markdown", "txt"}
#: Sumber yang sudah menghasilkan content list jadi (``notion_sync``). Tidak ada
#: yang bisa di-parse lagi dari file semacam ini — isinya justru bentuk KELUARAN
#: parser, jadi Textract dilewati sepenuhnya.
CONTENT_LIST_EXTENSIONS = {"json"}
SUPPORTED_INGEST_EXTENSIONS = (
    {"pdf"} | OFFICE_EXTENSIONS | TEXT_EXTENSIONS | CONTENT_LIST_EXTENSIONS
)

def _read_text_document(path: Path) -> list[dict]:
    """Turn a .md / .txt file into a one-item content list."""
    return [
        {
            "type": "text",
            "text": path.read_text(encoding="utf-8", errors="replace"),
            "page_idx": 0,
        }
    ]

def _read_content_list(path: Path) -> list[dict]:
    """Read a pre-built content list, validated rather than trusted.

    A malformed file that slips through here fails much later and much more
    confusingly — ``separate_content`` calls ``.get`` on whatever it is handed,
    so a JSON object or a list of strings surfaces as an ``AttributeError`` deep
    in the pipeline with nothing pointing back at the file.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(
            f"{path.name}: content list harus berupa array of object, "
            f"isinya hanya ini : {type(data).__name__}"
        )
    return data

@asynccontextmanager
async def parse_file(file_path: str | Path) -> AsyncIterator[list[dict]]:
    """Parse one file into a content list, ready for ``rag.insert_content_list``.

    Writes nothing to the RAG storages, so callers run it outside the workspace
    write lock: several files of one workspace can be parsed (Textract) at once
    while the previous one is still being inserted. Textract jobs stay capped
    process-wide by ``TEXTRACT_MAX_CONCURRENT_JOBS``.

    A context manager, not a plain function: the temp directory holds the figure
    crops, and a crop whose S3 upload failed keeps a local ``img_path`` that the
    VLM still reads during insert. The directory is removed only when the
    ``async with`` block — insert included — is done.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lstrip(".").lower()
    if ext not in SUPPORTED_INGEST_EXTENSIONS:
        raise ValueError(
            f"Tipe file .{ext} tidak didukung. "
            f"Didukung: {', '.join('.' + e for e in sorted(SUPPORTED_INGEST_EXTENSIONS))}"
        )

    logger.info(f"[parse] starting file={path.name}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_assets_"))
    office_outdir: Optional[str] = None

    try:
        progress.stage("parse")
        if ext in CONTENT_LIST_EXTENSIONS:
            content_list = await asyncio.to_thread(_read_content_list, path)
            logger.info("[parse] loaded pre-built content list, skipping Textract")
        elif ext in TEXT_EXTENSIONS:
            content_list = await asyncio.to_thread(_read_text_document, path)
            logger.info(f"[parse] read {ext} directly, skipping Textract")
        else:
            parse_path = path
            if ext in OFFICE_EXTENSIONS:
                logger.info(f"[parse] converting office file to PDF: {path.name}")
                pdf_path = await convert_office_to_pdf(str(path))
                office_outdir = os.path.dirname(pdf_path)
                parse_path = Path(pdf_path)

            # ── Step 1: Parse document via Textract ──
            raw_parse = await async_amazon_textract.parse_document_to_json(
                str(parse_path), output_file=str(tmp_dir / "raw_parsing_result.json")
            )

            # ── Step 2: Reconstruct content_list ──
            content_list = await asyncio.to_thread(
                reconstruct_content_list,
                raw_parse,
                output_file=str(tmp_dir / "formatted_parsing_result.json"),
            )

            # Gambar tidak ditarik ke lokal: ``img_path`` sudah berupa URL S3
            # dan VLM membacanya langsung dari sana.

        logger.info(f"[parse] prepared {len(content_list)} items from file={path.name}")
        yield content_list

    finally:
        if office_outdir:
            shutil.rmtree(office_outdir, ignore_errors=True)
            logger.debug(f"[parse] cleaned up office conversion dir: {office_outdir}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.debug(f"[parse] cleaned up tmp assets dir: {tmp_dir}")
