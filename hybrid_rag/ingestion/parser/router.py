import asyncio
import json
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
import logging
from hybrid_rag import progress
from hybrid_rag.ingestion.parser.figures import fit_figures, host_figures
from hybrid_rag.ingestion.parser.office.office_to_pdf import SUPPORTED_EXTENSIONS as OFFICE_EXTENSIONS, convert_office_to_pdf
from hybrid_rag.ingestion.parser.pdf.amazon_textract import parse_document_to_json, reconstruct_content_list
from hybrid_rag.ingestion.parser.pdf.slides import is_slide_deck, parse_slides
from hybrid_rag.ingestion.parser.spreadsheet.parser import parse_xlsx

logger = logging.getLogger(__name__)

def _read_text_document(path: Path) -> list[dict]:
    """Turn a .md / .txt file into a one-item content list."""
    return [
        {
            "type": "text",
            "text": path.read_text(encoding="utf-8", errors="replace"),
        }
    ]

def _read_content_list(path: Path) -> list[dict]:
    """Read a pre-built content list, validated rather than trusted."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(
            f"{path.name}: content list harus berupa array of object, "
            f"isinya hanya ini : {type(data).__name__}"
        )
    return data

async def _parse_pdf(path: Path, tmp_dir: Path, slides: bool = False) -> list[dict]:
    """Dek slide → VLM per slide; selain itu Textract (TEXTRACT_MODE: async/sync), lalu content list dari blocks."""
    if slides or is_slide_deck(path):
        return await parse_slides(path, tmp_dir)
    raw_parse = await parse_document_to_json(str(path), output_file=str(tmp_dir / "raw_parsing_result.json"))
    return await asyncio.to_thread(
        reconstruct_content_list, raw_parse, output_file=str(tmp_dir / "formatted_parsing_result.json")
    )

async def _parse_office(path: Path, tmp_dir: Path) -> list[dict]:
    """docx/pptx/… → PDF lewat unoserver, lalu jalur PDF. pptx selalu dek slide, apa pun rasio halamannya
    (``is_slide_deck`` hanya mengenali 16:9/16:10). PDF dipindah ke ``tmp_dir`` agar ikut dibersihkan."""
    converted = Path(await convert_office_to_pdf(str(path)))
    pdf = tmp_dir / converted.name
    shutil.move(converted, pdf)
    shutil.rmtree(converted.parent, ignore_errors=True)
    return await _parse_pdf(pdf, tmp_dir, slides=path.suffix.lower() == ".pptx")

async def _parse_xlsx(path: Path, tmp_dir: Path) -> list[dict]:
    return await asyncio.to_thread(parse_xlsx, path, tmp_dir / "images")

async def _read_text(path: Path, tmp_dir: Path) -> list[dict]:
    return await asyncio.to_thread(_read_text_document, path)

async def _read_prebuilt(path: Path, tmp_dir: Path) -> list[dict]:
    """A content list that is already the parser's output (``notion_sync``): nothing to parse,
    only its images brought under the VLM limits like every parser's figures."""
    content_list = await asyncio.to_thread(_read_content_list, path)
    await asyncio.to_thread(fit_figures, content_list, tmp_dir)
    return content_list

PARSERS = {
    "pdf": (_parse_pdf, True),
    **{ext: (_parse_office, True) for ext in OFFICE_EXTENSIONS},
    "xlsx": (_parse_xlsx, True),
    "md": (_read_text, False),
    "markdown": (_read_text, False),
    "txt": (_read_text, False),
    "json": (_read_prebuilt, True),
}
SUPPORTED_INGEST_EXTENSIONS = frozenset(PARSERS)

@asynccontextmanager
async def parse_file(file_path: str | Path) -> AsyncIterator[list[dict]]:
    """Parse one file into a content list, ready for ``rag.insert_content_list``"""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lstrip(".").lower()
    if ext not in PARSERS:
        raise ValueError(
            f"Tipe file .{ext} tidak didukung. "
            f"Didukung: {', '.join('.' + e for e in sorted(PARSERS))}"
        )
    parser, has_local_figures = PARSERS[ext]

    logger.info(f"[parse] starting file={path.name}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_assets_"))

    try:
        progress.stage("parse")
        content_list = await parser(path, tmp_dir)
        if has_local_figures:
            await asyncio.to_thread(host_figures, content_list)

        logger.info(f"[parse] prepared {len(content_list)} items from file={path.name}")
        yield content_list

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.debug(f"[parse] cleaned up tmp assets dir: {tmp_dir}")