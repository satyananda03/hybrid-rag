"""Gambar (.jpg/.jpeg/.png) → content list: Textract selalu membaca, VLM hanya untuk yang visual."""
import asyncio
from pathlib import Path

from PIL import Image, ImageOps

from hybrid_rag.ingestion.parser.figures import _flatten, save_for_vlm
from hybrid_rag.ingestion.parser.pdf.amazon_textract import parse_document_to_json, reconstruct_content_list
from hybrid_rag.ingestion.parser.pdf.amazon_textract.blocks import build_block_map, get_children
from hybrid_rag.ingestion.parser.pdf.amazon_textract.content_list import (
    IGNORED_LAYOUT_TYPES,
    LAYOUT_TEXT_TYPES,
    _text_item,
)
from hybrid_rag.ingestion.parser.pdf.amazon_textract.pages import RENDER_DPI
from hybrid_rag.ingestion.parser.pdf.contract import ContentItem

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")
# Di bawah porsi teks/tabel ini sebuah gambar dianggap visual. Terukur di layout Textract
# (pageindex): gambar dokumen 34-57 %, foto dan chart 0-3 %.
VISUAL_TEXT_COVERAGE = 0.15
_TEXTUAL_TYPES = LAYOUT_TEXT_TYPES | IGNORED_LAYOUT_TYPES | {"LAYOUT_TABLE"}


def _upright(img: Image.Image) -> Image.Image:
    """EXIF diterapkan (foto HP tidak sampai miring ke Textract/VLM), mode yang bisa disimpan JPEG."""
    return _flatten(ImageOps.exif_transpose(img))


def image_to_pdf(src: Path, dst: Path) -> Path:
    """PDF satu halaman pada ``RENDER_DPI``, jadi render halaman = piksel gambar itu sendiri."""
    with Image.open(src) as img:
        _upright(img).save(dst, "PDF", resolution=RENDER_DPI)
    return dst


def text_coverage(raw_page: dict) -> float:
    """Porsi luas halaman yang dibaca Textract sebagai teks atau tabel (header/footer termasuk)."""
    blocks = raw_page.get("Blocks") or []
    page = next((b for b in blocks if b["BlockType"] == "PAGE"), None)
    if page is None:
        return 0.0
    boxes = [b["Geometry"]["BoundingBox"] for b in get_children(page, build_block_map(blocks))
             if b["BlockType"] in _TEXTUAL_TYPES]
    return sum(box["Width"] * box["Height"] for box in boxes)


def margin_items(raw_page: dict) -> tuple[list, list]:
    """Header, footer, dan nomor halaman yang dibuang ``reconstruct_content_list``, sebagai item ``text``.

    Letaknya, bukan tipenya, yang menentukan tempatnya: di atas tengah halaman ke awal content
    list, di bawah ke akhir (nomor halaman di pojok atas ikut header-nya). Urutan baca Textract.
    """
    blocks = raw_page.get("Blocks") or []
    page = next((b for b in blocks if b["BlockType"] == "PAGE"), None)
    if page is None:
        return [], []
    block_map = build_block_map(blocks)
    top, bottom = [], []
    for block in get_children(page, block_map):
        if block["BlockType"] in IGNORED_LAYOUT_TYPES and (item := _text_item(block, block_map, 0)):
            (top if block["Geometry"]["BoundingBox"]["Top"] < 0.5 else bottom).append(item)
    return top, bottom


def _whole_image(src: Path, tmp_dir: Path) -> ContentItem:
    with Image.open(src) as img:
        path = save_for_vlm(_upright(img), tmp_dir / "whole_image.jpg")
    return {"type": "image", "img_path": str(path.resolve()), "image_caption": None,
            "image_footnote": None, "page_idx": 0}


async def parse_image(path: Path, tmp_dir: Path) -> list[ContentItem]:
    """Gambar → content list: OCR untuk gambar dokumen; gambar utuh + kata OCR untuk gambar visual."""
    pdf = await asyncio.to_thread(image_to_pdf, path, tmp_dir / f"{path.stem}.pdf")
    raw = await parse_document_to_json(str(pdf), output_file=str(tmp_dir / "raw_parsing_result.json"))
    content_list = await asyncio.to_thread(
        reconstruct_content_list, raw, output_file=str(tmp_dir / "formatted_parsing_result.json")
    )
    top, bottom = margin_items(raw["pages"][0])
    content_list = [*top, *content_list, *bottom]
    if text_coverage(raw["pages"][0]) >= VISUAL_TEXT_COVERAGE:
        return content_list
    # Crop figure tergantikan gambar utuh; teks dan tabel tetap — itu data yang sudah terbaca.
    whole = await asyncio.to_thread(_whole_image, path, tmp_dir)
    return [whole, *(item for item in content_list if item.get("type") != "image")]
