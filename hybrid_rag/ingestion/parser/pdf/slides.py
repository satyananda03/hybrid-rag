"""PDF slide deck → satu item teks per slide, ditulis VLM dari gambar slide utuh, plus satu item tabel per grid.

Parser teks meratakan tata letak slide: tanggal lepas dari langkah timeline-nya,
legenda warna hilang, teks di logo tidak terbaca. VLM melihat slide utuh, dengan
text layer PDF sebagai referensi ejaan. Teksnya menjadi item ``text`` biasa, jadi ia
lewat jalur teks (chunking, ekstraksi KG, sitasi halaman). ``img_path`` item teks
hanya membawa gambar slide ke S3 (``host_figures``) dan ke hasil retrieval; isinya
tidak pernah masuk ke teks chunk.

Grid (tabel, matriks) dikembalikan VLM terpisah sebagai data terstruktur, bukan
Markdown di dalam teks: pemecah chunk teks memotong tabel per baris dan memisahkan
header dari datanya. Sebagai item ``table`` ia lewat jalur yang sama dengan tabel
PDF/Excel: satu chunk utuh, analisis tabel, entity ``(table)``.
"""
import asyncio
import logging
import re
from pathlib import Path

import pymupdf

from hybrid_rag.config import settings
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import acronym_glossary
from hybrid_rag.ingestion.multimodal.content import encode_image_to_base64
from hybrid_rag.ingestion.multimodal.chain import SlideTable, slide_transcription_chain
from hybrid_rag.ingestion.parser.figures import FIGURE_MAX_PIXELS, save_for_vlm
from hybrid_rag.ingestion.parser.pdf.amazon_textract.pages import render_page_image

logger = logging.getLogger(__name__)

# 16:10 dan 16:9 (≥1.5) saja; dek 4:3 (1.33) tetap ke Textract, tambah override eksplisit bila sering muncul. A4 landscape (1.41) dan Letter landscape (1.29) sengaja di bawah ambang.
SLIDE_MIN_ASPECT = 1.5
SLIDE_MIN_SHARE = 0.8
MIN_COVERAGE = 0.8
# Satu slide 3840 px ≈ 25 MB mentah; tanpa batas, semua slide dirender sekaligus dan
# worker kena OOM (dek 22 slide: puncak ~1.8 GB). Serial juga karena MuPDF tidak
# thread-safe. VLM tetap paralel (vlm_max_async), jadi total waktu hampir tidak berubah.
SLIDE_RENDER_CONCURRENCY = 1

def is_slide_deck(path: str | Path) -> bool:
    """Sebagian besar halaman berbentuk layar lebar → PDF ini ekspor presentasi.

    Metadata (creator/producer) tidak dipakai: dek hasil iLovePDF atau Canva
    tidak menyebut PowerPoint di sana. PDF yang tidak terbaca PyMuPDF dianggap
    bukan dek, supaya jalur Textract tetap yang menangani (dan melaporkan) PDF itu.
    """
    try:
        doc = pymupdf.open(path)
    except pymupdf.FileDataError:
        return False
    with doc:
        if not doc.page_count:
            return False
        wide = sum(page.rect.width / page.rect.height >= SLIDE_MIN_ASPECT for page in doc)
        return wide / doc.page_count >= SLIDE_MIN_SHARE


def _gray(color: int) -> bool:
    r, g, b = color >> 16 & 255, color >> 8 & 255, color & 255
    return max(r, g, b) - min(r, g, b) < 24


def style_groups(page: pymupdf.Page) -> str:
    """Blok ``<style_groups>``: teks slide dikelompokkan per gaya non-default (warna non-abu, lalu tebal).

    VLM melihat warna dan tebal dengan benar, tetapi sering salah memasangkan judul
    kelompok ("4 SUBSEKTOR BARU", biru) dengan item bergaya sama. Pengelompokan dari
    font PDF membuat pasangan itu tidak perlu dicari secara visual. "" bila slide
    tidak punya teks bergaya.
    """
    groups: dict[str, list[str]] = {}
    for block in page.get_text("dict", sort=True)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                if not _gray(span["color"]):
                    groups.setdefault(f"warna #{span['color']:06x}", []).append(text)
                elif span["flags"] & pymupdf.TEXT_FONT_BOLD or "bold" in span["font"].lower():
                    groups.setdefault("tebal", []).append(text)
    if not groups:
        return ""
    lines = "\n".join(f"{style}: " + " | ".join(texts) for style, texts in groups.items())
    return f"<style_groups>\n{lines}\n</style_groups>"


def _coverage(reference: str, transcript: str) -> float:
    """Porsi kata text layer (> 3 huruf) yang ikut tertulis di transkripsi."""
    words = set(re.findall(r"\w{4,}", reference.lower()))
    if not words:
        return 1.0
    found = set(re.findall(r"\w{4,}", transcript.lower()))
    return len(words & found) / len(words)


def table_markdown(table: SlideTable) -> str:
    """Grid VLM → Markdown table yang selalu valid: header + pemisah, semua baris selebar baris terlebar.

    Baris yang lebih lebar dari header diberi kolom kosong, bukan dipotong: sel tidak boleh hilang.
    """
    width = max([len(table.columns), *map(len, table.rows)])

    def line(cells: list[str]) -> str:
        cells = [c.replace("|", "\\|").replace("\n", " ").strip() for c in cells]
        return "| " + " | ".join(cells + [""] * (width - len(cells))) + " |"

    return "\n".join([line(table.columns), "|" + " --- |" * width, *map(line, table.rows)])


def slide_items(transcription, page_idx: int, image_path: str) -> list[dict]:
    """Satu item ``text`` (bila ada teks) lalu satu item ``table`` per grid yang punya baris, urut seperti di slide."""
    text = (transcription.markdown or "").strip()
    items = [{"type": "text", "text": text, "page_idx": page_idx, "img_path": image_path}] if text else []
    items += [
        {
            "type": "table",
            "table_body": table_markdown(t),
            "table_caption": [t.caption] if t.caption.strip() else [],
            "table_footnote": t.footnotes,
            "page_idx": page_idx,
        }
        for t in transcription.tables
        if t.rows
    ]
    return items


async def parse_slides(path: Path, tmp_dir: Path, entity_registry: str | None = None) -> list[dict]:
    """Satu item ``text`` per slide (dengan gambar slidenya) diikuti item ``table`` per grid, berurutan, dengan ``page_idx``.

    ``entity_registry`` memberi VLM kepanjangan akronim yang disebut slide (lihat ``acronym_glossary``).
    """
    with pymupdf.open(path) as doc:
        references = [page.get_text(sort=True).strip() for page in doc]
        styles = [style_groups(page) for page in doc]
        # Render langsung ke FIGURE_MAX_PIXELS di sisi panjang: teks kecil (spanduk di foto) masih terbaca.
        dpis = [FIGURE_MAX_PIXELS * 72 / max(page.rect.width, page.rect.height) for page in doc]

    semaphore = asyncio.Semaphore(max(1, settings.vlm_max_async))
    render_gate = asyncio.Semaphore(SLIDE_RENDER_CONCURRENCY)
    chain = slide_transcription_chain()

    def _render(page_idx: int) -> Path:
        # Gambar mentah hanya hidup di sini; yang menunggu giliran VLM cukup file JPEG-nya.
        image = render_page_image(str(path), page_idx, dpis[page_idx], False)
        image_path = save_for_vlm(image, tmp_dir / f"slide_{page_idx}.jpg")
        # Cache MuPDF menahan gambar ter-decode antar-render (ratusan MB); hasil render tidak terpengaruh.
        pymupdf.TOOLS.store_shrink(100)
        return image_path

    async def _transcribe(page_idx: int) -> list[dict]:
        async with render_gate:
            image_path = await asyncio.to_thread(_render, page_idx)
        async with semaphore:
            transcription = await chain.ainvoke({
                "page_number": page_idx + 1,
                "reference_text": references[page_idx] or "(kosong)",
                "style_groups": styles[page_idx],
                "glossary": acronym_glossary(entity_registry, references[page_idx]),
                "image_url": f"data:image/jpeg;base64,{encode_image_to_base64(str(image_path))}",
            })
        items = slide_items(transcription, page_idx, str(image_path))
        # Gagal keras, bukan jatuh ke teks mentah: slide tanpa transkripsi berarti isinya hilang diam-diam.
        if not items:
            raise RuntimeError(f"{path.name}: slide {page_idx + 1} tidak menghasilkan transkripsi")
        written = "\n".join(item.get("text") or item["table_body"] for item in items)
        coverage = _coverage(references[page_idx], written)
        if coverage < MIN_COVERAGE:
            logger.warning(f"[slides] {path.name} slide {page_idx + 1}: hanya {coverage:.0%} kata text layer ikut tertulis")
        return items

    logger.info(f"[slides] {path.name}: {len(references)} slide → VLM")
    pages = await asyncio.gather(*(_transcribe(i) for i in range(len(references))))
    return [item for items in pages for item in items]
