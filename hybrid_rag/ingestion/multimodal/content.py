
"""Helpers for the multimodal ingestion path."""

import base64
from functools import lru_cache
from typing import Dict, List, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag.config import settings
from hybrid_rag.core.chunk_schema import normalize_page
import logging

logger = logging.getLogger(__name__)


#: Hierarki separator dari kasar ke halus — sama seperti yang dipakai chunking
#: parent-child. Konteks dipotong di batas paragraf/baris/kalimat, tidak pernah
#: putus di tengah kata.
_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def normalize_caption_list(value: Any) -> List[str]:
    """Return captions and footnotes as a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def get_table_body(item: Dict[str, Any]) -> Any:
    """Read table content across common content-list alias fields."""
    if item.get("table_body") not in (None, ""):
        return item.get("table_body")
    if item.get("table_data") not in (None, ""):
        return item.get("table_data")
    return item.get("text", "")


def format_table_body(table_body: Any) -> str:
    """Serialize table content for prompts and chunks without dropping aliases."""
    if isinstance(table_body, str):
        return table_body
    if isinstance(table_body, list):
        if not table_body:
            return ""
        if all(isinstance(row, (list, tuple)) for row in table_body):
            rendered_rows = [
                "| " + " | ".join(str(cell) for cell in row) + " |"
                for row in table_body
            ]
            if len(rendered_rows) >= 1:
                column_count = max(len(row) for row in table_body)
                separator = "| " + " | ".join(["---"] * column_count) + " |"
                rendered_rows.insert(1, separator)
            return "\n".join(rendered_rows)
        return "\n".join(str(row) for row in table_body)
    return str(table_body)


def get_equation_text_and_format(item: Dict[str, Any]) -> Tuple[str, str]:
    """Read equation content while preserving LaTeX aliases from content lists."""
    text = str(item.get("text", "") or "").strip()
    latex = str(item.get("latex", "") or "").strip()
    equation = str(item.get("equation", "") or "").strip()
    equation_format = str(item.get("text_format", "") or "").strip()

    if text:
        equation_text = text
    elif latex:
        equation_text = latex
        if not equation_format:
            equation_format = "latex"
    elif equation:
        equation_text = equation
    else:
        equation_text = ""

    return equation_text, equation_format


def extract_section_path_from_content_list(
    content_list: List[Dict[str, Any]], current_index: int
) -> str:
    """Build a hierarchical section path from preceding heading blocks.

    MinerU content lists keep document order, and heading blocks are exposed as
    text items with a positive ``text_level``.  For a given item index, we walk
    the preceding items and keep the latest heading at each level to reconstruct
    a stable chapter/section path such as ``Introduction > Method > Ablation``.
    """
    if not content_list or current_index is None:
        return ""

    try:
        limit = max(0, int(current_index))
    except (TypeError, ValueError):
        return ""

    heading_chain: List[Tuple[int, str]] = []

    for item in content_list[:limit]:
        if not isinstance(item, dict):
            continue

        if item.get("type", "text") != "text":
            continue

        text = str(item.get("text", "") or "").strip()
        if not text:
            continue

        try:
            level = int(item.get("text_level", 0) or 0)
        except (TypeError, ValueError):
            continue

        if level <= 0:
            continue

        while heading_chain and heading_chain[-1][0] >= level:
            heading_chain.pop()
        heading_chain.append((level, text))

    return " > ".join(text for _, text in heading_chain)


#: Model yang sama dengan ``TiktokenTokenizer`` di core/utils, supaya hitungan
#: token di sini dan di splitter tidak berbeda.
_TOKEN_MODEL = "gpt-4o-mini"


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.encoding_for_model(_TOKEN_MODEL)


@lru_cache(maxsize=4)
def _splitter(max_tokens: int) -> RecursiveCharacterTextSplitter:
    """Splitter yang menghitung ukuran dalam token, bukan karakter."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=_TOKEN_MODEL,
        chunk_size=max_tokens,
        chunk_overlap=0,
        separators=_SPLIT_SEPARATORS,
    )


def _take_nearest(blocks: List[str], max_tokens: int, from_end: bool) -> str:
    """Blok terdekat item, sebanyak yang muat dalam budget token.

    Batas potongnya adalah batas blok — batas paling bersih yang tersedia,
    karena tiap blok memang satu paragraf utuh dari parser. Baru kalau satu
    blok sendirian sudah melewati budget, isinya dipotong
    :class:`RecursiveCharacterTextSplitter` mengikuti ``_SPLIT_SEPARATORS``,
    jadi konteks tidak pernah terputus di tengah kata.
    """
    if not blocks or max_tokens <= 0:
        return ""

    picked: List[str] = []
    used = 0

    # Dari yang terdekat item ke yang terjauh, berhenti begitu budget habis.
    for block in reversed(blocks) if from_end else blocks:
        length = len(_encoding().encode(block))
        if used + length <= max_tokens:
            picked.append(block)
            used += length
            continue
        if not picked:
            # Blok tetangga langsung pun sudah kebesaran: ambil ujungnya yang
            # menempel ke item. Separator ". " membuat titik pemisah jatuh ke
            # potongan sebelah, jadi ujung depannya dirapikan.
            chunks = _splitter(max_tokens).split_text(block)
            if chunks:
                chunk = (chunks[-1] if from_end else chunks[0]).strip()
                picked.append(chunk.lstrip(". ") if from_end else chunk)
        break

    if from_end:
        picked.reverse()
    return "\n\n".join(picked)


def extract_leading_trailing_from_content_list(
    content_list: List[Dict[str, Any]],
    current_index: int,
    leading_tokens: int | None = None,
    trailing_tokens: int | None = None,
) -> Tuple[str, str]:
    """Teks sebelum dan sesudah satu item, masing-masing dibatasi token.

    Dipanggil sekali per item saat ``separate_content`` dan hasilnya menempel
    di item itu, sehingga ikut tersimpan ke ``full_docs.multimodal_items`` dan
    tetap ada saat tahap analisis berjalan dari database (dokumen yang
    di-resume atau di-retry tidak pernah melewati insert lagi).
    """
    if not content_list or current_index is None:
        return "", ""

    try:
        idx = int(current_index)
    except (TypeError, ValueError):
        return "", ""

    if idx < 0 or idx >= len(content_list):
        return "", ""

    def texts(positions) -> List[str]:
        out = []
        for pos in positions:
            item = content_list[pos]
            if not isinstance(item, dict) or item.get("type", "text") != "text":
                continue
            text = str(item.get("text", "") or "").strip()
            if text:
                out.append(text)
        return out

    leading = _take_nearest(
        texts(range(0, idx)),
        leading_tokens if leading_tokens is not None else settings.leading_context_tokens,
        from_end=True,
    )
    trailing = _take_nearest(
        texts(range(idx + 1, len(content_list))),
        trailing_tokens if trailing_tokens is not None else settings.trailing_context_tokens,
        from_end=False,
    )
    return leading, trailing


def page_offsets_of(text_parts: List[str], pages: List[int | str | None]) -> List[Dict[str, Any]]:
    """Awal tiap halaman di dalam teks gabungan, dalam koordinat teks yang **tersimpan**.

    Enqueue menyimpan ``sanitize_text_for_encoding("\\n\\n".join(parts))`` (worker.py),
    dan sanitize bisa mengubah panjang (``html.unescape``, strip). Maka offset dihitung
    pada hasil fungsi yang sama atas input yang sama — bukan pada teks mentah — dengan
    ``find`` berurutan atas tiap part yang sudah disanitasi. Teks gabungan sendiri tidak
    disentuh; ini murni perhitungan di samping.

    Part yang tidak ketemu (semestinya tidak pernah) dilewati, bukan digalatkan: halaman
    yang hilang hanya membuat chunk itu tanpa ``sidecar``, sedangkan ingest yang gagal
    membuat dokumen hilang seluruhnya. Part tanpa halaman (``None``: .md/.txt) juga
    dilewati, jadi dokumen tanpa halaman menghasilkan ``[]``.
    """
    from hybrid_rag.core.utils import sanitize_text_for_encoding

    whole = sanitize_text_for_encoding("\n\n".join(text_parts))
    offsets: List[Dict[str, Any]] = []
    cursor = 0
    for part, page in zip(text_parts, pages):
        if page is None:
            continue
        needle = sanitize_text_for_encoding(part)
        if not needle:
            continue
        pos = whole.find(needle, cursor)
        if pos < 0:
            logger.warning("[page-offsets] part halaman %s tidak ditemukan di teks tersanitasi", page)
            continue
        cursor = pos + len(needle)
        if not offsets or offsets[-1]["page"] != page:
            offsets.append({"page": page, "start": pos})
    return offsets


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, int]]]:
    """
    Separate text content and multimodal content

    Args:
        content_list: Content list from MinerU parsing

    Returns:
        (text_content, multimodal_items, page_offsets): Pure text content, multimodal
        items list, and ``[{"page": idx, "start": char}]`` — awal tiap halaman di dalam
        ``text_content`` setelah sanitasi (lihat :func:`page_offsets_of`).
    """
    text_parts = []
    text_pages = []
    multimodal_items = []

    for index, item in enumerate(content_list):
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
                text_pages.append(normalize_page(item.get("page_idx")))
        else:
            # Multimodal content (image, table, equation, etc.)
            multimodal_item = dict(item)
            multimodal_item.setdefault("_content_list_index", index)
            multimodal_item.setdefault(
                "_section_path",
                extract_section_path_from_content_list(content_list, index),
            )
            # Menempel di item, bukan disimpan sebagai state processor: item
            # inilah yang tersimpan ke DB dan dibaca lagi oleh tahap analisis.
            leading, trailing = extract_leading_trailing_from_content_list(
                content_list, index
            )
            multimodal_item.setdefault("_leading_text", leading)
            multimodal_item.setdefault("_trailing_text", trailing)
            multimodal_items.append(multimodal_item)

    # Merge all text content
    text_content = "\n\n".join(text_parts)
    page_offsets = page_offsets_of(text_parts, text_pages)

    logger.info("Content separation complete:")
    logger.info(f"  - Text content length: {len(text_content)} characters")
    logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

    # Count multimodal types
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items, page_offsets


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Tidak ada validasi ukuran/dimensi di sini, dan itu disengaja: penyaringan
    sudah terjadi lebih dulu di parser: semua parser (PDF, docx, xlsx) membuang
    figure di bawah FIGURE_MIN_WIDTH_PX/FIGURE_MIN_HEIGHT_PX (dan
    FIGURE_MIN_AREA_RATIO untuk PDF), lihat ``parser/figures.py``, jadi gambar
    noise tidak pernah sampai ke sini.
    Jangan tambahkan lagi knob VLM_MIN_IMAGE_PIXEL / VLM_MAX_IMAGE_BYTES —
    keduanya pernah ada, tidak punya pembaca, dan sudah dihapus.

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        return None  # unknown type; caller's guard raises MultimodalAnalysisError


