"""Orkestrasi PDF → raw JSON Textract: cek kualitas → split → AnalyzeDocument per halaman."""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
from .pages import split_pdf_pages
from .quality import check_effective_dpi
from .textract_api import TextractPageError, analyze_page, get_client

logger = logging.getLogger(__name__)


def _analyze_pages(textract, file_path: str, n_pages: int, max_workers: int) -> list:
    """Kirim semua halaman ke Textract; return hasil per halaman sesuai urutan."""
    raw_pages = [None] * n_pages

    if max_workers > 1:
        # Concurrent processing: split PDF cuma ~ms/halaman, jadi semua halaman
        # langsung disubmit; PDF per halaman kecil (~0.1-0.3 MB) sehingga antrean di RAM ringan.
        print(f"[Textract] Concurrent processing dengan {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(analyze_page, textract, page): page["page_idx"]
                for page in split_pdf_pages(file_path)
            }
            print(f"[Textract] {n_pages} halaman disubmit ke Textract")
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                raw_pages[idx] = future.result()  # analyze_page tidak pernah raise
                logger.info(f"[Textract] Halaman {idx + 1}/{n_pages} selesai" + (f" (ERROR: {raw_pages[idx]['error']})" if raw_pages[idx]["error"] else ""))
    else:
        # Sequential processing
        for page in split_pdf_pages(file_path):
            idx = page["page_idx"]
            print(f"[Textract] Memproses halaman {idx + 1}/{n_pages}...")
            raw_pages[idx] = analyze_page(textract, page)
            if raw_pages[idx]["error"]:
                logger.warning(f"[Textract] Halaman {idx + 1} error: {raw_pages[idx]['error']}")
            else:
                logger.info(f"[Textract] Halaman {idx + 1} selesai")
    return raw_pages


def parse_document_to_json(file_path: str, output_file: str = "raw_output.json", region_name: str = None, max_workers: int = 5) -> dict:
    """Pipeline PDF -> raw Output JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {file_path}")

    all_dpi = check_effective_dpi(file_path)

    output_file = os.path.abspath(output_file)
    tmp_dir = os.path.dirname(output_file)
    os.makedirs(tmp_dir, exist_ok=True)

    images_dir = os.path.join(tmp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"[Textract] Memecah PDF per halaman: {file_path}")
    raw_pages = _analyze_pages(get_client(), file_path, len(all_dpi), max_workers)

    raw_data = {
        "source_pdf": os.path.abspath(file_path),
        "images_dir": images_dir,
        "pages": raw_pages,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Textract] Raw JSON disimpan ke: {output_file}")

    failed_pages = [p for p in raw_pages if p["error"]]
    if failed_pages:
        raise TextractPageError(failed_pages)
    return raw_data
