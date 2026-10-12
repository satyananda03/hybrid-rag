"""Orkestrasi PDF → raw JSON Textract mode async: StartDocumentAnalysis + polling GetDocumentAnalysis."""
import asyncio
import io
import os
import random
import time
import uuid
import fitz
from hybrid_rag.config import settings
import logging
from hybrid_rag import progress
from hybrid_rag.storage.s3 import delete_object, require_bucket, upload_fileobj
from ..amazon_textract.pages import RENDER_DPI
from ..amazon_textract.quality import check_effective_dpi
from ..amazon_textract.textract_api import TextractPageError, get_client

logger = logging.getLogger(__name__)

FEATURE_TYPES = ["LAYOUT", "TABLES"]
POLL_FIRST_S = 2.0
POLL_MAX_S = 20.0  # 50 job / 20 s = 2,5 TPS polling; sisa kuota Get 5 TPS untuk ambil hasil
POLL_TIMEOUT_S = 30 * 60


def _normalized_pdf(pdf_path: str, dpi: int = RENDER_DPI) -> tuple[bytes, list[tuple[int, int]]]:
    """PDF utuh dengan /Rotate dibakar ke konten + ukuran piksel render ``dpi`` per halaman.

    Sama dengan ``split_pdf_pages``: bbox Textract = orientasi render yang dipakai crop
    figure, dan ``img_w``/``img_h`` memakai rumus yang identik.
    """
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    try:
        sizes = []
        for page in doc:
            irect = (page.rect * mat).irect  # dihitung sebelum rotasi dibakar, seperti sync
            sizes.append((irect.width, irect.height))
            page.remove_rotation()
        return doc.tobytes(garbage=3, deflate=True), sizes
    finally:
        doc.close()


_job_slots: asyncio.Semaphore | None = None


def _slots() -> asyncio.Semaphore:
    """Semaphore job Textract per proses, dibuat saat pertama dipakai (nilai dari settings)."""
    global _job_slots
    if _job_slots is None:
        _job_slots = asyncio.Semaphore(settings.textract_max_concurrent_jobs)
    return _job_slots


async def _wait_for_job(textract, job_id: str) -> dict:
    """Polling sampai job tidak ``IN_PROGRESS``; return respons Get pertama (halaman hasil 1).

    ponytail: polling memakan kuota Get (5 TPS default di ap-southeast-1) — cukup sampai
    ~50 job bersamaan; di atas itu naikkan kuota Get atau ganti SNS→SQS.
    """
    delay, deadline = POLL_FIRST_S, time.monotonic() + POLL_TIMEOUT_S
    while True:
        response = await asyncio.to_thread(textract.get_document_analysis, JobId=job_id, MaxResults=1000)
        if response["JobStatus"] != "IN_PROGRESS":
            return response
        if time.monotonic() > deadline:
            raise TimeoutError(f"Job Textract {job_id} belum selesai setelah {POLL_TIMEOUT_S} detik")
        # Jitter: job yang dimulai bersamaan tidak polling serempak.
        await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        delay = min(delay * 1.5, POLL_MAX_S)


def _collect_blocks(textract, job_id: str, first: dict) -> list:
    """Kumpulkan semua Blocks lewat ``NextToken`` (pola contoh Python dokumentasi AWS)."""
    blocks, response = list(first.get("Blocks", [])), first
    while response.get("NextToken"):
        response = textract.get_document_analysis(JobId=job_id, MaxResults=1000, NextToken=response["NextToken"])
        blocks.extend(response.get("Blocks", []))
    return blocks


def _raw_pages(first: dict, blocks: list, sizes: list) -> list:
    """Blocks satu job → daftar per halaman berbentuk sama dengan ``analyze_page`` sync."""
    by_page: dict[int, list] = {}
    for b in blocks:
        by_page.setdefault(b.get("Page", 1) - 1, []).append(b)

    status = first["JobStatus"]
    warned = {p - 1: w.get("ErrorCode") for w in first.get("Warnings") or [] for p in w.get("Pages") or []}
    raw_pages = []
    for idx, (img_w, img_h) in enumerate(sizes):
        page_blocks = by_page.get(idx, [])
        if status == "FAILED":
            error = first.get("StatusMessage") or "Job Textract FAILED"
        elif idx in warned:
            error = warned[idx] or "PARTIAL_SUCCESS"
        elif not any(b["BlockType"] == "PAGE" for b in page_blocks):
            error = "tidak ada hasil Textract untuk halaman ini"
        else:
            error = None
        raw_pages.append({
            "page_idx": idx,
            "img_w": img_w,
            "img_h": img_h,
            "DocumentMetadata": first.get("DocumentMetadata"),
            "AnalyzeDocumentModelVersion": first.get("AnalyzeDocumentModelVersion"),
            "Blocks": [] if error else page_blocks,
            "error": error,
        })
    return raw_pages


async def _run_job(pdf_bytes: bytes, n_pages: int) -> tuple[dict, list]:
    """Upload → Start → polling → kumpulkan Blocks; input S3 selalu dihapus."""
    token = uuid.uuid4().hex
    key = f"{settings.s3_prefix.strip('/')}/textract-input/{token}.pdf"
    await asyncio.to_thread(upload_fileobj, io.BytesIO(pdf_bytes), key, "application/pdf")
    try:
        textract = get_client()
        job_id = (await asyncio.to_thread(
            textract.start_document_analysis,
            DocumentLocation={"S3Object": {"Bucket": require_bucket(), "Name": key}},
            FeatureTypes=FEATURE_TYPES,
            ClientRequestToken=token,  # retry boto atas Start tidak memulai job ganda
        ))["JobId"]
        logger.info(f"[Textract] Job async {job_id} dimulai ({n_pages} halaman)")
        progress.activity(f"Textract async: memproses {n_pages} halaman")
        first = await _wait_for_job(textract, job_id)
        blocks = await asyncio.to_thread(_collect_blocks, textract, job_id, first) if first["JobStatus"] != "FAILED" else []
        logger.info(f"[Textract] Job {job_id} {first['JobStatus']}: {len(blocks)} blocks")
        progress.activity(f"Textract async selesai: {first['JobStatus']}")
        return first, blocks
    finally:
        try:
            await asyncio.to_thread(delete_object, key)
        except Exception:
            logger.warning(f"[Textract] Gagal menghapus input S3 {key}", exc_info=True)


async def parse_document_to_json(file_path: str, output_file: str = "raw_output.json") -> dict:
    """Pipeline PDF -> raw Output JSON via Textract async.

    ``output_file`` only places the working directory: the figure crops go to
    ``images/`` next to it.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {file_path}")
    if settings.textract_region != settings.resolved_s3_region:
        raise ValueError(
            f"Textract async wajib satu region dengan bucket S3: "
            f"TEXTRACT_REGION={settings.textract_region}, S3_REGION={settings.resolved_s3_region}"
        )

    await asyncio.to_thread(check_effective_dpi, file_path)

    output_file = os.path.abspath(output_file)
    tmp_dir = os.path.dirname(output_file)
    os.makedirs(tmp_dir, exist_ok=True)
    images_dir = os.path.join(tmp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    pdf_bytes, sizes = await asyncio.to_thread(_normalized_pdf, file_path)
    slots = _slots()
    if slots.locked():
        progress.activity("Menunggu antrean Textract")
    async with slots:
        first, blocks = await _run_job(pdf_bytes, len(sizes))

    raw_pages = _raw_pages(first, blocks, sizes)
    raw_data = {
        "source_pdf": os.path.abspath(file_path),
        "images_dir": images_dir,
        "pages": raw_pages,
    }

    failed_pages = [p for p in raw_pages if p["error"]]
    if failed_pages:
        raise TextractPageError(failed_pages)
    return raw_data
