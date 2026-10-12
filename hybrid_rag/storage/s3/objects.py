"""S3 for the Textract parser: the async job's input PDF and the figure crops.

Configured by ``S3_BUCKET``, ``S3_PREFIX``, ``S3_REGION`` and, optionally,
``S3_ACCESS_KEY_ID`` / ``S3_SECRET_ACCESS_KEY`` (boto3's default credential chain
otherwise). Needs the ``ingestion`` extra (boto3).
"""

import hashlib
import logging
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from hybrid_rag.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client():
    """The process-wide boto3 client, built on first use."""
    return boto3.client(
        "s3",
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        config=BotoConfig(signature_version="s3v4"),
    )


def require_bucket() -> str:
    """The configured bucket, or a clear error instead of a NoneType crash in boto3."""
    if not settings.s3_bucket:
        raise ValueError("S3_BUCKET belum diset — upload ke S3 tidak bisa jalan.")
    return settings.s3_bucket


def upload_fileobj(fileobj, key: str, content_type: str | None = None) -> None:
    get_client().upload_fileobj(
        fileobj,
        require_bucket(),
        key,
        ExtraArgs={"ContentType": content_type or "application/octet-stream"},
    )


def delete_object(key: str) -> None:
    get_client().delete_object(Bucket=require_bucket(), Key=key)


def object_url(key: str) -> str:
    """Public URL of one object. The VLM fetches figures from it directly."""
    return f"https://{require_bucket()}.s3.{settings.s3_region}.amazonaws.com/{key}"


def figure_key(file_path: str) -> str:
    """Object key of one figure crop, under ``S3_PREFIX``."""
    ext = Path(file_path).suffix.lower() or ".jpg"
    stamp = time.strftime("%y%m%d%H%M%S")
    short_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    return f"{settings.s3_prefix.strip('/')}/{stamp}_{short_hash}{ext}"


def upload_figures_to_s3(items: list[dict], max_workers: int = 5) -> None:
    """Upload every figure crop in parallel and replace its ``img_path`` with the URL.

    An item whose upload fails keeps its local path: the figure still reaches the
    VLM, it just cannot be reused once the temp dir is gone. So a failure is
    logged, not raised.
    """
    if not items:
        return

    def _put(path: str) -> str:
        key = figure_key(path)
        with open(path, "rb") as fh:
            upload_fileobj(fh, key, mimetypes.guess_type(path)[0])
        return object_url(key)

    get_client()  # built in this thread: the client is thread-safe, its construction is not
    uploaded = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = {pool.submit(_put, item["img_path"]): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                item["img_path"] = future.result()
                uploaded += 1
            except Exception as e:
                logger.warning(f"Upload figure ke S3 gagal ({e}); pakai file lokal: {item['img_path']}")
    logger.info(f"[S3] {uploaded}/{len(items)} figure diupload")
