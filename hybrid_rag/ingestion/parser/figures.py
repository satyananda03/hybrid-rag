"""What every parser does with the figures it finds: drop the small ones, host the rest.

PDF, docx and xlsx all end the same way: an ``image`` item whose ``img_path`` is a
local file is uploaded to S3 and now points at a URL the VLM can fetch. The size
thresholds are one setting for all of them (``FIGURE_MIN_*`` in ``config.py``).
"""
import io
import logging
from PIL import Image
from hybrid_rag.config import settings
from hybrid_rag.storage.s3 import upload_figures_to_s3

logger = logging.getLogger(__name__)

def is_large_enough(width_px: int, height_px: int, area_ratio: float | None = None) -> bool:
    """Whether a figure clears the size thresholds."""
    return (
        width_px >= settings.figure_min_width_px
        and height_px >= settings.figure_min_height_px
        and (area_ratio is None or area_ratio >= settings.figure_min_area_ratio)
    )

def is_large_image(data: bytes, name: str) -> bool:
    """:func:`is_large_enough` for embedded image bytes (docx, xlsx)."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception as e:
        logger.warning(f"Gambar {name} dilewati: tidak bisa dibaca ({type(e).__name__})")
        return False
    return is_large_enough(width, height)

def host_figures(content_list: list[dict]) -> None:
    """Upload every local ``image`` item of ``content_list`` to S3, in place."""
    upload_figures_to_s3([item for item in content_list if item["type"] == "image"])