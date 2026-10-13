
"""Image — described by the VLM from the image bytes plus its context."""

import json
import mimetypes
from typing import Dict, Any, Tuple, List
from pathlib import Path

import logging

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass

# Import prompt templates
from hybrid_rag.ingestion.multimodal.base_processor import BaseModalProcessor
from hybrid_rag.ingestion.multimodal.vlm import image_analysis_chain
from hybrid_rag.ingestion.multimodal.content import encode_image_to_base64

logger = logging.getLogger(__name__)


class ImageModalProcessor(BaseModalProcessor):
    """Processor specialized for image content"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
        cache_keys: List[str] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate image description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Image content to process
            content_type: Type of modal content ("image")
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (enhanced_caption, entity_info)
        """
        # Parse image content (reuse existing logic)
        if isinstance(modal_content, str):
            try:
                content_data = json.loads(modal_content)
            except json.JSONDecodeError:
                content_data = {"description": modal_content}
        else:
            content_data = modal_content

        image_path = content_data.get("img_path")
        captions = content_data.get(
            "image_caption", content_data.get("img_caption", [])
        )
        footnotes = content_data.get(
            "image_footnote", content_data.get("img_footnote", [])
        )
        section_path = content_data.get("_section_path", "")

        # Validate image path
        if not image_path:
            raise ValueError(
                f"No image path provided in modal_content: {modal_content}"
            )

        # URL (S3) dikirim apa adanya — VLM yang menarik gambarnya, tidak
        # ada file temporary yang dibaca di sini. File lokal (sumber non-S3
        # atau upload S3 yang gagal) tetap jalan lewat data URI.
        if image_path.startswith(("http://", "https://")):
            image_url = image_path
        else:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image_base64 = encode_image_to_base64(image_path)
            if not image_base64:
                raise RuntimeError(f"Failed to encode image to base64: {image_path}")
            mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            image_url = f"data:{mime};base64,{image_base64}"

        logger.info(f"[VLM] Calling vision model for: {image_path}")
        analysis = await image_analysis_chain().ainvoke(
            {
                "context_block": self._context_block(content_data),
                "section_path": section_path or "None",
                "image_path": image_path,
                "captions": captions or "None",
                "footnotes": footnotes or "None",
                "image_url": image_url,
            },
            config=self._cache_config(cache_keys=cache_keys),
        )
        enhanced_caption, entity_info = self._unpack_analysis(
            analysis, entity_name, "image"
        )

        # print("\n" + "="*70)
        # print(f"🖼️  IMAGE PROCESSING RESULT")
        # print(f"  entity_name : {entity_info['entity_name']}")
        # print(f"  entity_type : {entity_info['entity_type']}")
        # print(f"  summary     : {entity_info['summary']}")
        # print(f"  img_path    : {image_path}")
        # print(f"  captions    : {captions}")
        # print(f"─"*70)
        # print(f"  CHUNK yang akan disimpan:")
        # print(enhanced_caption)
        # print("="*70 + "\n")

        return enhanced_caption, entity_info

