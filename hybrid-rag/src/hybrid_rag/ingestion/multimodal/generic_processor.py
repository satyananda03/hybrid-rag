
"""Fallback for any content type without a dedicated processor."""

from typing import Dict, Any, Tuple, List

import logging

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass

# Import prompt templates
from hybrid_rag.ingestion.multimodal.base_processor import BaseModalProcessor
from hybrid_rag.ingestion.multimodal.vlm import analysis_chain

logger = logging.getLogger(__name__)

class GenericModalProcessor(BaseModalProcessor):
    """Generic processor for other types of modal content"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
        cache_keys: List[str] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate generic modal description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Generic modal content to process
            content_type: Type of modal content
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (enhanced_caption, entity_info)
        """
        analysis = await analysis_chain("generic_analysis").ainvoke(
            {
                "context_block": self._context_block(modal_content),
                "content_type": content_type,
                "entity_name": entity_name
                or f"descriptive name for this {content_type}",
                "content": str(modal_content),
            },
            config=self._cache_config(cache_keys=cache_keys),
        )
        enhanced_caption, entity_info = self._unpack_analysis(
            analysis, entity_name, content_type
        )

        return enhanced_caption, entity_info

