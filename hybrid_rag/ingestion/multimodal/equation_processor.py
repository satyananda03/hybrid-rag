
"""Equation — described from its LaTeX/MathML body and surrounding text."""

import json
from typing import Dict, Any, Tuple, List

import logging

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass

# Import prompt templates
from hybrid_rag.ingestion.multimodal.base_processor import BaseModalProcessor
from hybrid_rag.ingestion.multimodal.chain import equation_analysis_chain
from hybrid_rag.ingestion.multimodal.content import get_equation_text_and_format

logger = logging.getLogger(__name__)

class EquationModalProcessor(BaseModalProcessor):
    """Processor specialized for equation content"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
        cache_keys: List[str] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate equation description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Equation content to process
            content_type: Type of modal content ("equation")
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (enhanced_caption, entity_info)
        """
        # Parse equation content (reuse existing logic)
        if isinstance(modal_content, str):
            try:
                content_data = json.loads(modal_content)
            except json.JSONDecodeError:
                content_data = {"equation": modal_content}
        else:
            content_data = modal_content

        equation_text, equation_format = get_equation_text_and_format(content_data)

        analysis = await equation_analysis_chain().ainvoke(
            {
                "context_block": self._context_block(content_data),
                "entity_name": entity_name or "descriptive name for this equation",
                "equation_text": equation_text,
                "equation_format": equation_format,
            },
            config=self._cache_config(cache_keys=cache_keys),
        )
        enhanced_caption, entity_info = self._unpack_analysis(
            analysis, entity_name, "equation"
        )

        return enhanced_caption, entity_info

