
"""Table — described from its rows, caption and footnotes. Text-only, no VLM."""

import json
from typing import Dict, Any, Tuple, List

import logging

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass

# Import prompt templates
from hybrid_rag.ingestion.multimodal.base_processor import BaseModalProcessor
from hybrid_rag.ingestion.multimodal.vlm import table_analysis_chain
from hybrid_rag.ingestion.multimodal.content import (
    format_table_body,
    get_table_body,
    normalize_caption_list,
)

logger = logging.getLogger(__name__)

class TableModalProcessor(BaseModalProcessor):
    """Processor specialized for table content"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
        cache_keys: List[str] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate table description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Table content to process
            content_type: Type of modal content ("table")
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (enhanced_caption, entity_info)
        """
        # Parse table content (reuse existing logic)
        if isinstance(modal_content, str):
            try:
                content_data = json.loads(modal_content)
            except json.JSONDecodeError:
                content_data = {"table_body": modal_content}
        else:
            content_data = modal_content

        table_caption = normalize_caption_list(content_data.get("table_caption"))
        table_body = format_table_body(get_table_body(content_data))
        table_footnote = normalize_caption_list(content_data.get("table_footnote"))

        analysis = await table_analysis_chain().ainvoke(
            {
                "context_block": self._context_block(content_data),
                "entity_name": entity_name or "descriptive name for this table",
                "table_caption": table_caption or "None",
                "table_body": table_body,
                "table_footnote": table_footnote or "None",
            },
            config=self._cache_config(cache_keys=cache_keys),
        )
        enhanced_caption, entity_info = self._unpack_analysis(
            analysis, entity_name, "table"
        )
        # print("\n" + "="*70)
        # print(f"📊  TABLE PROCESSING RESULT")
        # print(f"  entity_name  : {entity_info['entity_name']}")
        # print(f"  entity_type  : {entity_info['entity_type']}")
        # print(f"  summary      : {entity_info['summary']}")
        # print(f"  caption      : {table_caption}")
        # print(f"─"*70)
        # print(f"  table_body (dikirim ke LLM):")
        # print(table_body[:500])  # preview
        # print(f"─"*70)
        # print(f"  CHUNK yang akan disimpan (LLM analysis):")
        # print(enhanced_caption)
        # print("="*70 + "\n")
        return enhanced_caption, entity_info

