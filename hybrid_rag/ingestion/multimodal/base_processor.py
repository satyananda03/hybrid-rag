
"""The shared half of every modal processor.

:class:`BaseModalProcessor` owns everything that does not depend on the
modality: the cache key, the context block and unpacking the model's answer.
Subclasses supply only ``generate_description_only`` — how to ask the model
about *this* modality.
"""

from typing import Dict, Any, Tuple, List

import logging

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from hybrid_rag.ingestion.rag import IngestionRAG

# Import prompt templates
from hybrid_rag.core.llm_cache import CACHE_CHUNK_ID, CACHE_KEYS, CACHE_KV
from hybrid_rag.ingestion.multimodal.vlm import (
    ImageAnalysis,
    TableAnalysis,
    EquationAnalysis,
    surrounding_context_block,
)

logger = logging.getLogger(__name__)


class BaseModalProcessor:
    """Base class for modal processors"""

    def __init__(self, rag: "IngestionRAG"):
        """Initialize base processor

        Args:
            rag: IngestionRAG instance
        """
        self.rag = rag

        # The instance's own storages
        self.text_chunks_db = rag.text_chunks
        self.chunks_vdb = rag.chunks_vdb
        self.entities_vdb = rag.entities_vdb
        self.relationships_vdb = rag.relationships_vdb
        self.knowledge_graph_inst = rag.chunk_entity_relation_graph

        self.embedding_func = rag.embedding_func
        # The same config the storages got; asdict() turns the EmbeddingFunc
        # into a plain dict, so the real callable is put back.
        self.global_config = rag._build_global_config()
        self.global_config["embedding_func"] = self.embedding_func
        self.hashing_kv = rag.llm_response_cache
        self.tokenizer = rag.tokenizer

    def _context_block(self, content_data: Dict[str, Any] | None) -> str:
        """Render the surrounding-text block for an item, "" when it has none.

        Teks tetangganya sudah menempel di item sejak ``separate_content`` —
        dihitung sekali, ikut tersimpan ke ``full_docs.multimodal_items``, jadi
        tetap ada meski analisis berjalan dari database di run berikutnya.
        """
        if not content_data:
            return ""
        return surrounding_context_block(
            content_data.get("_leading_text", ""),
            content_data.get("_trailing_text", ""),
        )

    def _cache_config(
        self, chunk_id: str | None = None, cache_keys: list[str] | None = None
    ) -> dict:
        """Config that lets an analysis chain use this instance's LLM cache.

        The multimodal chunk this analysis becomes does not exist yet, so its
        cache keys cannot be stamped on it here. Pass ``cache_keys`` and the
        caller can carry them onto the chunk once it is built — otherwise the
        entries survive deletion of the document that created them.
        """
        return {
            "configurable": {
                CACHE_KV: self.hashing_kv,
                CACHE_CHUNK_ID: chunk_id,
                CACHE_KEYS: cache_keys,
            }
        }

    @staticmethod
    def _unpack_analysis(
        analysis: ImageAnalysis | TableAnalysis | EquationAnalysis | None,
        entity_name: str | None,
        modality: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Turn a chain result into ``(description, entity_info)``.

        The model may propose its own name; an explicit ``entity_name`` from the
        caller always wins, since that is what the chunk and the graph were
        keyed on. When structured output failed there is nothing to describe, so
        raise: the caller fails the document rather than index a placeholder.
        """
        if analysis is None:
            raise ValueError(f"{modality} analysis returned no structured output")
        # The schema guarantees the field exists, not that the model filled it;
        # an empty description would become an empty chunk.
        if not analysis.detailed_description.strip():
            raise ValueError(f"{modality} analysis returned an empty description")

        entity_info = analysis.entity_info.model_dump()
        entity_info["entity_type"] = entity_info.get("entity_type") or modality
        entity_info["entity_name"] = (
            entity_name
            or f"{entity_info['entity_name']} ({entity_info['entity_type']})"
        )
        return analysis.detailed_description, entity_info

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
        cache_keys: List[str] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Modal content to process
            content_type: Type of modal content
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (description, entity_info)
        """
        # Subclasses must implement this method
        raise NotImplementedError("Subclasses must implement this method")

