"""Points entity extraction results, made on parents, at the children that are kept."""
import logging
from hybrid_rag.core.constants import GRAPH_FIELD_SEP

logger = logging.getLogger(__name__)


def redirect_source_ids(
    chunk_results: list,
    parent_to_child_map: dict[str, list[str]],
) -> None:
    """In-place replace source_id in extraction results from parent → child keys.

    After entity/relation extraction runs on parent chunks, each entity's
    source_id points to a parent chunk_key. This function redirects those
    to the child chunk_keys so that graph retrieval returns child chunks.

    Args:
        chunk_results: List of (maybe_nodes, maybe_edges) from _process_extract_entities.
        parent_to_child_map: ``{parent_chunk_key: [child_chunk_keys]}``, dibangun
            di worker dari ``parent_index`` tiap child (lihat pipeline.py).
    """
    redirected_count = 0

    for maybe_nodes, maybe_edges in chunk_results:
        for _entity_name, entities in maybe_nodes.items():
            for entity in entities:
                source_id = entity.get("source_id", "")
                if source_id in parent_to_child_map:
                    entity["source_id"] = GRAPH_FIELD_SEP.join(
                        parent_to_child_map[source_id]
                    )
                    redirected_count += 1

        for _edge_key, edges in maybe_edges.items():
            for edge in edges:
                source_id = edge.get("source_id", "")
                if source_id in parent_to_child_map:
                    edge["source_id"] = GRAPH_FIELD_SEP.join(
                        parent_to_child_map[source_id]
                    )
                    redirected_count += 1

    logger.info(f"[parent-child] Redirected {redirected_count} source_ids")
