"""Retrieval-side helpers — used while answering, never while ingesting.

Shaping the retrieved payload for the caller (``convert_to_user_format``,
``generate_reference_list_from_chunks``) and the two chunk-picking strategies
the context builder chooses between.
"""


import re
from collections import Counter
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hybrid_rag.core.base import BaseKVStorage, BaseVectorStorage

from hybrid_rag.core.chunk_schema import format_location
from hybrid_rag.core.utils import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# Image chunks carry their image as "Image Path: <url>" (multimodal_template.py).
IMAGE_PATH_PATTERN = re.compile(
    r"Image Path:\s*([^\r\n]+?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif)(?:\?[^\r\n]*)?)"
)


def pick_by_weighted_polling(
    entities_or_relations: list[dict],
    max_related_chunks: int,
    min_related_chunks: int = 1,
) -> list[str]:
    """
    Linear gradient weighted polling algorithm for text chunk selection.

    This algorithm ensures that entities/relations with higher importance get more text chunks,
    forming a linear decreasing allocation pattern.

    Args:
        entities_or_relations: List of entities or relations sorted by importance (high to low)
        max_related_chunks: Expected number of text chunks for the highest importance entity/relation
        min_related_chunks: Expected number of text chunks for the lowest importance entity/relation

    Returns:
        List of selected text chunk IDs
    """
    if not entities_or_relations:
        return []

    n = len(entities_or_relations)
    if n == 1:
        # Only one entity/relation, return its first max_related_chunks text chunks
        entity_chunks = entities_or_relations[0].get("sorted_chunks", [])
        return entity_chunks[:max_related_chunks]

    # Calculate expected text chunk count for each position (linear decrease)
    expected_counts = []
    for i in range(n):
        # Linear interpolation: from max_related_chunks to min_related_chunks
        ratio = i / (n - 1) if n > 1 else 0
        expected = max_related_chunks - ratio * (
            max_related_chunks - min_related_chunks
        )
        expected_counts.append(int(round(expected)))

    # First round allocation: allocate by expected values
    selected_chunks = []
    used_counts = []  # Track number of chunks used by each entity
    total_remaining = 0  # Accumulate remaining quotas

    for i, entity_rel in enumerate(entities_or_relations):
        entity_chunks = entity_rel.get("sorted_chunks", [])
        expected = expected_counts[i]

        # Actual allocatable count
        actual = min(expected, len(entity_chunks))
        selected_chunks.extend(entity_chunks[:actual])
        used_counts.append(actual)

        # Accumulate remaining quota
        remaining = expected - actual
        if remaining > 0:
            total_remaining += remaining

    # Second round allocation: multi-round scanning to allocate remaining quotas
    for _ in range(total_remaining):
        allocated = False

        # Scan entities one by one, allocate one chunk when finding unused chunks
        for i, entity_rel in enumerate(entities_or_relations):
            entity_chunks = entity_rel.get("sorted_chunks", [])

            # Check if there are still unused chunks
            if used_counts[i] < len(entity_chunks):
                # Allocate one chunk
                selected_chunks.append(entity_chunks[used_counts[i]])
                used_counts[i] += 1
                allocated = True
                break

        # If no chunks were allocated in this round, all entities are exhausted
        if not allocated:
            break

    return selected_chunks

async def pick_by_vector_similarity(
    query: str,
    text_chunks_storage: "BaseKVStorage",
    chunks_vdb: "BaseVectorStorage",
    num_of_chunks: int,
    entity_info: list[dict[str, Any]],
    embedding_func: callable,
    query_embedding=None,
) -> list[str]:
    """
    Vector similarity-based text chunk selection algorithm.

    This algorithm selects text chunks based on cosine similarity between
    the query embedding and text chunk embeddings.

    Args:
        query: User's original query string
        text_chunks_storage: Text chunks storage instance
        chunks_vdb: Vector database storage for chunks
        num_of_chunks: Number of chunks to select
        entity_info: List of entity information containing chunk IDs
        embedding_func: Embedding function to compute query embedding

    Returns:
        List of selected text chunk IDs sorted by similarity (highest first)
    """
    logger.debug(
        f"Vector similarity chunk selection: num_of_chunks={num_of_chunks}, entity_info_count={len(entity_info) if entity_info else 0}"
    )

    if not entity_info or num_of_chunks <= 0:
        return []

    # Collect all unique chunk IDs from entity info
    all_chunk_ids = set()
    for i, entity in enumerate(entity_info):
        chunk_ids = entity.get("sorted_chunks", [])
        all_chunk_ids.update(chunk_ids)

    if not all_chunk_ids:
        logger.warning(
            "Vector similarity chunk selection:  no chunk IDs found in entity_info"
        )
        return []

    logger.debug(
        f"Vector similarity chunk selection: {len(all_chunk_ids)} unique chunk IDs collected"
    )

    all_chunk_ids = list(all_chunk_ids)

    try:
        # Use pre-computed query embedding if provided, otherwise compute it
        if query_embedding is None:
            query_embedding = await embedding_func([query], context="query")
            query_embedding = query_embedding[
                0
            ]  # Extract first embedding from batch result
            logger.debug(
                "Computed query embedding for vector similarity chunk selection"
            )
        else:
            logger.debug(
                "Using pre-computed query embedding for vector similarity chunk selection"
            )

        # Get chunk embeddings from vector database
        chunk_vectors = await chunks_vdb.get_vectors_by_ids(all_chunk_ids)
        logger.debug(
            f"Vector similarity chunk selection: {len(chunk_vectors)} chunk vectors Retrieved"
        )

        if not chunk_vectors or len(chunk_vectors) != len(all_chunk_ids):
            if not chunk_vectors:
                logger.warning(
                    "Vector similarity chunk selection: no vectors retrieved from chunks_vdb"
                )
            else:
                logger.warning(
                    f"Vector similarity chunk selection: found {len(chunk_vectors)} but expecting {len(all_chunk_ids)}"
                )
            return []

        # Calculate cosine similarities
        similarities = []
        for chunk_id in all_chunk_ids:
            if chunk_id in chunk_vectors:
                chunk_embedding = chunk_vectors[chunk_id]
                try:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append((chunk_id, similarity))
                except Exception as e:
                    logger.warning(
                        f"Vector similarity chunk selection: failed to calculate similarity for chunk {chunk_id}: {e}"
                    )
            else:
                logger.warning(
                    f"Vector similarity chunk selection:  no vector found for chunk {chunk_id}"
                )

        # Sort by similarity (highest first) and select top num_of_chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        selected_chunks = [chunk_id for chunk_id, _ in similarities[:num_of_chunks]]

        logger.debug(
            f"Vector similarity chunk selection: {len(selected_chunks)} chunks from {len(all_chunk_ids)} candidates"
        )

        return selected_chunks

    except Exception as e:
        logger.error(f"[VECTOR_SIMILARITY] Error in vector similarity sorting: {e}")
        import traceback

        logger.error(f"[VECTOR_SIMILARITY] Traceback: {traceback.format_exc()}")
        # Fallback to simple truncation
        logger.debug("[VECTOR_SIMILARITY] Falling back to simple truncation")
        return all_chunk_ids[:num_of_chunks]

def convert_to_user_format(
    entities_context: list[dict],
    relations_context: list[dict],
    chunks: list[dict],
    references: list[dict],
    query_mode: str,
    entity_id_to_original: dict = None,
    relation_id_to_original: dict = None,
) -> dict[str, Any]:
    """Convert internal data format to user-friendly format using original database data"""

    # Convert entities format using original data when available
    formatted_entities = []
    for entity in entities_context:
        entity_name = entity.get("entity", "")

        # Try to get original data first
        original_entity = None
        if entity_id_to_original and entity_name in entity_id_to_original:
            original_entity = entity_id_to_original[entity_name]

        if original_entity:
            # Use original database data
            formatted_entities.append(
                {
                    "entity_name": original_entity.get("entity_name", entity_name),
                    "entity_type": original_entity.get("entity_type", "UNKNOWN"),
                    "description": original_entity.get("description", ""),
                    "source_id": original_entity.get("source_id", ""),
                    "file_path": original_entity.get("file_path", "unknown_source"),
                    "created_at": original_entity.get("created_at", ""),
                }
            )
        else:
            # Fallback to LLM context data (for backward compatibility)
            formatted_entities.append(
                {
                    "entity_name": entity_name,
                    "entity_type": entity.get("type", "UNKNOWN"),
                    "description": entity.get("description", ""),
                    "source_id": entity.get("source_id", ""),
                    "file_path": entity.get("file_path", "unknown_source"),
                    "created_at": entity.get("created_at", ""),
                }
            )

    # Convert relationships format using original data when available
    formatted_relationships = []
    for relation in relations_context:
        entity1 = relation.get("entity1", "")
        entity2 = relation.get("entity2", "")
        relation_key = (entity1, entity2)

        # Try to get original data first
        original_relation = None
        if relation_id_to_original and relation_key in relation_id_to_original:
            original_relation = relation_id_to_original[relation_key]

        if original_relation:
            # Use original database data
            formatted_relationships.append(
                {
                    "src_id": original_relation.get("src_id", entity1),
                    "tgt_id": original_relation.get("tgt_id", entity2),
                    "description": original_relation.get("description", ""),
                    "keywords": original_relation.get("keywords", ""),
                    "weight": original_relation.get("weight", 1.0),
                    "source_id": original_relation.get("source_id", ""),
                    "file_path": original_relation.get("file_path", "unknown_source"),
                    "created_at": original_relation.get("created_at", ""),
                }
            )
        else:
            # Fallback to LLM context data (for backward compatibility)
            formatted_relationships.append(
                {
                    "src_id": entity1,
                    "tgt_id": entity2,
                    "description": relation.get("description", ""),
                    "keywords": relation.get("keywords", ""),
                    "weight": relation.get("weight", 1.0),
                    "source_id": relation.get("source_id", ""),
                    "file_path": relation.get("file_path", "unknown_source"),
                    "created_at": relation.get("created_at", ""),
                }
            )

    # Convert chunks format (chunks already contain complete data)
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        image = IMAGE_PATH_PATTERN.search(content)
        # Citation fields exist only when QueryParam.include_citations attached them.
        chunk_data = {
            **{k: chunk[k] for k in ("reference_id", "page_start", "page_end") if k in chunk},
            "content": content,
            "file_path": chunk.get("file_path", "unknown_source"),
            "chunk_id": chunk.get("chunk_id", ""),
            **({"image_path": image.group(1)} if image else {}),
        }
        formatted_chunks.append(chunk_data)

    logger.debug(
        f"[convert_to_user_format] Formatted {len(formatted_chunks)}/{len(chunks)} chunks"
    )

    # Build basic metadata (metadata details will be added by calling functions)
    metadata = {
        "query_mode": query_mode,
        "keywords": {
            "high_level": [],
            "low_level": [],
        },  # Placeholder, will be set by calling functions
    }

    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "entities": formatted_entities,
            "relationships": formatted_relationships,
            "chunks": formatted_chunks,
            "references": references,
        },
        "metadata": metadata,
    }

def generate_reference_list_from_chunks(
    chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Give each chunk a ``reference_id``: one per (file, location), most cited first.

    Ties keep first-appearance order; chunks without a known file get ``""``.
    Returns ``(reference_list, chunks_with_reference_ids)``. A reference is
    ``{"reference_id", "file_path"}`` plus ``"location"`` (``hal. 3–4``,
    ``sheet "Rekap"``) when the chunk has one.
    """

    def key(chunk: dict) -> tuple[str, str | None] | None:
        file_path = chunk.get("file_path", "")
        if not file_path or file_path == "unknown_source":
            return None
        return file_path, format_location(chunk.get("page_start"), chunk.get("page_end"))

    keys = [key(chunk) for chunk in chunks]
    counts = Counter(k for k in keys if k)
    # Counter keeps insertion order and sorted() is stable, so ties stay in appearance order.
    ranked = sorted(counts, key=lambda k: -counts[k])
    ref_ids = {k: str(i) for i, k in enumerate(ranked, 1)}

    reference_list = [
        {"reference_id": ref_ids[k], "file_path": k[0], **({"location": k[1]} if k[1] else {})}
        for k in ranked
    ]
    updated_chunks = [{**chunk, "reference_id": ref_ids.get(k, "")} for chunk, k in zip(chunks, keys)]
    return reference_list, updated_chunks
