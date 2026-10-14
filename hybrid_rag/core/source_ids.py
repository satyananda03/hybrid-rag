"""Source-id bookkeeping shared by ingestion (merge, limit) and deletion (subtract)."""

import logging
from typing import Collection, Iterable, Sequence

from hybrid_rag.core.constants import (
    GRAPH_FIELD_SEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    VALID_SOURCE_IDS_LIMIT_METHODS,
)

logger = logging.getLogger(__name__)


def normalize_source_ids_limit_method(method: str | None) -> str:
    """Normalize the source ID limiting strategy and fall back to default when invalid."""

    if not method:
        return SOURCE_IDS_LIMIT_METHOD_KEEP

    normalized = method.upper()
    if normalized not in VALID_SOURCE_IDS_LIMIT_METHODS:
        logger.warning(
            "Unknown SOURCE_IDS_LIMIT_METHOD '%s', falling back to %s",
            method,
            SOURCE_IDS_LIMIT_METHOD_KEEP,
        )
        return SOURCE_IDS_LIMIT_METHOD_KEEP

    return normalized


def subtract_source_ids(
    source_ids: Iterable[str],
    ids_to_remove: Collection[str],
) -> list[str]:
    """Remove a collection of IDs from an ordered iterable while preserving order."""

    removal_set = set(ids_to_remove)
    if not removal_set:
        return [source_id for source_id in source_ids if source_id]

    return [
        source_id
        for source_id in source_ids
        if source_id and source_id not in removal_set
    ]


def make_relation_chunk_key(src: str, tgt: str) -> str:
    """Create a deterministic storage key for relation chunk tracking."""

    return GRAPH_FIELD_SEP.join(sorted((src, tgt)))


def merge_source_ids(
    existing_ids: Iterable[str] | None, new_ids: Iterable[str] | None
) -> list[str]:
    """Merge two iterables of source IDs while preserving order and removing duplicates."""

    merged: list[str] = []
    seen: set[str] = set()

    for sequence in (existing_ids, new_ids):
        if not sequence:
            continue
        for source_id in sequence:
            if not source_id:
                continue
            if source_id not in seen:
                seen.add(source_id)
                merged.append(source_id)

    return merged


def apply_source_ids_limit(
    source_ids: Sequence[str],
    limit: int,
    method: str,
    *,
    identifier: str | None = None,
) -> list[str]:
    """Apply a limit strategy to a sequence of source IDs."""

    if limit <= 0:
        return []

    source_ids_list = list(source_ids)
    if len(source_ids_list) <= limit:
        return source_ids_list

    normalized_method = normalize_source_ids_limit_method(method)

    if normalized_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
        truncated = source_ids_list[-limit:]
    else:  # IGNORE_NEW
        truncated = source_ids_list[:limit]

    if identifier and len(truncated) < len(source_ids_list):
        logger.debug(
            "Source_id truncated: %s | %s keeping %s of %s entries",
            identifier,
            normalized_method,
            len(truncated),
            len(source_ids_list),
        )

    return truncated
