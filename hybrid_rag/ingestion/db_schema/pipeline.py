"""Database-schema ingestion entry point."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hybrid_rag.ingestion.db_schema.spec import DatabaseSpec, load_spec
from hybrid_rag.ingestion.workspace import workspace_pool


async def ingest_db_schema(
    source: DatabaseSpec | Mapping[str, Any] | str | Path,
    *,
    workspace: str,
) -> dict[str, Any]:
    """Index one database schema into ``workspace``, replacing its previous version.

    Args:
        source: the schema YAML's path, its parsed mapping, or a ``DatabaseSpec``.
        workspace: Workspace to write into. Required, never defaulted — it may be
            a dedicated one or the one holding the documents; schema entities
            never merge with document entities either way.

    Returns:
        ``status`` (``"processed"`` or ``"unchanged"``), ``doc_id``, ``workspace``
        and, when processed, the counts written. ``doc_id`` is what
        ``delete_document`` takes to remove the schema again.

    Raises:
        ValueError: ``source`` is not a valid schema, or its entity names are
            already owned by another document in ``workspace``.
    """
    spec = load_spec(source)  # validate before leasing a workspace
    async with workspace_pool.lease(workspace) as rag:
        return await rag.ainsert_db_schema(spec)
