"""Write one database schema into the RAG's storages.

Mixed into :class:`~hybrid_rag.ingestion.rag.IngestionRAG`, so ``self`` is the RAG:
the same storages, merge and flush the document pipeline uses, entered through a
path of its own.

**Status handling.** The document is FAILED ("ingest in progress") while it is
being written and PROCESSED once the flush succeeds. FAILED, not PENDING or
PROCESSING: the document queue deletes an in-flight row that has no ``full_docs``
body, but preserves a FAILED one — so a crash part way still leaves the row that
``delete_document`` or the next ingest uses to clean up. ``full_docs`` is never
written, which keeps the queue from ever picking this document up.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from hybrid_rag.config import settings
from hybrid_rag.core.base import DocStatus
from hybrid_rag.core.utils import compute_mdhash_id
from hybrid_rag.ingestion.db_schema.records import SchemaRecords, build_records
from hybrid_rag.ingestion.db_schema.spec import DatabaseSpec
from hybrid_rag.ingestion.doc_status_fields import compute_text_content_hash
from hybrid_rag.ingestion.knowledge_graph.extraction import merge_nodes_and_edges
from hybrid_rag.ingestion.text_cleaning import get_content_summary

logger = logging.getLogger(__name__)

SOURCE_TYPE = "db_schema"
_IN_PROGRESS = "ingest in progress"
_NAMES_IN_ERROR = 5


def schema_document(database: str) -> tuple[str, str]:
    """``(doc_id, file_path)`` of a database's schema document.

    Derived from the database name, not the YAML file name, so a re-ingest of the
    same database always replaces its previous version.
    """
    file_path = f"{SOURCE_TYPE}:{database}"
    return compute_mdhash_id(file_path, prefix="doc-"), file_path


class _DbSchemaMixin:
    """Ingest a database schema as deterministic graph + chunks — no LLM call."""

    async def ainsert_db_schema(self, spec: DatabaseSpec) -> dict[str, Any]:
        """Index ``spec`` into this workspace, replacing any previous version.

        **Concurrency:** like document ingest and delete, serializing writers of a
        workspace is the caller's job (the app's ``workspace_write_turn``).

        Returns:
            ``status`` is ``"unchanged"`` when the stored version has the same
            content, else ``"processed"`` with the counts written.

        Raises:
            ValueError: an entity this schema would write already belongs to
                another document (e.g. another database defining the same
                ``schema.table`` in this workspace).
            RuntimeError: the previous version could not be deleted.
        """
        doc_id, file_path = schema_document(spec.name)
        content_hash = compute_text_content_hash(spec.model_dump_json())
        result = {"doc_id": doc_id, "workspace": self.workspace}

        existing = await self.doc_status.get_by_id(doc_id)
        if existing:
            if existing.get("status") == DocStatus.PROCESSED and existing.get("content_hash") == content_hash:
                logger.info(f"[db_schema] {spec.name}: unchanged, nothing to do")
                return {**result, "status": "unchanged"}
            await self._delete_previous_schema(doc_id)

        records = build_records(
            spec,
            doc_id=doc_id,
            file_path=file_path,
            tokenizer=self.tokenizer,
            max_tokens=self.embedding_token_limit or settings.embedding_token_limit,
        )
        await self._refuse_foreign_entities(records)

        status_row = _status_row(records, file_path, content_hash)
        await self.doc_status.upsert(
            {doc_id: {**status_row, "status": DocStatus.FAILED, "error_msg": _IN_PROGRESS}}
        )
        try:
            await self._write_schema_records(doc_id, file_path, records)
        except Exception as e:
            await self._discard_pending_index_ops()
            await self.doc_status.upsert(
                {doc_id: {**status_row, "status": DocStatus.FAILED, "error_msg": f"{type(e).__name__}: {e}"}}
            )
            raise
        await self.doc_status.upsert({doc_id: {**status_row, "status": DocStatus.PROCESSED}})

        logger.info(f"[db_schema] {spec.name}: indexed {records.stats} into {self.workspace!r}")
        return {
            **result,
            "status": "processed",
            **records.stats,
            "chunks": len(records.chunks),
            "entities": len(records.nodes),
            "relations": len(records.edges),
        }

    async def _delete_previous_schema(self, doc_id: str) -> None:
        # Every entity of the old version is owned by this document alone, so the
        # delete removes it outright; nothing is left to rebuild from the LLM cache.
        deleted = await self.adelete_by_doc_id(doc_id)
        if deleted.status != "success":
            raise RuntimeError(f"could not replace schema document {doc_id}: {deleted.message}")

    async def _refuse_foreign_entities(self, records: SchemaRecords) -> None:
        # The previous version is gone by now, so any node still holding one of
        # these names is another document's. Merging into it would mix the two
        # documents' sources, and deleting either one could not undo that.
        existing = await self.chunk_entity_relation_graph.get_nodes_batch(list(records.nodes))
        if existing:
            names = sorted(existing)
            listed = ", ".join(names[:_NAMES_IN_ERROR]) + (" ..." if len(names) > _NAMES_IN_ERROR else "")
            raise ValueError(
                f"{len(names)} entity name(s) of this schema already exist in workspace "
                f"{self.workspace!r} from another document: {listed}"
            )

    async def _write_schema_records(self, doc_id: str, file_path: str, records: SchemaRecords) -> None:
        await asyncio.gather(
            self.text_chunks.upsert(records.chunks),
            self.chunks_vdb.upsert(records.chunks),
        )
        # Same storages as the document pipeline's merge. No resolver: the names
        # are exact identifiers, never variants of one another.
        await merge_nodes_and_edges(
            chunk_results=records.chunk_results,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entity_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            global_config={
                **self._build_global_config(),
                "kg_entity_registry": None,
                "entity_resolution_llm": False,
            },
            full_entities_storage=self.full_entities,
            full_relations_storage=self.full_relations,
            doc_id=doc_id,
            llm_response_cache=self.llm_response_cache,
            entity_chunks_storage=self.entity_chunks,
            relation_chunks_storage=self.relation_chunks,
            file_path=file_path,
            entity_resolver=None,
        )
        # Embeds and writes the buffered vectors; PROCESSED only after it succeeds.
        await self._insert_done()


def _status_row(records: SchemaRecords, file_path: str, content_hash: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    overview = next(iter(records.chunks.values()))["content"]
    return {
        "content_summary": get_content_summary(overview),
        "content_length": sum(len(c["content"]) for c in records.chunks.values()),
        "file_path": file_path,
        "chunks_count": len(records.chunks),
        "chunks_list": list(records.chunks),
        "content_hash": content_hash,
        "metadata": {"source_type": SOURCE_TYPE, **records.stats},
        "created_at": now,
        "updated_at": now,
    }
