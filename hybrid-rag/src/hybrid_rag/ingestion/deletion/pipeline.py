
"""Deletion entry points.

Removing a document is not a single delete: its text lives in ``full_docs``, its
chunks in a KV store and a vector index, and its entities and relations in the
graph — where they are usually *shared* with other documents. So the work is:

1. Read the document's chunk ids from ``doc_status``.
2. For every entity and relation the document contributed to, subtract its
   chunks from that entity's sources. Nothing left → delete it outright.
   Something left → keep it, and rebuild its description from the chunks that
   other documents still provide.
3. Delete the chunks, the orphaned graph elements, the per-document indexes,
   the LLM cache entries the document produced, and finally the document row.

That is :meth:`IngestionRAG.adelete_by_doc_id`; this module is the thin entry point
around it, resolving the workspace so a delete only ever touches the tenant that
owns the document.

**Deletion is not transactional across storages.** PostgreSQL and Neo4j are
separate systems, and vector deletes are buffered until a flush. A failure part
way through is therefore expected to happen; ``adelete_by_doc_id`` records its
progress in ``doc_status.metadata`` (``deletion_stage`` / ``deletion_failed`` /
``deletion_llm_cache_ids``) and deletes ``doc_status`` last, so a failed delete
can simply be re-run and picks up where it stopped. Never "clean up" a failed
delete by removing the doc_status row first — that is the only handle left to
retry with.
"""


from hybrid_rag.core.base import DeletionResult
import logging

from hybrid_rag.ingestion.workspace import workspace_pool

logger = logging.getLogger(__name__)


async def delete_document(
    doc_id: str,
    *,
    workspace: str,
    delete_llm_cache: bool = True,
) -> DeletionResult:
    """Delete one document and everything derived from it.

    Args:
        doc_id: Document to remove.
        workspace: Workspace that owns it. Required, never defaulted: a shared
            fallback would let a delete cross the workspace boundary.
        delete_llm_cache: Also drop the cached LLM responses produced while
            extracting this document. On by default: those entries are keyed to
            content that is going away, and keeping them leaks storage that
            nothing will ever look up again. Pass ``False`` to keep them when
            the same content is about to be re-ingested.

    Returns:
        DeletionResult: ``status`` is ``"success"``, ``"not_found"`` or
        ``"fail"`` — a failure is reported, not raised, so a caller deleting
        several documents can carry on.
    """
    async with workspace_pool.lease(workspace) as rag:
        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=delete_llm_cache)

    if result.status != "success":
        logger.warning(f"[delete] {doc_id}: {result.status} — {result.message}")
    return result
