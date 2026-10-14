
"""Document deletion — mixed into :class:`IngestionRAG`.

Deleting a document is the inverse of ingesting one and touches every store:
chunk rows, the chunks/entities/relations vector tables, the Neo4j nodes and
edges, the LLM extraction cache, and doc status. The graph part — removing the
document's chunks and rebuilding entities/relations other documents still use —
is shared with the pipeline's resume path and lives in
:mod:`hybrid_rag.ingestion.knowledge_graph.purge`. This module adds what only a
delete needs: retry state, LLM-cache cleanup, and removing the document rows.

``self`` here is the :class:`IngestionRAG` instance — the storages are its fields.
"""


import logging
import traceback
from datetime import datetime, timezone
from typing import Any

from hybrid_rag import progress
from hybrid_rag.core.base import DeletionResult, DocStatus
from hybrid_rag.core.utils import normalize_string_list

logger = logging.getLogger(__name__)


class _DeletionMixin:
    """Delete by document, by entity, or by relation."""
    async def _update_delete_retry_state(
        self,
        doc_id: str,
        doc_status_data: dict[str, Any],
        *,
        deletion_stage: str,
        doc_llm_cache_ids: list[str],
        error_message: str | None = None,
        failed: bool,
    ) -> dict[str, Any]:
        """Persist deletion retry metadata and return the updated status record."""
        metadata = doc_status_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        backup_cache_ids = normalize_string_list(
            metadata.get("deletion_llm_cache_ids", []),
            context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
        )
        retry_cache_ids = doc_llm_cache_ids or backup_cache_ids

        updated_metadata = dict(metadata)
        if retry_cache_ids:
            updated_metadata["deletion_llm_cache_ids"] = retry_cache_ids
        updated_metadata["last_deletion_attempt_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        if failed:
            updated_metadata["deletion_failed"] = True
            updated_metadata["deletion_failure_stage"] = deletion_stage
        else:
            updated_metadata.pop("deletion_failed", None)
            updated_metadata.pop("deletion_failure_stage", None)

        updated_status_data = {
            **doc_status_data,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": updated_metadata,
            "error_msg": error_message if failed else "",
        }

        await self.doc_status.upsert({doc_id: updated_status_data})
        return updated_status_data

    async def _get_existing_llm_cache_ids(self, cache_ids: list[str]) -> list[str]:
        """Return cache IDs that still exist in cache storage.

        Some KV storage backends only log delete failures and return without
        raising, so callers must verify which records still exist after delete.

        Returns an empty list immediately if cache storage is unavailable.
        Callers must check storage availability independently before treating
        an empty result as a confirmed deletion.
        """
        if not self.llm_response_cache or not cache_ids:
            return []

        try:
            existing_records = await self.llm_response_cache.get_by_ids(cache_ids)
        except Exception as verification_error:
            raise Exception(
                f"Failed to verify LLM cache deletion "
                f"(delete may have succeeded): {verification_error}"
            ) from verification_error
        return [
            cache_id
            for cache_id, record in zip(cache_ids, existing_records)
            if record is not None
        ]

    async def adelete_by_doc_id(
        self, doc_id: str, delete_llm_cache: bool = False
    ) -> DeletionResult:
        """Delete a document and all its related data, including chunks, graph elements.

        This method orchestrates a comprehensive deletion process for a given document ID.
        It ensures that not only the document itself but also all its derived and associated
        data across different storage layers are removed or rebuiled. If entities or relationships
        are partially affected, they will be rebuilded using LLM cached from remaining documents.

        **Concurrency:** serializing callers is not this method's job. The app's
        routes take the workspace's write turn (``workspace_write_turn``) around
        every ingest and delete, so one workspace never runs both at once.

        The upstream ``pipeline_status["busy"]`` handshake that used to sit
        here guarded nothing and was removed: it is per-process state, and the
        batch-deletion job it coordinated with (``job_name`` starting with
        ``"deleting"``) was never ported into this codebase, so its "join the
        deletion queue" branch could not be reached from any call path.

        Args:
            doc_id (str): The unique identifier of the document to be deleted.
            delete_llm_cache (bool): Whether to delete cached LLM extraction results
                associated with the document. Defaults to False.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
                - `status` (str): "success", "not_found", or "fail".
                - `doc_id` (str): The ID of the document attempted to be deleted.
                - `message` (str): A summary of the operation's result.
                - `status_code` (int): HTTP status code (e.g., 200, 404, 500).
                - `file_path` (str | None): The file path of the deleted document, if available.
        """
        deletion_operations_started = False
        deletion_fully_completed = False
        in_final_delete_stage = False
        original_exception = None
        doc_llm_cache_ids: list[str] = []
        deletion_stage = "initializing"
        progress.stage("initializing")

        def enter_stage(name: str) -> None:
            # Recorded as deletion_failure_stage if this step fails, and reported as progress.
            nonlocal deletion_stage
            deletion_stage = name
            progress.stage(name)
        doc_status_data: dict[str, Any] | None = None
        file_path: str | None = None

        log_message = f"Starting deletion process for document {doc_id}"
        logger.info(log_message)

        try:
            # 1. Get the document status and related data
            doc_status_data = await self.doc_status.get_by_id(doc_id)
            if not doc_status_data:
                logger.warning(f"Document {doc_id} not found")
                return DeletionResult(
                    status="not_found",
                    doc_id=doc_id,
                    message=f"Document {doc_id} not found.",
                    status_code=404,
                    file_path="",
                )
            file_path = doc_status_data.get("file_path")

            # Check document status and log warning for non-completed documents
            raw_status = doc_status_data.get("status")
            try:
                doc_status = DocStatus(raw_status)
            except ValueError:
                doc_status = raw_status

            if doc_status != DocStatus.PROCESSED:
                if doc_status == DocStatus.PENDING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PENDING)"
                    )
                elif doc_status == DocStatus.PROCESSING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PROCESSING)"
                    )
                elif doc_status == DocStatus.PREPROCESSED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PREPROCESSED)"
                    )
                elif doc_status == DocStatus.FAILED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: FAILED)"
                    )
                else:
                    status_text = (
                        doc_status.value
                        if isinstance(doc_status, DocStatus)
                        else str(doc_status)
                    )
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: {status_text})"
                    )
                logger.info(warning_msg)

            # 2. Get chunk IDs from document status
            metadata = doc_status_data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata_cache_ids = normalize_string_list(
                metadata.get("deletion_llm_cache_ids", []),
                context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
            )
            # Order-preserving dedup so chunk_ids stays a list and satisfies the
            # storage delete contract (``delete(ids: list[str])``).
            tracked_chunk_ids = normalize_string_list(
                doc_status_data.get("chunks_list", []),
                context=f"doc {doc_id} chunks_list",
            )
            stored_chunk_ids = await self._chunk_ids_from_storage(doc_id)
            chunk_ids = list(dict.fromkeys(tracked_chunk_ids + stored_chunk_ids))
            # Parent chunks are stored on purpose (``save_parent``) but never
            # listed in chunks_list, which counts children only; they are still
            # deleted through the full_doc_id lookup, so they are not drift.
            parent_prefix = f"{doc_id}-parent-"
            tracked_set = set(tracked_chunk_ids)
            untracked = sum(
                1
                for cid in chunk_ids
                if cid not in tracked_set and not cid.startswith(parent_prefix)
            )
            if untracked > 0:
                # Worth a warning, not a debug line: it means chunks_list drifted
                # from what was actually stored, which is how chunks leak.
                logger.warning(
                    f"Doc {doc_id}: {untracked} chunk(s) found via full_doc_id "
                    f"that chunks_list did not track"
                )

            if not chunk_ids:
                logger.warning(f"No chunks found for document {doc_id}")
                # Mark that deletion operations have started
                deletion_operations_started = True

                # A prior failed deletion may have collected LLM cache IDs before the
                # chunks were removed. If delete_llm_cache is requested and persisted IDs
                # exist, clean them up now before removing the doc/status entries.
                if delete_llm_cache and metadata_cache_ids:
                    if not self.llm_response_cache:
                        no_cache_msg = (
                            f"Cannot delete LLM cache for document {doc_id}: "
                            "cache storage is unavailable"
                        )
                        logger.error(no_cache_msg)
                        raise Exception(no_cache_msg)
                    try:
                        deletion_stage = "delete_llm_cache"
                        progress.stage("delete_llm_cache")
                        await self.llm_response_cache.delete(metadata_cache_ids)
                        remaining_cache_ids = await self._get_existing_llm_cache_ids(
                            metadata_cache_ids
                        )
                        if remaining_cache_ids:
                            raise Exception(
                                f"{len(remaining_cache_ids)} LLM cache entries still exist after delete"
                            )
                        logger.info(
                            "Cleaned up %d LLM cache entries from prior attempt for document %s",
                            len(metadata_cache_ids),
                            doc_id,
                        )
                    except Exception as cache_err:
                        raise Exception(
                            f"Failed to delete LLM cache for document {doc_id}: {cache_err}"
                        ) from cache_err

                try:
                    # Still need to delete the doc status and full doc.
                    # Delete doc_status first: if full_docs.delete fails on retry, the
                    # doc_status record is already gone so the retry finds no record and
                    # treats the document as already deleted rather than creating a zombie.
                    deletion_stage = "delete_doc_entries"
                    progress.stage("delete_doc_entries")
                    await self.doc_status.delete([doc_id])
                    await self.full_docs.delete([doc_id])
                except Exception as e:
                    logger.error(
                        f"Failed to delete document {doc_id} with no chunks: {e}"
                    )
                    raise Exception(f"Failed to delete document entry: {e}") from e

                log_message = (
                    f"Document deleted without associated chunks: {doc_id}"
                )
                logger.info(log_message)

                deletion_fully_completed = True
                return DeletionResult(
                    status="success",
                    doc_id=doc_id,
                    message=log_message,
                    status_code=200,
                    file_path=file_path,
                )

            # Mark that deletion operations have started
            deletion_operations_started = True

            if chunk_ids:
                # Always collect/persist cache IDs for chunk-backed documents, even when
                # this call does not request cache deletion. If a delete fails after the
                # chunks/graph have already been removed, a later retry may turn on
                # delete_llm_cache=True, and doc_status metadata is then the only durable
                # place left to recover the cache keys for cleanup.
                deletion_stage = "collect_llm_cache"
                progress.stage("collect_llm_cache")
                doc_llm_cache_ids = list(metadata_cache_ids)
                if not self.text_chunks:
                    logger.info(
                        "Skipping LLM cache id collection for document %s because text chunk storage is unavailable",
                        doc_id,
                    )
                else:
                    try:
                        chunk_data_list = await self.text_chunks.get_by_ids(chunk_ids)
                        seen_cache_ids: set[str] = set(doc_llm_cache_ids)
                        for chunk_data in chunk_data_list:
                            if not chunk_data or not isinstance(chunk_data, dict):
                                continue
                            cache_ids = chunk_data.get("llm_cache_list", [])
                            if not isinstance(cache_ids, list):
                                continue
                            for cache_id in cache_ids:
                                if (
                                    isinstance(cache_id, str)
                                    and cache_id
                                    and cache_id not in seen_cache_ids
                                ):
                                    doc_llm_cache_ids.append(cache_id)
                                    seen_cache_ids.add(cache_id)
                    except Exception as cache_collect_error:
                        logger.error(
                            "Failed to collect LLM cache ids for document %s: %s",
                            doc_id,
                            cache_collect_error,
                        )
                        raise Exception(
                            f"Failed to collect LLM cache ids for document {doc_id}: {cache_collect_error}"
                        ) from cache_collect_error

                if doc_llm_cache_ids:
                    try:
                        doc_status_data = await self._update_delete_retry_state(
                            doc_id,
                            doc_status_data,
                            deletion_stage=deletion_stage,
                            doc_llm_cache_ids=doc_llm_cache_ids,
                            failed=False,
                        )
                    except Exception as status_write_error:
                        logger.error(
                            "Failed to persist LLM cache IDs for document %s to retry state: %s",
                            doc_id,
                            status_write_error,
                        )
                        # Describe whether this is a fresh attempt or a retry so
                        # operators can tell whether prior partial deletions exist.
                        attempt_context = (
                            "retry — prior partial deletions may exist"
                            if metadata_cache_ids
                            else "deletion not yet started"
                        )
                        raise Exception(
                            f"Failed to persist LLM cache IDs for document {doc_id} "
                            f"({attempt_context}): {status_write_error}"
                        ) from status_write_error
                    logger.info(
                        "Collected %d LLM cache entries for document %s",
                        len(doc_llm_cache_ids),
                        doc_id,
                    )
                else:
                    logger.info("No LLM cache entries found for document %s", doc_id)

            # 4–8. Chunks and graph contributions: classify, update chunk tracking,
            # delete chunks / orphaned relations and entities, persist, rebuild
            # what other documents still use. Shared with the pipeline's resume path.
            await self._purge_graph_contributions(doc_id, chunk_ids, on_stage=enter_stage)

            # 9. Delete LLM cache while the document status still exists so a failure
            # remains retryable via the same doc_id.
            log_message = f"Document {doc_id} successfully deleted"
            if delete_llm_cache and doc_llm_cache_ids:
                if not self.llm_response_cache:
                    log_message = (
                        f"Cannot delete LLM cache for document {doc_id}: "
                        "cache storage is unavailable"
                    )
                    logger.error(log_message)
                    raise Exception(log_message)
                try:
                    deletion_stage = "delete_llm_cache"
                    progress.stage("delete_llm_cache")
                    await self.llm_response_cache.delete(doc_llm_cache_ids)
                    # Some storage implementations do not raise on delete errors and
                    # instead only log internally, so confirm the cache entries are
                    # actually gone before deleting the document/status records.
                    remaining_cache_ids = await self._get_existing_llm_cache_ids(
                        doc_llm_cache_ids
                    )
                    if remaining_cache_ids:
                        doc_llm_cache_ids = remaining_cache_ids
                        raise Exception(
                            f"{len(remaining_cache_ids)} LLM cache entries still exist after delete"
                        )
                    cache_log_message = f"Successfully deleted {len(doc_llm_cache_ids)} LLM cache entries for document {doc_id}"
                    logger.info(cache_log_message)
                    log_message = cache_log_message
                except Exception as cache_delete_error:
                    log_message = (
                        f"Failed to delete LLM cache for document {doc_id}: "
                        f"{cache_delete_error}"
                    )
                    logger.error(log_message)
                    logger.error(traceback.format_exc())
                    raise Exception(log_message) from cache_delete_error

            # 10. Delete from full_entities and full_relations storage — after the
            # LLM cache, so a failed cache delete still has the index to retry with.
            await self._drop_doc_graph_index(doc_id, on_stage=enter_stage)

            # 11. Delete original document and status.
            # doc_status is deleted first so that if full_docs.delete fails, a retry
            # finds no doc_status record and treats the document as already gone,
            # rather than finding a doc_status that points to a missing full_docs entry.
            try:
                deletion_stage = "delete_doc_entries"
                progress.stage("delete_doc_entries")
                in_final_delete_stage = True
                await self.doc_status.delete([doc_id])
                await self.full_docs.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete document and status: {e}")
                raise Exception(f"Failed to delete document and status: {e}") from e

            deletion_fully_completed = True
            return DeletionResult(
                status="success",
                doc_id=doc_id,
                message=log_message,
                status_code=200,
                file_path=file_path,
            )

        except Exception as e:
            original_exception = e
            error_message = f"Error while deleting document {doc_id}: {e}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            try:
                # Do not attempt to write retry state if doc_status was already deleted:
                # upsert would re-create the record as a zombie. All earlier stages still
                # have doc_status intact and can safely update it, even if some chunk/graph
                # data has already been removed.
                if doc_status_data is not None and not in_final_delete_stage:
                    doc_status_data = await self._update_delete_retry_state(
                        doc_id,
                        doc_status_data,
                        deletion_stage=deletion_stage,
                        doc_llm_cache_ids=doc_llm_cache_ids,
                        error_message=error_message,
                        failed=True,
                    )
            except Exception as status_update_error:
                logger.error(
                    "Failed to update deletion retry state for document %s: %s",
                    doc_id,
                    status_update_error,
                )
                logger.error(traceback.format_exc())
                error_message = (
                    f"{error_message}. Additionally, failed to persist retry state: "
                    f"{status_update_error}. Manual cleanup may be required."
                )
            return DeletionResult(
                status="fail",
                doc_id=doc_id,
                message=error_message,
                status_code=500,
                file_path=file_path,
            )

        finally:
            # ALWAYS ensure persistence if any deletion operations were started
            if deletion_operations_started:
                # Plain _insert_done: this finally reports the deletion as
                # successful after logging a persistence error, so discarding
                # pending DELETES here would drop them with no retry and leave
                # stale vectors/KV behind. Keep them buffered for a later flush.
                try:
                    await self._insert_done()
                except Exception as persistence_error:
                    persistence_error_msg = f"Failed to persist data after deletion attempt for {doc_id}: {persistence_error}"
                    logger.error(persistence_error_msg)
                    logger.error(traceback.format_exc())

                    if deletion_fully_completed:
                        # All deletion stages succeeded; the flush error is a post-cleanup
                        # concern. Do not override the success result already returned.
                        logger.error(
                            "Post-deletion persistence flush failed for %s, "
                            "but deletion completed successfully: %s",
                            doc_id,
                            persistence_error,
                        )
                    elif original_exception is None:
                        # Deletion stages were in-flight but the try-block return was never
                        # reached; treat the persistence failure as the primary error.
                        return DeletionResult(
                            status="fail",
                            doc_id=doc_id,
                            message=f"Deletion completed but failed to persist changes: {persistence_error}",
                            status_code=500,
                            file_path=file_path,
                        )
                    # If there was an original exception, log the persistence error but
                    # don't override it — the original error result was already returned.
            else:
                logger.debug(
                    f"No deletion operations were started for document {doc_id}, skipping persistence"
                )

            logger.info("Deletion process completed for document: %s", doc_id)
