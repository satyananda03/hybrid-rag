
"""``PGDocStatusStorage`` — per-document pipeline state.

Backs ``get_docs_by_status`` / ``get_docs_by_statuses`` and the duplicate
lookups by file basename and content hash.
"""
from hybrid_rag.core.concurrency import _cooperative_yield
import time
import json
import datetime
from datetime import timezone
from dataclasses import dataclass, field
from typing import (
    Any,
    Union,
    final,
)
from hybrid_rag.core.base import (
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from hybrid_rag.core.utils import (
    performance_timing_log,
    validate_workspace,
)
import logging

from hybrid_rag.storage.shared import get_data_init_lock




import asyncpg  # type: ignore

from hybrid_rag.storage.postgres.connection import (
    ClientManager,
    PostgreSQLDB,
    _chunk_by_budget,
    _estimate_record_bytes,
    _resolve_pg_batch_limits,
)
from hybrid_rag.storage.postgres.sql import namespace_to_table_name

logger = logging.getLogger(__name__)


def _parse_doc_status_datetime(
    dt_str: Any,
    context: str = "",
) -> datetime.datetime | None:
    """Convert a datetime value to a naive UTC datetime for database storage.

    Accepts `datetime.datetime` objects, `datetime.date` objects, or ISO-format
    strings. Returns None on failure (which may trigger a NOT NULL constraint
    violation if the column does not allow nulls).
    The optional context string (e.g. "[workspace] doc <id> created_at") is
    included in the error log to help locate the offending record.
    """
    if dt_str is None:
        return None
    if isinstance(dt_str, datetime.datetime):
        if dt_str.tzinfo is None:
            dt_str = dt_str.replace(tzinfo=timezone.utc)
        return dt_str.astimezone(timezone.utc).replace(tzinfo=None)
    if isinstance(dt_str, datetime.date):
        return datetime.datetime(
            dt_str.year, dt_str.month, dt_str.day, tzinfo=timezone.utc
        ).replace(tzinfo=None)
    try:
        dt = datetime.datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except (ValueError, TypeError):
        logger.error(
            f"Unable to parse doc status datetime string"
            f"{f' ({context})' if context else ''}: {dt_str!r}"
        )
        return None


@final
@dataclass
class PGDocStatusStorage(DocStatusStorage):
    db: PostgreSQLDB = field(default=None)

    def __post_init__(self):
        validate_workspace(self.workspace)
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
            self._max_delete_records_per_batch,
        ) = _resolve_pg_batch_limits()

    def _format_datetime_with_timezone(self, dt):
        """Convert datetime to ISO format string with timezone info"""
        if dt is None:
            return None
        # If no timezone info, assume it's UTC time (as stored in database)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # If datetime already has timezone info, keep it as is
        return dt.isoformat()

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client(self.global_config["role"])

            if not self.workspace:
                self.workspace = "default"

            # NOTE: Table creation is handled by PostgreSQLDB.initdb() during initialization
            # No need to create table here as it's already created in the TABLES dict

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
        if not keys:
            return set()

        table_name = namespace_to_table_name(self.namespace)
        sql = f"SELECT id FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"
        params = {"workspace": self.workspace, "ids": list(keys)}
        try:
            res = await self.db.query(sql, list(params.values()), multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            new_keys = set([s for s in keys if s not in exist_keys])
            # print(f"keys: {keys}")
            # print(f"new_keys: {new_keys}")
            return new_keys
        except Exception as e:
            logger.error(
                f"[{self.workspace}] PostgreSQL database,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
            raise

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        sql = "select * from hybrid_rag_doc_status where workspace=$1 and id=$2"
        params = {"workspace": self.workspace, "id": id}
        result = await self.db.query(sql, list(params.values()), True)
        if result is None or result == []:
            return None
        else:
            # Parse chunks_list JSON string back to list
            chunks_list = result[0].get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = result[0].get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(result[0]["created_at"])
            updated_at = self._format_datetime_with_timezone(result[0]["updated_at"])

            return dict(
                content_length=result[0]["content_length"],
                content_summary=result[0]["content_summary"],
                status=result[0]["status"],
                chunks_count=result[0]["chunks_count"],
                created_at=created_at,
                updated_at=updated_at,
                file_path=result[0]["file_path"],
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=result[0].get("error_msg"),
                track_id=result[0].get("track_id"),
                content_hash=result[0].get("content_hash"),
            )

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get doc_chunks data by multiple IDs."""
        if not ids:
            return []

        sql = "SELECT * FROM hybrid_rag_doc_status WHERE workspace=$1 AND id = ANY($2)"
        params = {"workspace": self.workspace, "ids": ids}

        results = await self.db.query(sql, list(params.values()), True)

        if not results:
            return []

        processed_map: dict[str, dict[str, Any]] = {}
        for row in results:
            # Parse chunks_list JSON string back to list
            chunks_list = row.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(row["created_at"])
            updated_at = self._format_datetime_with_timezone(row["updated_at"])

            processed_map[str(row.get("id"))] = {
                "content_length": row["content_length"],
                "content_summary": row["content_summary"],
                "status": row["status"],
                "chunks_count": row["chunks_count"],
                "created_at": created_at,
                "updated_at": updated_at,
                "file_path": row["file_path"],
                "chunks_list": chunks_list,
                "metadata": metadata,
                "error_msg": row.get("error_msg"),
                "track_id": row.get("track_id"),
                "content_hash": row.get("content_hash"),
            }

        ordered_results: list[dict[str, Any] | None] = []
        for requested_id in ids:
            ordered_results.append(processed_map.get(str(requested_id)))

        return ordered_results

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        """PG-native override of basename-based document lookup.

        Replaces the base-class full-table scan with a database-level query on
        the canonical ``file_path`` column. The caller is responsible for
        passing an already-canonical basename; storage performs an exact match
        only.
        """
        if not basename:
            return None

        if basename == "unknown_source":
            return None

        sql = (
            "SELECT * FROM hybrid_rag_doc_status "
            "WHERE workspace=$1 AND file_path = $2 "
            "ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        params = [self.workspace, basename]

        result = await self.db.query(sql, params, True)
        if not result:
            return None
        row = result[0]

        chunks_list = row.get("chunks_list", [])
        if isinstance(chunks_list, str):
            try:
                chunks_list = json.loads(chunks_list)
            except json.JSONDecodeError:
                chunks_list = []

        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        created_at = self._format_datetime_with_timezone(row["created_at"])
        updated_at = self._format_datetime_with_timezone(row["updated_at"])

        doc = dict(
            content_length=row["content_length"],
            content_summary=row["content_summary"],
            status=row["status"],
            chunks_count=row["chunks_count"],
            created_at=created_at,
            updated_at=updated_at,
            file_path=row["file_path"],
            chunks_list=chunks_list,
            metadata=metadata,
            error_msg=row.get("error_msg"),
            track_id=row.get("track_id"),
            content_hash=row.get("content_hash"),
        )
        return str(row["id"]), doc

    async def get_doc_by_content_hash(
        self, content_hash: str, exclude_doc_id: str | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        """PG-native override of content-hash document lookup.

        An indexed query on ``workspace + content_hash``; ``exclude_doc_id``
        stays inside that query rather than scanning the workspace for the
        next match. Empty strings are treated as a miss to align with the
        partial-index predicate.
        """
        if not content_hash:
            return None

        params = [self.workspace, content_hash]
        exclude = ""
        if exclude_doc_id:
            params.append(exclude_doc_id)
            exclude = "AND id <> $3 "
        sql = (
            "SELECT * FROM hybrid_rag_doc_status "
            f"WHERE workspace=$1 AND content_hash=$2 {exclude}"
            "ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        result = await self.db.query(sql, params, True)
        if not result:
            return None
        row = result[0]

        chunks_list = row.get("chunks_list", [])
        if isinstance(chunks_list, str):
            try:
                chunks_list = json.loads(chunks_list)
            except json.JSONDecodeError:
                chunks_list = []

        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        created_at = self._format_datetime_with_timezone(row["created_at"])
        updated_at = self._format_datetime_with_timezone(row["updated_at"])

        doc = dict(
            content_length=row["content_length"],
            content_summary=row["content_summary"],
            status=row["status"],
            chunks_count=row["chunks_count"],
            created_at=created_at,
            updated_at=updated_at,
            file_path=row["file_path"],
            chunks_list=chunks_list,
            metadata=metadata,
            error_msg=row.get("error_msg"),
            track_id=row.get("track_id"),
            content_hash=row.get("content_hash"),
        )
        return str(row["id"]), doc

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        """Fetch documents matching any of the given statuses in a single query.

        Replaces multiple sequential/parallel get_docs_by_status() calls when the
        caller needs documents across several statuses (e.g. PROCESSING + FAILED + PENDING).
        Uses a single ANY($2) query instead of N separate round-trips.
        """
        if not statuses:
            return {}

        status_values = [s.value for s in statuses]
        sql = (
            "SELECT * FROM hybrid_rag_doc_status WHERE workspace=$1 AND status = ANY($2)"
        )
        result = await self.db.query(
            sql, [self.workspace, status_values], multirows=True
        )

        docs: dict[str, DocProcessingStatus] = {}
        for element in result or []:
            try:
                chunks_list = element.get("chunks_list", [])
                if isinstance(chunks_list, str):
                    try:
                        chunks_list = json.loads(chunks_list)
                    except json.JSONDecodeError:
                        chunks_list = []

                metadata = element.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}

                file_path = element.get("file_path") or "no-file-path"

                docs[element["id"]] = DocProcessingStatus(
                    content_summary=element["content_summary"],
                    content_length=element["content_length"],
                    status=element["status"],
                    created_at=self._format_datetime_with_timezone(
                        element["created_at"]
                    ),
                    updated_at=self._format_datetime_with_timezone(
                        element["updated_at"]
                    ),
                    chunks_count=element["chunks_count"],
                    file_path=file_path,
                    chunks_list=chunks_list,
                    metadata=metadata,
                    error_msg=element.get("error_msg"),
                    track_id=element.get("track_id"),
                    content_hash=element.get("content_hash"),
                )
            except (KeyError, TypeError) as e:
                doc_id_hint = element.get("id", "<unknown>") if element else "<unknown>"
                logger.error(
                    f"[{self.workspace}] Skipping document '{doc_id_hint}' — "
                    f"required field missing or wrong type while parsing DB row: {e!r}"
                )
                continue

        return docs

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}"
            )
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        # Chunk the id list so each statement's ANY($2) array stays bounded
        # (a non-positive cap disables chunking). All chunks run in ONE
        # transaction so a mid-delete failure rolls every chunk back, preserving
        # the original single-statement all-or-nothing behaviour; _run_with_retry
        # re-runs the whole closure on transient errors (DELETE is idempotent).
        chunk = (
            self._max_delete_records_per_batch
            if self._max_delete_records_per_batch > 0
            else len(ids)
        )
        if len(ids) > chunk:
            logger.info(
                f"[{self.workspace}] {self.namespace} delete: {len(ids)} ids "
                f"split into chunks (chunk={chunk})"
            )

        async def _batch_delete(connection: asyncpg.Connection) -> None:
            async with connection.transaction():
                for i in range(0, len(ids), chunk):
                    await connection.execute(
                        delete_sql, self.workspace, ids[i : i + chunk]
                    )

        try:
            await self.db._run_with_retry(_batch_delete)
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            # Raise, never swallow: document deletion deletes doc_status last and
            # relies on a failure here to keep the retry handle.
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}"
            )
            raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Update or insert document status

        Args:
            data: dictionary of document IDs and their status data
        """
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        timing_label = f"{self.workspace} PGDocStatusStorage.upsert"
        total_start = time.perf_counter()
        performance_timing_log(
            "[%s] start records=%s",
            timing_label,
            len(data),
        )

        # NOTE: content_hash uses COALESCE(NULLIF(...,''), existing) rather than
        # a straight EXCLUDED overwrite. This gives write-once-after-set
        # semantics: once a non-empty content_hash is recorded, subsequent
        # upserts that omit it (or pass '' / NULL) will NOT clear it. Required
        # because pipeline state transitions (e.g. processing -> processed)
        # reuse the existing DocProcessingStatus payload without re-supplying
        # the hash.
        #
        # This is a deliberate behavioral divergence from JsonDocStatusStorage,
        # which overwrites unconditionally. No caller today wants to clear a
        # content_hash, so the divergence is invisible — but if that ever
        # changes, this guard must be revisited.
        sql = """insert into hybrid_rag_doc_status(workspace,id,content_summary,content_length,chunks_count,status,file_path,chunks_list,track_id,metadata,error_msg,content_hash,created_at,updated_at)
                 values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                  on conflict(id,workspace) do update set
                  content_summary = EXCLUDED.content_summary,
                  content_length = EXCLUDED.content_length,
                  chunks_count = EXCLUDED.chunks_count,
                  status = EXCLUDED.status,
                  file_path = EXCLUDED.file_path,
                  chunks_list = EXCLUDED.chunks_list,
                  track_id = EXCLUDED.track_id,
                  metadata = EXCLUDED.metadata,
                  error_msg = EXCLUDED.error_msg,
                  content_hash = COALESCE(
                      NULLIF(EXCLUDED.content_hash, ''),
                      hybrid_rag_doc_status.content_hash
                  ),
                  created_at = EXCLUDED.created_at,
                  updated_at = EXCLUDED.updated_at"""

        # Tuple order must match SQL: (workspace, id, content_summary, content_length,
        #   chunks_count, status, file_path, chunks_list, track_id, metadata,
        #   error_msg, content_hash, created_at, updated_at)
        batch: list[tuple] = []
        skipped: list[str] = []
        batch_build_start = time.perf_counter()
        for i, (k, v) in enumerate(data.items(), start=1):
            try:
                batch.append(
                    (
                        self.workspace,
                        k,
                        v["content_summary"],
                        v["content_length"],
                        v.get("chunks_count", -1),
                        v["status"],
                        v["file_path"],
                        json.dumps(v.get("chunks_list", [])),
                        v.get("track_id"),
                        json.dumps(v.get("metadata", {})),
                        v.get("error_msg"),
                        v.get("content_hash"),
                        _parse_doc_status_datetime(
                            v.get("created_at"),
                            f"[{self.workspace}] doc {k} created_at",
                        ),
                        _parse_doc_status_datetime(
                            v.get("updated_at"),
                            f"[{self.workspace}] doc {k} updated_at",
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.error(
                    f"[{self.workspace}] Skipping document '{k}' in batch upsert — "
                    f"invalid or missing field: {e!r}"
                )
                skipped.append(k)
            await _cooperative_yield(i)

        if skipped:
            logger.warning(
                f"[{self.workspace}] {len(skipped)} document(s) skipped in batch upsert: {skipped}"
            )
        performance_timing_log(
            "[%s] batch validation/assembly completed in %.4fs valid_count=%s skipped_count=%s",
            timing_label,
            time.perf_counter() - batch_build_start,
            len(batch),
            len(skipped),
        )

        # Split into payload-byte / record-count bounded sub-batches, each its
        # own transaction (mirrors KV upsert / mongo_impl). ON CONFLICT makes
        # every chunk idempotent, so a mid-flush failure is safely retryable.
        batches = _chunk_by_budget(
            batch,
            _estimate_record_bytes,
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
        )
        num_batches = len(batches)
        log_prefix = f"[{self.workspace}] {self.namespace} upsert:"
        if num_batches > 1:
            logger.info(
                f"{log_prefix} split into {num_batches} batches "
                f"for {len(batch)} records"
            )
        for batch_index, (sub_batch, estimated_bytes) in enumerate(batches, start=1):
            if (
                len(sub_batch) == 1
                and self._max_upsert_payload_bytes > 0
                and estimated_bytes > self._max_upsert_payload_bytes
            ):
                logger.warning(
                    f"{log_prefix} single record estimated {estimated_bytes} "
                    f"bytes exceeds {self._max_upsert_payload_bytes}"
                )

            async def _batch_upsert(
                connection: asyncpg.Connection,
                _sql: str = sql,
                _data: list[tuple] = sub_batch,
                _batch_index: int = batch_index,
                _num_batches: int = num_batches,
            ) -> None:
                execute_start = time.perf_counter()
                async with connection.transaction():
                    await connection.executemany(_sql, _data)
                performance_timing_log(
                    "[%s] sub-batch %s/%s transaction + executemany completed in %.4fs batch_size=%s",
                    timing_label,
                    _batch_index,
                    _num_batches,
                    time.perf_counter() - execute_start,
                    len(_data),
                )

            await self.db._run_with_retry(_batch_upsert, timing_label=timing_label)
        logger.debug(
            f"[{self.workspace}] Batch upserted {len(batch)} records to {self.namespace}"
        )
        performance_timing_log(
            "[%s] total complete in %.4fs valid_count=%s skipped_count=%s",
            timing_label,
            time.perf_counter() - total_start,
            len(batch),
            len(skipped),
        )

