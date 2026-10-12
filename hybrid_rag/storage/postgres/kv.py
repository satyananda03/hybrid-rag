"""``PGKVStorage`` — key/value tables: full docs, chunks, entity and relation
chunk maps, and the LLM response cache.
"""

from hybrid_rag.core.concurrency import _cooperative_yield
import json
import datetime
from datetime import timezone
from dataclasses import dataclass, field
from typing import (
    Any,
    final,
)


from hybrid_rag.core.base import BaseKVStorage
from hybrid_rag.core.namespace import NameSpace, is_namespace
from hybrid_rag.core.utils import validate_workspace
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
from hybrid_rag.storage.postgres.sql import SQL_TEMPLATES, namespace_to_table_name

logger = logging.getLogger(__name__)


def _parse_json_list(value: Any) -> list:
    """asyncpg hands JSONB back as a string; every reader here wants the list.

    A missing / malformed value normalizes to ``[]`` rather than propagating a
    string, because callers iterate the result — and iterating a JSON string
    yields characters, which fails far away from the cause.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


@final
@dataclass
class PGKVStorage(BaseKVStorage):
    db: PostgreSQLDB = field(default=None)

    def __post_init__(self):
        validate_workspace(self.workspace)
        self._max_batch_size = 200  # DB batch size, independent of embedding batch size
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
            self._max_delete_records_per_batch,
        ) = _resolve_pg_batch_limits()

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client(self.global_config["role"])

            if not self.workspace:
                self.workspace = "default"

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    ################ QUERY METHODS ################
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get data by id."""
        sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.workspace, "id": id}
        response = await self.db.query(sql, list(params.values()))

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list
            llm_cache_list = response.get("llm_cache_list", [])
            if isinstance(llm_cache_list, str):
                try:
                    llm_cache_list = json.loads(llm_cache_list)
                except json.JSONDecodeError:
                    llm_cache_list = []
            response["llm_cache_list"] = llm_cache_list

            # Parse heading JSON string back to dict; normalize None/missing to {}
            heading = response.get("heading")
            if isinstance(heading, str):
                try:
                    heading = json.loads(heading)
                except json.JSONDecodeError:
                    heading = {}
            if not isinstance(heading, dict):
                heading = {}
            response["heading"] = heading

            # Parse sidecar JSON string back to dict; normalize None/missing to {}
            sidecar = response.get("sidecar")
            if isinstance(sidecar, str):
                try:
                    sidecar = json.loads(sidecar)
                except json.JSONDecodeError:
                    sidecar = {}
            if not isinstance(sidecar, dict):
                sidecar = {}
            response["sidecar"] = sidecar

            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            # Parse chunk_options JSON string back to dict; normalize None/missing to {}
            chunk_options = response.get("chunk_options")
            if isinstance(chunk_options, str):
                try:
                    chunk_options = json.loads(chunk_options)
                except json.JSONDecodeError:
                    chunk_options = {}
            if not isinstance(chunk_options, dict):
                chunk_options = {}
            response["chunk_options"] = chunk_options
            response["multimodal_items"] = _parse_json_list(
                response.get("multimodal_items")
            )

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if response and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            # Parse queryparam JSON string back to dict
            queryparam = response.get("queryparam")
            if isinstance(queryparam, str):
                try:
                    queryparam = json.loads(queryparam)
                except json.JSONDecodeError:
                    queryparam = None
            # Map field names for compatibility (mode field removed)
            response = {
                **response,
                "return": response.get("return_value", ""),
                "cache_type": response.get("cache_type"),
                "original_prompt": response.get("original_prompt", ""),
                "chunk_id": response.get("chunk_id"),
                "queryparam": queryparam,
                "create_time": create_time,
                "update_time": create_time if update_time == 0 else update_time,
            }

        # Special handling for FULL_ENTITIES namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            # Parse entity_names JSON string back to list
            entity_names = response.get("entity_names", [])
            if isinstance(entity_names, str):
                try:
                    entity_names = json.loads(entity_names)
                except json.JSONDecodeError:
                    entity_names = []
            response["entity_names"] = entity_names
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            # Parse relation_pairs JSON string back to list
            relation_pairs = response.get("relation_pairs", [])
            if isinstance(relation_pairs, str):
                try:
                    relation_pairs = json.loads(relation_pairs)
                except json.JSONDecodeError:
                    relation_pairs = []
            response["relation_pairs"] = relation_pairs
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for ENTITY_CHUNKS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            # Parse chunk_ids JSON string back to list
            chunk_ids = response.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except json.JSONDecodeError:
                    chunk_ids = []
            response["chunk_ids"] = chunk_ids
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for RELATION_CHUNKS namespace
        if response and is_namespace(
            self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS
        ):
            # Parse chunk_ids JSON string back to list
            chunk_ids = response.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except json.JSONDecodeError:
                    chunk_ids = []
            response["chunk_ids"] = chunk_ids
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        return response if response else None

    # Query by id
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get data by ids"""
        if not ids:
            return []

        sql = SQL_TEMPLATES["get_by_ids_" + self.namespace]
        params = {"workspace": self.workspace, "ids": ids}
        results = await self.db.query(sql, list(params.values()), multirows=True)

        def _order_results(
            rows: list[dict[str, Any]] | None,
        ) -> list[dict[str, Any] | None]:
            """Preserve the caller requested ordering for bulk id lookups."""
            if not rows:
                return [None for _ in ids]

            id_map: dict[str, dict[str, Any]] = {}
            for row in rows:
                if row is None:
                    continue
                row_id = row.get("id")
                if row_id is not None:
                    id_map[str(row_id)] = row

            ordered: list[dict[str, Any] | None] = []
            for requested_id in ids:
                ordered.append(id_map.get(str(requested_id)))
            return ordered

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list / heading / sidecar JSON strings for each result
            for result in results:
                llm_cache_list = result.get("llm_cache_list", [])
                if isinstance(llm_cache_list, str):
                    try:
                        llm_cache_list = json.loads(llm_cache_list)
                    except json.JSONDecodeError:
                        llm_cache_list = []
                result["llm_cache_list"] = llm_cache_list

                heading = result.get("heading")
                if isinstance(heading, str):
                    try:
                        heading = json.loads(heading)
                    except json.JSONDecodeError:
                        heading = {}
                if not isinstance(heading, dict):
                    heading = {}
                result["heading"] = heading

                sidecar = result.get("sidecar")
                if isinstance(sidecar, str):
                    try:
                        sidecar = json.loads(sidecar)
                    except json.JSONDecodeError:
                        sidecar = {}
                if not isinstance(sidecar, dict):
                    sidecar = {}
                result["sidecar"] = sidecar

                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            for result in results:
                chunk_options = result.get("chunk_options")
                if isinstance(chunk_options, str):
                    try:
                        chunk_options = json.loads(chunk_options)
                    except json.JSONDecodeError:
                        chunk_options = {}
                if not isinstance(chunk_options, dict):
                    chunk_options = {}
                result["chunk_options"] = chunk_options
                result["multimodal_items"] = _parse_json_list(
                    result.get("multimodal_items")
                )

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if results and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            processed_results = []
            for row in results:
                create_time = row.get("create_time", 0)
                update_time = row.get("update_time", 0)
                # Parse queryparam JSON string back to dict
                queryparam = row.get("queryparam")
                if isinstance(queryparam, str):
                    try:
                        queryparam = json.loads(queryparam)
                    except json.JSONDecodeError:
                        queryparam = None
                # Map field names for compatibility (mode field removed)
                processed_row = {
                    **row,
                    "return": row.get("return_value", ""),
                    "cache_type": row.get("cache_type"),
                    "original_prompt": row.get("original_prompt", ""),
                    "chunk_id": row.get("chunk_id"),
                    "queryparam": queryparam,
                    "create_time": create_time,
                    "update_time": create_time if update_time == 0 else update_time,
                }
                processed_results.append(processed_row)
            return _order_results(processed_results)

        # Special handling for FULL_ENTITIES namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            for result in results:
                # Parse entity_names JSON string back to list
                entity_names = result.get("entity_names", [])
                if isinstance(entity_names, str):
                    try:
                        entity_names = json.loads(entity_names)
                    except json.JSONDecodeError:
                        entity_names = []
                result["entity_names"] = entity_names
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            for result in results:
                # Parse relation_pairs JSON string back to list
                relation_pairs = result.get("relation_pairs", [])
                if isinstance(relation_pairs, str):
                    try:
                        relation_pairs = json.loads(relation_pairs)
                    except json.JSONDecodeError:
                        relation_pairs = []
                result["relation_pairs"] = relation_pairs
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for ENTITY_CHUNKS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            for result in results:
                # Parse chunk_ids JSON string back to list
                chunk_ids = result.get("chunk_ids", [])
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                result["chunk_ids"] = chunk_ids
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for RELATION_CHUNKS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            for result in results:
                # Parse chunk_ids JSON string back to list
                chunk_ids = result.get("chunk_ids", [])
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                result["chunk_ids"] = chunk_ids
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        return _order_results(results)

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
            return new_keys
        except Exception as e:
            logger.error(
                f"[{self.workspace}] PostgreSQL database,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
            raise

    async def get_chunk_ids_by_doc_id(self, doc_id: str) -> list[str]:
        """Every chunk id this table holds for ``doc_id``, straight from the DB.

        The authoritative answer to "what belongs to this document" — the
        ``full_doc_id`` column is written by both the text pipeline and the
        multimodal processors, so it covers chunks that ``doc_status.chunks_list``
        may have lost track of. Deletion unions this with ``chunks_list`` rather
        than trusting either alone.
        """
        if not is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            raise ValueError(
                f"get_chunk_ids_by_doc_id is only valid for the text-chunk "
                f"namespace, not {self.namespace!r}"
            )
        table_name = namespace_to_table_name(self.namespace)
        sql = f"SELECT id FROM {table_name} WHERE workspace=$1 AND full_doc_id=$2"
        rows = await self.db.query(sql, [self.workspace, doc_id], multirows=True)
        return [str(row["id"]) for row in (rows or []) if row.get("id")]

    ################ INSERT METHODS ################
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        batch_values: list[tuple] = []
        upsert_sql = ""

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            upsert_sql = SQL_TEMPLATES["upsert_text_chunk"]
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for i, (k, v) in enumerate(data.items(), start=1):
                # Tuple order must match SQL: (workspace, id, tokens, chunk_order_index,
                #   full_doc_id, content, file_path, llm_cache_list, heading, sidecar,
                #   create_time, update_time)
                batch_values.append(
                    (
                        self.workspace,
                        k,
                        v["tokens"],
                        v["chunk_order_index"],
                        v["full_doc_id"],
                        v["content"],
                        v["file_path"],
                        json.dumps(v.get("llm_cache_list", [])),
                        json.dumps(v.get("heading") or {}),
                        json.dumps(v.get("sidecar") or {}),
                        current_time,
                        current_time,
                    )
                )
                await _cooperative_yield(i)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            upsert_sql = SQL_TEMPLATES["upsert_doc_full"]
            for i, (k, v) in enumerate(data.items(), start=1):
                # Tuple order must match SQL: (id, content, doc_name, workspace,
                #   sidecar_location, parse_format, content_hash, process_options,
                #   chunk_options, parse_engine, multimodal_items)
                #
                # All pipeline-derived fields pass through untouched so the
                # SQL-level COALESCE guard in upsert_doc_full can distinguish
                # "caller did not supply" (None/'') from "caller supplied a
                # real value". The 'raw' default for parse_format is provided
                # by the column DDL on initial insert; do NOT default it here
                # or the COALESCE guard never triggers on subsequent partial
                # writes.
                batch_values.append(
                    (
                        k,
                        v["content"],
                        v.get("file_path", ""),
                        self.workspace,
                        v.get("sidecar_location"),
                        v.get("parse_format"),
                        v.get("content_hash"),
                        v.get("process_options"),
                        json.dumps(v.get("chunk_options") or {}),
                        v.get("parse_engine"),
                        json.dumps(v.get("multimodal_items") or []),
                    )
                )
                await _cooperative_yield(i)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
            for i, (k, v) in enumerate(data.items(), start=1):
                # Tuple order must match SQL: (workspace, id, original_prompt, return_value,
                #   chunk_id, cache_type, queryparam)
                batch_values.append(
                    (
                        self.workspace,
                        k,
                        v["original_prompt"],
                        v["return"],
                        v.get("chunk_id"),
                        v.get("cache_type", "extract"),
                        json.dumps(v.get("queryparam"))
                        if v.get("queryparam")
                        else None,
                    )
                )
                await _cooperative_yield(i)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            upsert_sql = SQL_TEMPLATES["upsert_full_entities"]
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for i, (k, v) in enumerate(data.items(), start=1):
                # Tuple order must match SQL: (workspace, id, entity_names, count,
                #   create_time, update_time)
                batch_values.append(
                    (
                        self.workspace,
                        k,
                        json.dumps(v["entity_names"]),
                        v["count"],
                        current_time,
                        current_time,
                    )
                )
                await _cooperative_yield(i)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            upsert_sql = SQL_TEMPLATES["upsert_full_relations"]
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for i, (k, v) in enumerate(data.items(), start=1):
                # Tuple order must match SQL: (workspace, id, relation_pairs, count,
                #   create_time, update_time)
                batch_values.append(
                    (
                        self.workspace,
                        k,
                        json.dumps(v["relation_pairs"]),
                        v["count"],
                        current_time,
                        current_time,
                    )
                )
                await _cooperative_yield(i)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            upsert_sql = SQL_TEMPLATES["upsert_entity_chunks"]
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for i, (k, v) in enumerate(data.items(), start=1):
                # Tuple order must match SQL: (workspace, id, chunk_ids, count,
                #   create_time, update_time)
                batch_values.append(
                    (
                        self.workspace,
                        k,
                        json.dumps(v["chunk_ids"]),
                        v["count"],
                        current_time,
                        current_time,
                    )
                )
                await _cooperative_yield(i)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            upsert_sql = SQL_TEMPLATES["upsert_relation_chunks"]
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for i, (k, v) in enumerate(data.items(), start=1):
                # Tuple order must match SQL: (workspace, id, chunk_ids, count,
                #   create_time, update_time)
                batch_values.append(
                    (
                        self.workspace,
                        k,
                        json.dumps(v["chunk_ids"]),
                        v["count"],
                        current_time,
                        current_time,
                    )
                )
                await _cooperative_yield(i)
        else:
            logger.error(f"Unknown namespace: {self.namespace}")
            raise ValueError(f"Unknown namespace: {self.namespace}")

        # upsert_sql is always set here; unknown namespace raises ValueError above
        if batch_values:
            # Split into payload-byte (primary) / record-count (secondary)
            # bounded sub-batches to bound peak memory and transaction duration
            # (mirrors mongo_impl's _run_batched_bulk_write). asyncpg pipelines
            # each row in executemany, so the byte budget caps client-side
            # assembly rather than a single server message.
            batches = _chunk_by_budget(
                batch_values,
                _estimate_record_bytes,
                self._max_upsert_payload_bytes,
                self._max_upsert_records_per_batch,
            )
            num_batches = len(batches)
            log_prefix = f"[{self.workspace}] {self.namespace} upsert:"
            if num_batches > 1:
                logger.info(
                    f"{log_prefix} split into {num_batches} batches "
                    f"for {len(batch_values)} records"
                )
            for sub_batch, estimated_bytes in batches:
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
                    _sql: str = upsert_sql,
                    _data: list[tuple] = sub_batch,
                ) -> None:
                    await connection.executemany(_sql, _data)

                await self.db._run_with_retry(_batch_upsert)

            logger.debug(
                f"[{self.workspace}] Batch upserted {len(batch_values)} records to {self.namespace} "
                f"in {num_batches} sub-batches"
            )

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
            # Raise, never swallow: a caller told "deleted" while the rows stay
            # would leak a deleted document's chunks into retrieval.
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}"
            )
            raise

