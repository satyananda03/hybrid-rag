
"""PostgreSQL connection pool, schema bootstrap, and batching helpers.

:class:`PostgreSQLDB` owns the asyncpg pool, creates the tables and indexes on
first use, and runs every query with retry. :class:`ClientManager` hands one
shared pool to all four storage classes so a process opens the pool once.

The module-level helpers size write batches: asyncpg has a parameter ceiling and
a statement-size ceiling, so upserts are chunked by estimated payload bytes as
well as record count.
"""

import asyncio
import time
import hashlib
import json
from typing import (
    Any,
    Awaitable,
    Callable,
    TypeVar,
)
import numpy as np


from hybrid_rag.config import settings

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)
from hybrid_rag.core.utils import (
    performance_timing_log,
)
import logging





import asyncpg  # type: ignore
from asyncpg import Pool  # type: ignore
from pgvector.asyncpg import register_vector  # type: ignore

from hybrid_rag.storage.postgres.sql import TABLES

logger = logging.getLogger(__name__)



T = TypeVar("T")

# PostgreSQL identifier length limit (in bytes)
PG_MAX_IDENTIFIER_LENGTH = 63

# Flush-time batching limits shared by the PostgreSQL non-graph write paths
# (PGKVStorage, PGVectorStorage, PGDocStatusStorage). Mirrors the MONGO_* /
# OPENSEARCH_* contract: the payload-byte budget is the primary limiter and the
# record-count caps are a secondary guard. Unlike Mongo/OpenSearch there is no
# single server-side bulk message to stay under -- asyncpg's executemany
# pipelines each row over a reused prepared statement -- so for PostgreSQL the
# byte budget mainly bounds client-side peak memory and transaction duration.
# The default record cap keeps PGKVStorage's historical 200 behaviour; delete
# ids are short strings so a larger count cap is safe. The numbers themselves
# live on ``Settings`` (POSTGRES_UPSERT_* / POSTGRES_DELETE_*) — a second copy
# here would be one more place to drift.

def _estimate_record_bytes(record: tuple[Any, ...]) -> int:
    """Estimate the serialized byte size of one asyncpg parameter tuple.

    A splitting *heuristic* for ``_chunk_by_budget``, not the exact wire size.
    numpy vectors dominate vector rows and large text dominates KV rows, so the
    estimate sums those accurately and treats scalars as a small constant:

      * ``np.ndarray`` -> ``.nbytes`` (the binary pgvector payload)
      * ``str`` -> UTF-8 byte length
      * ``bytes`` / ``bytearray`` -> ``len``
      * ``dict`` / ``list`` -> compact-JSON UTF-8 length (already-serialized
        JSON columns are passed as ``str`` and handled above; this covers any
        not-yet-serialized field)
      * ``None`` -> 0
      * everything else (int / float / datetime / ...) -> a small constant
    """
    total = 0
    for field_value in record:
        if isinstance(field_value, np.ndarray):
            total += field_value.nbytes
        elif isinstance(field_value, str):
            total += len(field_value.encode("utf-8"))
        elif isinstance(field_value, (bytes, bytearray)):
            total += len(field_value)
        elif field_value is None:
            total += 0
        elif isinstance(field_value, (dict, list)):
            total += len(
                json.dumps(
                    field_value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
        else:
            total += 16
    return total


def _chunk_by_budget(
    items: list[T],
    size_of: Callable[[T], int],
    max_payload_bytes: int,
    max_records_per_batch: int,
) -> list[tuple[list[T], int]]:
    """Split items into batches by estimated payload size (primary) and count.

    The byte budget is the primary limiter: items accumulate until adding the
    next one would exceed ``max_payload_bytes``, then a new batch starts.
    ``size_of(item)`` returns an item's estimated serialized byte size. A single
    item larger than the byte budget is emitted as its own batch rather than
    raising; the server stays the final arbiter. A non-positive limit disables
    that dimension. Returns ``(batch, summed_estimated_bytes)`` tuples (the
    estimate is used for logging). Semantically identical to mongo_impl's
    ``_chunk_by_budget``.
    """
    if not items:
        return []

    payload_limit = max_payload_bytes if max_payload_bytes > 0 else float("inf")
    records_limit = max_records_per_batch if max_records_per_batch > 0 else float("inf")

    batches: list[tuple[list[T], int]] = []
    current: list[T] = []
    # JSON array overhead ("[]")
    current_bytes = 2

    for item in items:
        item_bytes = size_of(item)
        # If current batch not empty, a comma is needed before next element.
        separator_overhead = 1 if current else 0
        next_bytes = current_bytes + separator_overhead + item_bytes

        if current and (len(current) >= records_limit or next_bytes > payload_limit):
            batches.append((current, current_bytes))
            current = []
            current_bytes = 2
            next_bytes = current_bytes + item_bytes

        current.append(item)
        current_bytes = next_bytes

    if current:
        batches.append((current, current_bytes))

    return batches


def _resolve_pg_batch_limits() -> tuple[int, int, int]:
    """Resolve flush-time batching limits from :class:`Settings`.

    Shared by every PostgreSQL non-graph write path so the byte/record caps that
    bound a single ``executemany`` / ``DELETE ... ANY($2)`` are consistent across
    all of them. A non-positive value disables that splitting dimension and logs
    a warning. Returns ``(upsert_payload_bytes, upsert_records, delete_records)``.
    """
    upsert_payload_bytes = settings.postgres_upsert_max_payload_bytes
    upsert_records = settings.postgres_upsert_max_records_per_batch
    delete_records = settings.postgres_delete_max_records_per_batch
    if upsert_payload_bytes <= 0:
        logger.warning(
            f"POSTGRES_UPSERT_MAX_PAYLOAD_BYTES={upsert_payload_bytes} is non-positive, disable payload-size splitting"
        )
    if upsert_records <= 0:
        logger.warning(
            f"POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH={upsert_records} is non-positive, disable upsert record-count splitting"
        )
    if delete_records <= 0:
        logger.warning(
            f"POSTGRES_DELETE_MAX_RECORDS_PER_BATCH={delete_records} is non-positive, disable delete record-count splitting"
        )
    return upsert_payload_bytes, upsert_records, delete_records


# Sufiks indeks vektor yang dijatuhkan saat berganti tipe. ivfflat/vchordrq
# tidak bisa lagi DIBUAT (lihat create_sql di bawah), tapi tetap didaftarkan di
# sini supaya indeks warisan dari konfigurasi lama ikut terbersihkan.
_VECTOR_INDEX_SUFFIXES = [
    "hnsw_cosine",
    "hnsw_halfvec_cosine",
    "ivfflat_cosine",
    "vchordrq_cosine",
]

def _safe_index_name(table_name: str, index_suffix: str) -> str:
    """
    Generate a PostgreSQL-safe index name that won't be truncated.

    PostgreSQL silently truncates identifiers to 63 bytes. This function
    ensures index names stay within that limit by hashing long table names.

    Args:
        table_name: The table name (may be long with model suffix)
        index_suffix: The index type suffix (e.g., 'hnsw_cosine', 'id', 'workspace_id')

    Returns:
        A deterministic index name that fits within 63 bytes
    """
    # Construct the full index name
    full_name = f"idx_{table_name.lower()}_{index_suffix}"

    # If it fits within the limit, use it as-is
    if len(full_name.encode("utf-8")) <= PG_MAX_IDENTIFIER_LENGTH:
        return full_name

    # Otherwise, hash the table name to create a shorter unique identifier
    # Keep 'idx_' prefix and suffix readable, hash the middle
    hash_input = table_name.lower().encode("utf-8")
    table_hash = hashlib.md5(hash_input).hexdigest()[:12]  # 12 hex chars

    # Format: idx_{hash}_{suffix} - guaranteed to fit
    # Maximum: idx_ (4) + hash (12) + _ (1) + suffix (variable) = 17 + suffix
    shortened_name = f"idx_{table_hash}_{index_suffix}"

    return shortened_name


def _timing_details_suffix(**details: Any) -> str:
    parts = [f"{key}={value}" for key, value in details.items()]
    return f" {' '.join(parts)}" if parts else ""


class PostgreSQLDB:
    def __init__(self, config: dict[str, Any], **kwargs: Any):
        self.host = config["host"]
        self.port = config["port"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.max = int(config["max_connections"])
        self.increment = 1
        self.pool: Pool | None = None

        # TLS. Server saat ini ssl=off; isi POSTGRES_SSL_MODE=require begitu
        # server mengaktifkannya. Sertifikat klien (verify-ca/verify-full) tidak
        # didukung — tambahkan lagi kalau memang dipakai.
        self.ssl_mode = config.get("ssl_mode")

        # Vector configuration. pgvector selalu aktif — satu-satunya vector
        # storage yang dipakai pipeline ini adalah PGVectorStorage.
        self.vector_index_type = config.get("vector_index_type")
        self.hnsw_m = config.get("hnsw_m")
        self.hnsw_ef = config.get("hnsw_ef")

        # Schema configuration — defaults to "rag". All hybrid_rag tables are
        # created inside this schema. The search_path GUC is set both at pool
        # creation (server_settings) and restored after every pool reset so
        # that unqualified table names always resolve to this schema.
        self.schema_name: str = config.get("schema_name") or "rag"

        if self.user is None or self.password is None or self.database is None:
            raise ValueError("Missing database user, password, or database")

        # Guard concurrent pool resets
        self._pool_reconnect_lock = asyncio.Lock()

        self._transient_exceptions = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.TooManyConnectionsError,
            asyncpg.exceptions.CannotConnectNowError,
            asyncpg.exceptions.PostgresConnectionError,
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.ConnectionFailureError,
        )

        # Connection retry configuration
        self.connection_retry_attempts = config["connection_retry_attempts"]
        self.connection_retry_backoff = config["connection_retry_backoff"]
        self.connection_retry_backoff_max = max(
            self.connection_retry_backoff,
            config["connection_retry_backoff_max"],
        )
        self.pool_close_timeout = config["pool_close_timeout"]
        logger.info(
            "PostgreSQL, Retry config: attempts=%s, backoff=%.1fs, backoff_max=%.1fs, pool_close_timeout=%.1fs",
            self.connection_retry_attempts,
            self.connection_retry_backoff,
            self.connection_retry_backoff_max,
            self.pool_close_timeout,
        )

    def _connect_kwargs(self) -> dict[str, Any]:
        """asyncpg connection arguments shared by the pool and :meth:`bootstrap`."""
        connection_params: dict[str, Any] = {
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "host": self.host,
            "port": self.port,
        }

        # TLS: hanya mode sederhana. asyncpg merundingkan sendiri sisanya.
        if self.ssl_mode:
            mode = self.ssl_mode.lower()
            if mode in ("require", "prefer", "allow"):
                connection_params["ssl"] = True
            elif mode == "disable":
                connection_params["ssl"] = False
            else:
                logger.warning(
                    f"PostgreSQL, mode SSL '{self.ssl_mode}' tidak didukung "
                    "(pakai require/prefer/allow/disable) — TLS tidak dipakai"
                )
            logger.info(f"PostgreSQL, SSL mode set to: {self.ssl_mode}")

        return connection_params

    async def bootstrap(self) -> None:
        """CREATE SCHEMA and the vector extension, before the pool exists.

        Runs on a connection of its own: the pool registers the pgvector codec
        on every connection, which fails on a database without the extension.
        Called by :mod:`hybrid_rag.db_init` only; the runtime runs no DDL.
        """
        conn = await asyncpg.connect(**self._connect_kwargs())
        try:
            await self._ensure_schema(conn)
            await self.configure_vector_extension(conn)
        finally:
            await conn.close()

    async def initdb(self):
        connection_params = {**self._connect_kwargs(), "min_size": 1, "max_size": self.max}
        # search_path ke schema target supaya nama tabel tanpa prefix ketemu.
        connection_params["server_settings"] = {"search_path": f"{self.schema_name},public"}
        logger.info(f"PostgreSQL, search_path set to: {self.schema_name}")

        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async def _init_connection(connection: asyncpg.Connection) -> None:
            """Daftarkan codec pgvector pada tiap koneksi fisik baru.

            Dipanggil sekali per pembuatan koneksi (bukan saat pool dipakai
            ulang). register_vector adalah registrasi codec level-Python yang
            selamat dari RESET ALL milik asyncpg.
            """
            await register_vector(connection)

        async def _create_pool_once() -> None:
            # The vector extension must already exist (make db-init): the
            # register_vector callback fails on a database without it.
            pool = await asyncpg.create_pool(
                **connection_params,
                init=_init_connection,  # register pgvector codec on new connections
                # Tanpa reset=: cleanup bawaan asyncpg (RESET ALL) sudah
                # mengembalikan search_path ke nilai server_settings di atas.
            )  # type: ignore
            self.pool = pool

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.connection_retry_attempts),
                retry=retry_if_exception_type(self._transient_exceptions),
                wait=wait_strategy,
                before_sleep=self._before_sleep,
                reraise=True,
            ):
                with attempt:
                    await _create_pool_once()

            ssl_status = "with SSL" if connection_params.get("ssl") else "without SSL"
            logger.info(
                f"PostgreSQL, Connected to database at {self.host}:{self.port}/{self.database} {ssl_status}"
            )
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to connect database at {self.host}:{self.port}/{self.database}, Got:{e}"
            )
            raise

    async def _ensure_pool(self) -> None:
        """Ensure the connection pool is initialised."""
        if self.pool is None:
            async with self._pool_reconnect_lock:
                if self.pool is None:
                    await self.initdb()

    async def _reset_pool(self) -> None:
        async with self._pool_reconnect_lock:
            if self.pool is not None:
                try:
                    await asyncio.wait_for(
                        self.pool.close(), timeout=self.pool_close_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "PostgreSQL, Timed out closing connection pool after %.2fs",
                        self.pool_close_timeout,
                    )
                except Exception as close_error:  # pragma: no cover - defensive logging
                    logger.warning(
                        f"PostgreSQL, Failed to close existing connection pool cleanly: {close_error!r}"
                    )
            self.pool = None

    async def _before_sleep(self, retry_state: RetryCallState) -> None:
        """Hook invoked by tenacity before sleeping between retries."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "PostgreSQL transient connection issue on attempt %s/%s: %r",
            retry_state.attempt_number,
            self.connection_retry_attempts,
            exc,
        )
        await self._reset_pool()

    async def _run_with_retry(
        self,
        operation: Callable[[asyncpg.Connection], Awaitable[T]],
        *,
        timing_label: str | None = None,
    ) -> T:
        """
        Execute a database operation with automatic retry for transient failures.

        Args:
            operation: Async callable that receives an active connection.

        Returns:
            The result returned by the operation.

        Raises:
            Exception: Propagates the last error if all retry attempts fail or a non-transient error occurs.
        """
        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.connection_retry_attempts),
            retry=retry_if_exception_type(self._transient_exceptions),
            wait=wait_strategy,
            before_sleep=self._before_sleep,
            reraise=True,
        ):
            with attempt:
                await self._ensure_pool()
                assert self.pool is not None
                if timing_label:
                    pool_snapshot_before = self._get_pool_snapshot()
                    performance_timing_log(
                        "[%s] pool.acquire waiting %s",
                        timing_label,
                        pool_snapshot_before,
                    )
                acquire_start = time.perf_counter()
                async with self.pool.acquire() as connection:  # type: ignore[arg-type]
                    acquire_elapsed = time.perf_counter() - acquire_start
                    if timing_label:
                        pool_snapshot_after = self._get_pool_snapshot()
                        performance_timing_log(
                            "[%s] pool.acquire completed in %.4fs %s",
                            timing_label,
                            acquire_elapsed,
                            pool_snapshot_after,
                        )
                    return await operation(connection)

    def _get_pool_snapshot(self) -> str:
        """Best-effort snapshot of asyncpg pool state for diagnostics.

        Uses asyncpg private attributes defensively; if a field is unavailable in the
        installed asyncpg version, return '?' for that metric instead of failing.
        """
        pool = self.pool
        if pool is None:
            return "pool_state=uninitialized"

        holders = getattr(pool, "_holders", None)
        queue = getattr(pool, "_queue", None)
        max_size = getattr(pool, "_maxsize", None)
        min_size = getattr(pool, "_minsize", None)

        total_holders = len(holders) if holders is not None else "?"
        idle_count: int | str = "?"
        acquired_count: int | str = "?"

        if holders is not None:
            idle_count = 0
            acquired_count = 0
            for holder in holders:
                # asyncpg holder uses _in_use Future/Event-like marker; treat present value as acquired
                in_use_marker = getattr(holder, "_in_use", None)
                if in_use_marker:
                    acquired_count += 1
                else:
                    idle_count += 1

        waiting_count: int | str = "?"
        if queue is not None:
            getters = getattr(queue, "_getters", None)
            if getters is not None:
                waiting_count = len(getters)

        return (
            f"pool_state[min={min_size}, max={max_size}, holders={total_holders}, "
            f"acquired={acquired_count}, idle={idle_count}, waiting={waiting_count}]"
        )

    async def _ensure_schema(self, connection: asyncpg.Connection) -> None:
        """Create the target schema if it does not exist.

        Called once during bootstrap (before pool creation) so all subsequent
        DDL statements issued by check_tables() land in the correct schema.
        """
        try:
            await connection.execute(
                f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}"
            )
            logger.info(f"PostgreSQL, Schema '{self.schema_name}' ensured")
        except Exception as e:
            logger.error(f"PostgreSQL, Failed to create schema '{self.schema_name}': {e}")
            raise

    async def configure_vector_extension(self, connection: asyncpg.Connection) -> None:
        """Create VECTOR extension if it doesn't exist for vector similarity operations.

        When vector_index_type is HNSW_HALFVEC, validates that pgvector >= 0.7.0
        (required for halfvec support) and raises RuntimeError if older.
        """
        try:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")  # type: ignore
            logger.info("PostgreSQL, VECTOR extension enabled")
        except Exception as e:
            logger.warning(f"Could not create VECTOR extension: {e}")
            # Don't raise - let the system continue without vector extension
            return

        if getattr(self, "vector_index_type", None) == "HNSW_HALFVEC":
            row = await connection.fetchrow(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            if not row or not row["extversion"]:
                raise RuntimeError(
                    "POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC requires the pgvector "
                    "extension. Ensure it is installed and CREATE EXTENSION vector succeeded."
                )
            raw_version = row["extversion"]
            try:
                parts = [int(p) for p in str(raw_version).split(".")[:3]]
                while len(parts) < 3:
                    parts.append(0)
                version_tuple = (parts[0], parts[1], parts[2])
            except (ValueError, IndexError):
                raise RuntimeError(
                    f"Could not parse pgvector version {raw_version!r}. "
                    "HNSW_HALFVEC requires pgvector >= 0.7.0."
                ) from None
            if version_tuple < (0, 7, 0):
                raise RuntimeError(
                    f"POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC requires pgvector >= 0.7.0, "
                    f"but installed version is {raw_version}. Upgrade the pgvector extension "
                    "or use a different index type (e.g. HNSW with embeddings <= 2000 dimensions)."
                )

    async def check_tables(self):
        # Vector tables that should be skipped - they are created by PGVectorStorage.setup_table()
        # with proper embedding model and dimension suffix for data isolation
        vector_tables_to_skip = {
            "hybrid_rag_vdb_chunks",
            "hybrid_rag_vdb_entities",
            "hybrid_rag_vdb_relations",
        }

        # First create all tables (except vector tables)
        for k, v in TABLES.items():
            # Skip vector tables - they are created by PGVectorStorage.setup_table()
            if k in vector_tables_to_skip:
                continue

            if await self.check_table_exists(k):
                continue
            try:
                logger.info(f"PostgreSQL, Creating table {k}")
                await self.execute(v["ddl"])
            except Exception as e:
                logger.error(f"PostgreSQL, Failed to create table {k}: {e}")
                raise

        # Batch check all indexes at once (optimization: single query instead of N queries)
        try:
            # Exclude vector tables from index creation since they are created by PGVectorStorage.setup_table()
            table_names = [k for k in TABLES.keys() if k not in vector_tables_to_skip]
            table_names_lower = [t.lower() for t in table_names]

            # Get all existing indexes for our tables in one query
            check_all_indexes_sql = """
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = ANY($1)
            """
            existing_indexes_result = await self.query(
                check_all_indexes_sql, [table_names_lower], multirows=True
            )

            # Build a set of existing index names for fast lookup
            existing_indexes = set()
            if existing_indexes_result:
                existing_indexes = {row["indexname"] for row in existing_indexes_result}

            # Create missing indexes
            for k in table_names:
                # Create index for id column if missing
                index_name = f"idx_{k.lower()}_id"
                if index_name not in existing_indexes:
                    try:
                        create_index_sql = f"CREATE INDEX {index_name} ON {k}(id)"
                        logger.info(
                            f"PostgreSQL, Creating index {index_name} on table {k}"
                        )
                        await self.execute(create_index_sql)
                    except Exception as e:
                        logger.error(
                            f"PostgreSQL, Failed to create index {index_name}, Got: {e}"
                        )

                # Create composite index for (workspace, id) if missing
                composite_index_name = f"idx_{k.lower()}_workspace_id"
                if composite_index_name not in existing_indexes:
                    try:
                        create_composite_index_sql = (
                            f"CREATE INDEX {composite_index_name} ON {k}(workspace, id)"
                        )
                        logger.info(
                            f"PostgreSQL, Creating composite index {composite_index_name} on table {k}"
                        )
                        await self.execute(create_composite_index_sql)
                    except Exception as e:
                        logger.error(
                            f"PostgreSQL, Failed to create composite index {composite_index_name}, Got: {e}"
                        )

                # Deletion resolves a document's chunks through full_doc_id, so
                # that lookup must not be a sequential scan on the chunk table.
                if k == "hybrid_rag_doc_chunks":
                    doc_index_name = f"idx_{k.lower()}_workspace_full_doc_id"
                    if doc_index_name not in existing_indexes:
                        try:
                            logger.info(
                                f"PostgreSQL, Creating index {doc_index_name} on table {k}"
                            )
                            await self.execute(
                                f"CREATE INDEX {doc_index_name} ON {k}(workspace, full_doc_id)"
                            )
                        except Exception as e:
                            logger.error(
                                f"PostgreSQL, Failed to create index {doc_index_name}, Got: {e}"
                            )
        except Exception as e:
            logger.error(f"PostgreSQL, Failed to batch check/create indexes: {e}")

        # Vector tables and their indexes are created by PGVectorStorage.setup_table().
        try:
            await self._create_doc_status_indexes()
        except Exception as e:
            logger.error(f"PostgreSQL, Failed to create doc status indexes: {e}")

    async def _create_doc_status_indexes(self):
        """Secondary indexes on hybrid_rag_doc_status: status scans, file-path and content-hash dedup."""
        indexes = [
            {
                "name": "idx_hybrid_rag_doc_status_workspace_status_updated_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hybrid_rag_doc_status_workspace_status_updated_at ON hybrid_rag_doc_status (workspace, status, updated_at DESC)",
                "description": "Composite index for workspace + status + updated_at pagination",
            },
            {
                "name": "idx_hybrid_rag_doc_status_workspace_status_created_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hybrid_rag_doc_status_workspace_status_created_at ON hybrid_rag_doc_status (workspace, status, created_at DESC)",
                "description": "Composite index for workspace + status + created_at pagination",
            },
            {
                "name": "idx_hybrid_rag_doc_status_workspace_updated_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hybrid_rag_doc_status_workspace_updated_at ON hybrid_rag_doc_status (workspace, updated_at DESC)",
                "description": "Index for workspace + updated_at pagination (all statuses)",
            },
            {
                "name": "idx_hybrid_rag_doc_status_workspace_created_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hybrid_rag_doc_status_workspace_created_at ON hybrid_rag_doc_status (workspace, created_at DESC)",
                "description": "Index for workspace + created_at pagination (all statuses)",
            },
            {
                "name": "idx_hybrid_rag_doc_status_workspace_id",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hybrid_rag_doc_status_workspace_id ON hybrid_rag_doc_status (workspace, id)",
                "description": "Index for workspace + id sorting",
            },
            {
                "name": "idx_hybrid_rag_doc_status_workspace_file_path",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hybrid_rag_doc_status_workspace_file_path ON hybrid_rag_doc_status (workspace, file_path)",
                "description": "Index for workspace + file_path sorting",
            },
            {
                "name": "idx_hybrid_rag_doc_status_workspace_content_hash",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hybrid_rag_doc_status_workspace_content_hash ON hybrid_rag_doc_status (workspace, content_hash) WHERE content_hash IS NOT NULL AND content_hash <> ''",
                "description": "Partial index for workspace + content_hash duplicate lookup",
            },
        ]

        # Fetch all existing index names in one query instead of N separate checks.
        index_names = [idx["name"] for idx in indexes]
        check_sql = """
            SELECT indexname FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = 'hybrid_rag_doc_status'
            AND indexname = ANY($1)
        """
        try:
            rows = await self.query(check_sql, [index_names], multirows=True)
            existing_names = {row["indexname"] for row in (rows or [])}
        except asyncpg.PostgresError as e:
            logger.warning(
                f"[{self.schema_name}] Failed to query existing doc status indexes "
                f"({type(e).__name__}), will attempt to create all: {e}"
            )
            existing_names = set()

        for index in indexes:
            if index["name"] in existing_names:
                logger.debug(f"Index already exists: {index['name']}")
                continue
            try:
                logger.info(f"Creating doc status index: {index['description']}")
                await self.execute(index["sql"])
                logger.info(f"Successfully created index: {index['name']}")
            except asyncpg.PostgresError as e:
                logger.warning(
                    f"Failed to create index {index['name']} ({type(e).__name__}): {e}"
                )

    async def _create_vector_index(self, table_name: str, embedding_dim: int):
        """
        Create vector index for a specific table.

        Args:
            table_name: Name of the table to create index on
            embedding_dim: Embedding dimension for the vector column
        """
        if not self.vector_index_type:
            return

        create_sql = {
            "HNSW": f"""
                CREATE INDEX {{vector_index_name}}
                ON {{table_name}} USING hnsw (content_vector vector_cosine_ops)
                WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef})
            """,
            "HNSW_HALFVEC": f"""
                CREATE INDEX {{vector_index_name}}
                ON {{table_name}} USING hnsw (content_vector halfvec_cosine_ops)
                WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef})
            """,
        }

        if self.vector_index_type not in create_sql:
            logger.warning(
                f"Unsupported vector index type: {self.vector_index_type}. "
                "Supported types: HNSW, HNSW_HALFVEC"
            )
            return

        k = table_name
        # Use _safe_index_name to avoid PostgreSQL's 63-byte identifier truncation
        index_suffix = f"{self.vector_index_type.lower()}_cosine"
        vector_index_name = _safe_index_name(k, index_suffix)
        check_vector_index_sql = f"""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = '{vector_index_name}' AND tablename = '{k.lower()}'
        """
        if self.vector_index_type == "HNSW_HALFVEC":
            column_type = "HALFVEC"
        else:
            column_type = "VECTOR"
        try:
            vector_index_exists = await self.query(check_vector_index_sql)
            if not vector_index_exists:
                for suffix in _VECTOR_INDEX_SUFFIXES:
                    if suffix == index_suffix:
                        continue
                    old_name = _safe_index_name(k, suffix)
                    await self.execute(f"DROP INDEX IF EXISTS {old_name}")
                alter_sql = f"ALTER TABLE {k} ALTER COLUMN content_vector TYPE {column_type}({embedding_dim})"
                await self.execute(alter_sql)
                logger.debug(f"Ensured vector dimension for {k}")
                logger.info(
                    f"Creating {self.vector_index_type} index {vector_index_name} on table {k}"
                )
                await self.execute(
                    create_sql[self.vector_index_type].format(
                        vector_index_name=vector_index_name, table_name=k
                    )
                )
                logger.info(
                    f"Successfully created vector index {vector_index_name} on table {k}"
                )
            else:
                logger.info(
                    f"{self.vector_index_type} vector index {vector_index_name} already exists on table {k}"
                )
        except Exception as e:
            logger.error(f"Failed to create vector index on table {k}, Got: {e}")

    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        multirows: bool = False,
        timing_label: str | None = None,
    ) -> dict[str, Any] | None | list[dict[str, Any]]:
        async def _operation(connection: asyncpg.Connection) -> Any:
            prepared_params = tuple(params) if params else ()
            fetch_start = time.perf_counter()
            if prepared_params:
                rows = await connection.fetch(sql, *prepared_params)
            else:
                rows = await connection.fetch(sql)
            fetch_elapsed = time.perf_counter() - fetch_start

            if timing_label:
                performance_timing_log(
                    "[%s] connection.fetch completed in %.4fs row_count=%s",
                    timing_label,
                    fetch_elapsed,
                    len(rows),
                )

            conversion_start = time.perf_counter()

            if multirows:
                if rows:
                    columns = [col for col in rows[0].keys()]
                    converted_rows = [dict(zip(columns, row)) for row in rows]
                else:
                    converted_rows = []

                if timing_label:
                    conversion_elapsed = time.perf_counter() - conversion_start
                    performance_timing_log(
                        "[%s] result conversion completed in %.4fs multirows=%s",
                        timing_label,
                        conversion_elapsed,
                        True,
                    )
                return converted_rows

            if rows:
                columns = rows[0].keys()
                converted_row = dict(zip(columns, rows[0]))
            else:
                converted_row = None

            if timing_label:
                conversion_elapsed = time.perf_counter() - conversion_start
                performance_timing_log(
                    "[%s] result conversion completed in %.4fs multirows=%s",
                    timing_label,
                    conversion_elapsed,
                    False,
                )
            if converted_row is not None:
                return converted_row
            return None

        try:
            return await self._run_with_retry(
                _operation,
                timing_label=timing_label,
            )
        except Exception as e:
            logger.error(f"PostgreSQL database, error:{e}")
            raise

    async def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in PostgreSQL database

        Args:
            table_name: Name of the table to check

        Returns:
            bool: True if table exists, False otherwise
        """
        query = "SELECT to_regclass($1) IS NOT NULL AS exists"
        result = await self.query(query, [table_name.lower()])
        return result.get("exists", False) if result else False

    async def execute(
        self,
        sql: str,
        data: dict[str, Any] | None = None,
        upsert: bool = False,
        ignore_if_exists: bool = False,
        timing_label: str | None = None,
    ):
        async def _operation(connection: asyncpg.Connection) -> Any:
            prepared_values = tuple(data.values()) if data else ()
            execute_start = time.perf_counter()
            try:
                if not data:
                    result = await connection.execute(sql)
                else:
                    result = await connection.execute(sql, *prepared_values)
            except (
                asyncpg.exceptions.UniqueViolationError,
                asyncpg.exceptions.DuplicateTableError,
                asyncpg.exceptions.DuplicateObjectError,
                asyncpg.exceptions.InvalidSchemaNameError,
            ) as e:
                if ignore_if_exists:
                    logger.debug("PostgreSQL, ignoring duplicate during execute: %r", e)
                    result = None
                elif upsert:
                    logger.info(
                        "PostgreSQL, duplicate detected but treated as upsert success: %r",
                        e,
                    )
                    result = None
                else:
                    raise
            except Exception:
                if timing_label:
                    performance_timing_log(
                        "[%s] connection.execute failed after %.4fs",
                        timing_label,
                        time.perf_counter() - execute_start,
                    )
                raise
            if timing_label:
                performance_timing_log(
                    "[%s] connection.execute completed in %.4fs result=%s",
                    timing_label,
                    time.perf_counter() - execute_start,
                    result,
                )
            return result

        try:
            await self._run_with_retry(
                _operation,
                timing_label=timing_label,
            )
        except Exception as e:
            logger.error(f"PostgreSQL database,\nsql:{sql},\ndata:{data},\nerror:{e}")
            raise


def _pool_config() -> dict[str, Any]:
    """Connection + tuning parameters for one role's asyncpg pool.

    Batas atas pada retry/timeout sengaja dipertahankan: nilai .env yang
    kelewat besar akan menahan startup alih-alih gagal cepat.
    """
    return {
        "host": settings.postgres_host,
        "port": settings.postgres_port,
        "user": settings.postgres_user,
        "password": settings.postgres_password,
        "database": settings.postgres_database,
        "max_connections": settings.pg_async_pool_size,
        "ssl_mode": settings.postgres_ssl_mode,
        "vector_index_type": settings.postgres_vector_index_type,
        "hnsw_m": settings.postgres_hnsw_m,
        "hnsw_ef": settings.postgres_hnsw_ef,
        "connection_retry_attempts": min(100, settings.postgres_connection_retries),
        "connection_retry_backoff": min(300.0, settings.postgres_connection_retry_backoff),
        "connection_retry_backoff_max": min(600.0, settings.postgres_connection_retry_backoff_max),
        "pool_close_timeout": min(30.0, settings.postgres_pool_close_timeout),
        "schema_name": settings.postgres_schema,
    }


class ClientManager:
    """One refcounted PostgreSQL pool per role, shared by that role's storages.

    Ingestion and retrieval each get their own pool (``POSTGRES_MAX_CONNECTIONS``
    budgets both in ``Settings``), so a burst of ingestion writes never queues
    retrieval reads behind it.
    """

    _instances: dict[str, dict[str, Any]] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls, role: str) -> PostgreSQLDB:
        """Return ``role``'s shared client, opening the pool on first use."""
        async with cls._lock:
            entry = cls._instances.get(role)
            if entry is None:
                config = _pool_config()
                logger.info(f"Database : {config['database']} ({role} pool)")
                db = PostgreSQLDB(config)
                await db.initdb()
                entry = cls._instances[role] = {"db": db, "ref_count": 0}
            entry["ref_count"] += 1
            return entry["db"]

    @classmethod
    async def release_client(cls, db: PostgreSQLDB | None) -> None:
        """Drop one reference; the last one closes that role's pool."""
        if db is None:
            return
        async with cls._lock:
            for role, entry in list(cls._instances.items()):
                if entry["db"] is db:
                    entry["ref_count"] -= 1
                    if entry["ref_count"] == 0:
                        if db.pool is not None:
                            await db.pool.close()
                        logger.info(f"Closed PostgreSQL {role} connection pool")
                        del cls._instances[role]
                    return
            if db.pool is not None:
                await db.pool.close()
