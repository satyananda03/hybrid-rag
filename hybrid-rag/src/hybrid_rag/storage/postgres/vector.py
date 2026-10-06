"""``PGVectorStorage`` — the vector tables, and where two of the three indexing
methods actually happen.

A single INSERT writes both the embedding (computed here in batches through
``embedding_func``) and ``content_cleaned``, the stopword-stripped text the
BM25 index is built on. :meth:`PGVectorStorage.query` reads them back: for
chunks it runs a vector search and a BM25 search and fuses them with RRF
(``SQL_TEMPLATES["chunks_hybrid_bm25"]``).
"""

from hybrid_rag.core.concurrency import _cooperative_yield
import asyncio
import time
import json
import datetime
from datetime import timezone
from dataclasses import dataclass, field
from typing import (
    Any,
    final,
)
import numpy as np


from hybrid_rag.storage.postgres.bm25_text import preprocess_indonesian_fast
from hybrid_rag.core.base import BaseVectorStorage
from hybrid_rag.core.namespace import NameSpace, is_namespace
from hybrid_rag.core.utils import (
    performance_timing_log,
    validate_workspace,
)
from hybrid_rag.config import settings
import logging

from hybrid_rag.storage.shared import get_data_init_lock, get_namespace_lock




import asyncpg  # type: ignore

from hybrid_rag.storage.postgres.connection import (
    ClientManager,
    PostgreSQLDB,
    _chunk_by_budget,
    _estimate_record_bytes,
    _resolve_pg_batch_limits,
)
from hybrid_rag.storage.postgres.connection import PG_MAX_IDENTIFIER_LENGTH, _safe_index_name
from hybrid_rag.storage.postgres.sql import SQL_TEMPLATES, TABLES, namespace_to_table_name

logger = logging.getLogger(__name__)


def _as_float32_vector(value: Any) -> np.ndarray | None:
    """A stored ``content_vector`` as a float32 array, whatever the codec returned."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    elif not isinstance(value, (list, tuple, np.ndarray)):
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        else:
            return None
    return np.asarray(value, dtype=np.float32)


@dataclass
class _PendingPGVectorDoc:
    """Buffered PG vector upsert awaiting embedding and batched flush.

    ``vector`` is stored as a numpy ndarray (typically float32 from the
    embedding function) once embedded; pgvector's asyncpg codec accepts
    ndarray directly so no per-flush conversion is needed.
    """

    item: dict[str, Any]
    created_at: datetime.datetime
    vector: np.ndarray | None = None


@final
@dataclass
class PGVectorStorage(BaseVectorStorage):
    db: PostgreSQLDB | None = field(default=None)

    def __post_init__(self):
        validate_workspace(self.workspace)
        self._validate_embedding_func()
        self._max_batch_size = self.global_config["embedding_batch_num"]
        # DB-write batching limits (distinct from the embedding batch size above).
        (
            self._max_upsert_payload_bytes,
            self._max_upsert_records_per_batch,
            self._max_delete_records_per_batch,
        ) = _resolve_pg_batch_limits()
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = config.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Generate model suffix for table isolation
        self.model_suffix = self._generate_collection_suffix()

        # Get base table name
        base_table = namespace_to_table_name(self.namespace)
        if not base_table:
            raise ValueError(f"Unknown namespace: {self.namespace}")

        # New table name (with suffix)
        # Ensure model_suffix is not empty before appending
        if self.model_suffix:
            self.table_name = f"{base_table}_{self.model_suffix}"
            logger.info(f"PostgreSQL table: {self.table_name}")
        else:
            # Fallback: use base table name if model_suffix is unavailable
            self.table_name = base_table
            logger.warning(
                f"PostgreSQL table: {self.table_name} missing suffix. Pls add model_name to embedding_func for proper workspace data isolation."
            )

        # DDL template key in ``TABLES``.
        self.base_table = base_table

        # Validate table name length (PostgreSQL identifier limit is 63 characters)
        if len(self.table_name) > PG_MAX_IDENTIFIER_LENGTH:
            raise ValueError(
                f"PostgreSQL table name exceeds {PG_MAX_IDENTIFIER_LENGTH} character limit: '{self.table_name}' "
                f"(length: {len(self.table_name)}). "
                f"Consider using a shorter embedding model name or workspace name."
            )

        # Pending buffers: upsert() and delete() queue work here until
        # _flush_pending_vector_ops() runs from index_done_callback() /
        # finalize(). Mirrors OpenSearchVectorDBStorage / NanoVectorDBStorage.
        self._pending_vector_docs: dict[str, _PendingPGVectorDoc] = {}
        self._pending_vector_deletes: set[str] = set()
        # Namespace-keyed lock; created in initialize() after workspace is final.
        self._flush_lock = None
        # Serialises embed_pending() runs so two flushes never embed the same
        # pending doc twice; created on first use (binds to the running loop).
        self._embed_lock: asyncio.Lock | None = None
        # Embeddings started by prime() before their doc is upserted, keyed by
        # the exact content; embed_pending() takes them instead of re-embedding.
        self._primed: dict[str, asyncio.Future] = {}

    @staticmethod
    async def _pg_create_table(
        db: PostgreSQLDB, table_name: str, base_table: str, embedding_dim: int
    ) -> None:
        """Create a new vector table by replacing the table name in DDL template,
        and create indexes on id and (workspace, id) columns.

        Args:
            db: PostgreSQLDB instance
            table_name: Name of the new table to create
            base_table: Base table name for DDL template lookup
            embedding_dim: Embedding dimension for vector column
        """
        if base_table not in TABLES:
            raise ValueError(f"No DDL template found for table: {base_table}")

        ddl_template = TABLES[base_table]["ddl"]

        # Determine vector column type based on configuration
        # HALFVEC is used when HNSW_HALFVEC is selected
        vector_type = "VECTOR"
        if getattr(db, "vector_index_type", None) == "HNSW_HALFVEC":
            vector_type = "HALFVEC"

        # Replace embedding dimension placeholder if exists
        ddl = ddl_template.replace(
            "VECTOR(dimension)", f"{vector_type}({embedding_dim})"
        )

        # Replace table name
        ddl = ddl.replace(base_table, table_name)

        # Make creation idempotent to handle restarts and race conditions
        ddl = ddl.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
        await db.execute(ddl)

        # Create indexes similar to check_tables() but with safe index names
        # Create index for id column
        id_index_name = _safe_index_name(table_name, "id")
        try:
            create_id_index_sql = (
                f"CREATE INDEX IF NOT EXISTS {id_index_name} ON {table_name}(id)"
            )
            logger.info(
                f"PostgreSQL, Creating index {id_index_name} on table {table_name}"
            )
            await db.execute(create_id_index_sql)
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to create index {id_index_name}, Got: {e}"
            )

        # Create composite index for (workspace, id)
        workspace_id_index_name = _safe_index_name(table_name, "workspace_id")
        try:
            create_composite_index_sql = f"CREATE INDEX IF NOT EXISTS {workspace_id_index_name} ON {table_name}(workspace, id)"
            logger.info(
                f"PostgreSQL, Creating composite index {workspace_id_index_name} on table {table_name}"
            )
            await db.execute(create_composite_index_sql)
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to create composite index {workspace_id_index_name}, Got: {e}"
            )

        # Create pg_search extension + BM25 index for hybrid search (chunks only)
        if base_table == "hybrid_rag_vdb_chunks":
            try:
                await db.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
                logger.info("PostgreSQL, pg_search extension enabled")
            except Exception as e:
                logger.warning(
                    f"PostgreSQL, pg_search extension not available — "
                    f"BM25 hybrid search will be disabled: {e}"
                )

            # Deletion resolves a document's chunks through this column, so it
            # must not be a sequential scan on a table this size.
            doc_index_name = _safe_index_name(table_name, "workspace_full_doc_id")
            try:
                await db.execute(
                    f"CREATE INDEX IF NOT EXISTS {doc_index_name} "
                    f"ON {table_name}(workspace, full_doc_id)"
                )
                logger.info(
                    f"PostgreSQL, Creating index {doc_index_name} on table {table_name}"
                )
            except Exception as e:
                logger.error(
                    f"PostgreSQL, Failed to create index {doc_index_name}, Got: {e}"
                )

            bm25_index_name = _safe_index_name(table_name, "bm25")
            try:
                await db.execute(f"""
                    CREATE INDEX IF NOT EXISTS {bm25_index_name}
                    ON {table_name}
                    USING bm25 (id, content_cleaned)
                    WITH (key_field='id')
                """)
                logger.info(
                    f"PostgreSQL, BM25 index '{bm25_index_name}' created on {table_name}"
                )
            except Exception as e:
                logger.error(
                    f"PostgreSQL, Failed to create BM25 index {bm25_index_name}: {e}"
                )

    @staticmethod
    async def setup_table(
        db: PostgreSQLDB, table_name: str, base_table: str, embedding_dim: int
    ) -> None:
        """Create the model-suffixed vector table and its indexes; idempotent."""
        await PGVectorStorage._pg_create_table(db, table_name, base_table, embedding_dim)
        await db._create_vector_index(table_name, embedding_dim)

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client(self.global_config["role"])

            if not self.workspace:
                self.workspace = "default"

            # DDL tabel vector hanya saat bootstrap; lihat `database/rag_schema.py`.
            if settings.rag_schema_bootstrap:
                await PGVectorStorage.setup_table(
                    self.db,
                    self.table_name,
                    base_table=self.base_table,
                    embedding_dim=self.embedding_func.embedding_dim,
                )

        if self._flush_lock is None:
            # Keyed by role too: retrieval only reads (get_vectors_by_ids checks
            # the pending buffer under this lock) and must never queue behind
            # an ingestion flush of the same workspace.
            self._flush_lock = get_namespace_lock(
                f"{self.namespace}:{self.global_config['role']}", workspace=self.workspace
            )

    async def finalize(self):
        """Flush pending vector ops then release the shared PG client.

        Captures regular ``Exception`` from the flush so it can be re-raised
        as a ``RuntimeError`` naming the unflushed buffer counts after the
        client is released. ``BaseException`` (``CancelledError``,
        ``KeyboardInterrupt``, ``SystemExit``) is intentionally NOT caught
        so it can propagate through ``finally`` — the buffer-count reframing
        below is skipped in that case (the propagating exception already
        signals shutdown; conflating it with "left N pending" would be
        misleading).

        Idempotency:
            Re-entry after a successful or failed first call is a no-op for
            the flush (client is already released), but still raises if
            buffers remain non-empty so the operator sees the data-loss
            signal again.
        """
        if self.db is None:
            pending_docs = len(self._pending_vector_docs)
            pending_deletes = len(self._pending_vector_deletes)
            if pending_docs or pending_deletes:
                raise RuntimeError(
                    f"[{self.workspace}] PGVectorStorage.finalize() re-entry: "
                    f"client already released; {pending_docs} pending upserts "
                    f"and {pending_deletes} pending deletes cannot be flushed"
                )
            return

        flush_error: Exception | None = None
        try:
            try:
                await self._flush_pending_vector_ops()
            except Exception as e:
                flush_error = e
        finally:
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

        pending_docs = len(self._pending_vector_docs)
        pending_deletes = len(self._pending_vector_deletes)
        if flush_error is not None:
            raise RuntimeError(
                f"[{self.workspace}] PGVectorStorage.finalize() flush raised; "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes were left buffered (client released, data lost)"
            ) from flush_error
        if pending_docs or pending_deletes:
            raise RuntimeError(
                f"[{self.workspace}] PGVectorStorage.finalize() left "
                f"{pending_docs} pending upserts and {pending_deletes} "
                f"pending deletes buffered after final flush attempt"
            )

    def _upsert_chunks(
        self, item: dict[str, Any], current_time: datetime.datetime
    ) -> tuple[str, tuple[Any, ...]]:
        """Prepare upsert data for chunks.

        Returns:
            Tuple of (SQL template, values tuple for executemany)
        """
        try:
            upsert_sql = SQL_TEMPLATES["upsert_chunk"].format(
                table_name=self.table_name
            )
            # Preprocess content for BM25 index (Sastrawi stopword removal)
            # Applied to contextualized chunk content ("{context}. {child_content}")
            content_cleaned = preprocess_indonesian_fast(item["content"])

            # Return tuple in the exact order of SQL parameters ($1, $2, ...)
            values: tuple[Any, ...] = (
                self.workspace,  # $1
                item["__id__"],  # $2
                item["tokens"],  # $3
                item["chunk_order_index"],  # $4
                item["full_doc_id"],  # $5
                item["content"],  # $6
                item["__vector__"],  # $7 - numpy array, handled by pgvector codec
                item["file_path"],  # $8
                content_cleaned,  # $9 - preprocessed for BM25
                current_time,  # $10
                current_time,  # $11
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error to prepare upsert,\nerror: {e}\nitem: {item}"
            )
            raise

        return upsert_sql, values

    def _upsert_entities(
        self, item: dict[str, Any], current_time: datetime.datetime
    ) -> tuple[str, tuple[Any, ...]]:
        """Prepare upsert data for entities.

        Returns:
            Tuple of (SQL template, values tuple for executemany)
        """
        upsert_sql = SQL_TEMPLATES["upsert_entity"].format(table_name=self.table_name)
        source_id = item["source_id"]
        if isinstance(source_id, str) and "<SEP>" in source_id:
            chunk_ids = source_id.split("<SEP>")
        else:
            chunk_ids = [source_id]

        # Return tuple in the exact order of SQL parameters ($1, $2, ...)
        values: tuple[Any, ...] = (
            self.workspace,  # $1
            item["__id__"],  # $2
            item["entity_name"],  # $3
            item["content"],  # $4
            item["__vector__"],  # $5 - numpy array, handled by pgvector codec
            chunk_ids,  # $6
            item.get("file_path", None),  # $7
            current_time,  # $8
            current_time,  # $9
        )
        return upsert_sql, values

    def _upsert_relationships(
        self, item: dict[str, Any], current_time: datetime.datetime
    ) -> tuple[str, tuple[Any, ...]]:
        """Prepare upsert data for relationships.

        Returns:
            Tuple of (SQL template, values tuple for executemany)
        """
        upsert_sql = SQL_TEMPLATES["upsert_relationship"].format(
            table_name=self.table_name
        )
        source_id = item["source_id"]
        if isinstance(source_id, str) and "<SEP>" in source_id:
            chunk_ids = source_id.split("<SEP>")
        else:
            chunk_ids = [source_id]

        # Return tuple in the exact order of SQL parameters ($1, $2, ...)
        values: tuple[Any, ...] = (
            self.workspace,  # $1
            item["__id__"],  # $2
            item["src_id"],  # $3
            item["tgt_id"],  # $4
            item["content"],  # $5
            item["__vector__"],  # $6 - numpy array, handled by pgvector codec
            chunk_ids,  # $7
            item.get("file_path", None),  # $8
            current_time,  # $9
            current_time,  # $10
        )
        return upsert_sql, values

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vector docs for embedding and batched flush.

        Correctness premise:
            The ingestion pipeline is the only writer and guarantees a single
            writer per workspace. The pending buffer is process-local:
            committed rows are visible to every reader, buffered writes only
            after index_done_callback().

        Memory expectation:
            Pending docs (raw ``content`` strings, plus cached float32
            vectors once embedded) accumulate in process memory until the
            next ``index_done_callback()`` / ``finalize()``. This matches
            the OpenSearch/Nano/Faiss contract. Callers performing very
            large ingests should flush periodically (every N upserts) to
            cap working-set size.
        """
        if not data:
            return

        logger.debug(
            f"[{self.workspace}] Buffering {len(data)} vectors for {self.namespace}"
        )

        # Build pending docs outside the lock; UTC naive datetime mirrors
        # the previous direct-write code path (the _upsert_* helpers feed
        # this straight into asyncpg as a timestamp).
        current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
        pending_docs: list[tuple[str, _PendingPGVectorDoc]] = []
        for i, (k, v) in enumerate(data.items(), start=1):
            pending_docs.append(
                (
                    k,
                    _PendingPGVectorDoc(
                        item={"__id__": k, **v},
                        created_at=current_time,
                    ),
                )
            )
            await _cooperative_yield(i)

        async with self._flush_lock:
            for doc_id, pending_doc in pending_docs:
                # Invariant: a later upsert wins over an earlier delete; the
                # unconditional dict assignment also discards any cached
                # stale vector from a prior upsert of the same id.
                self._pending_vector_deletes.discard(doc_id)
                self._pending_vector_docs[doc_id] = pending_doc

    async def embed_pending(self) -> None:
        """Embed buffered docs that have no vector yet, without holding ``_flush_lock``.

        Safe to call at any time (the pipeline calls it early for chunks, while
        extraction is still running) and called again by every flush. Vectors are
        cached on the pending record only if that record is still the one in the
        buffer — a concurrent upsert/delete/drop during the embedding window wins,
        and the flush embeds the replacement itself. Same pattern as
        ``get_vectors_by_ids``.

        A doc whose ``content`` is byte-identical to the row already stored under
        its id reuses that row's vector instead of calling the model: same text and
        same model give the same vector, and the table is per model+dimension.
        """
        if self._flush_lock is None or self.db is None:
            return
        if self._embed_lock is None:
            self._embed_lock = asyncio.Lock()
        async with self._embed_lock:
            async with self._flush_lock:
                todo = [
                    (doc_id, pending)
                    for doc_id, pending in self._pending_vector_docs.items()
                    if pending.vector is None
                ]
            if not todo:
                return

            vectors = await self._stored_vectors_for_unchanged(todo)
            primed = [
                (doc_id, self._primed.pop(pending.item["content"]))
                for doc_id, pending in todo
                if doc_id not in vectors and pending.item["content"] in self._primed
            ]
            if primed:
                results = await asyncio.gather(
                    *(future for _, future in primed), return_exceptions=True
                )
                vectors.update(
                    (doc_id, result[0])
                    for (doc_id, _), result in zip(primed, results)
                    if not isinstance(result, BaseException)
                )
            to_embed = [(i, p) for i, p in todo if i not in vectors]
            if to_embed:
                contents = [p.item["content"] for _, p in to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                embedding_start = time.perf_counter()
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error embedding pending vector ops "
                        f"(upserts={len(to_embed)}): {e}"
                    )
                    raise
                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: "
                        f"expected {len(to_embed)}, got {len(embeddings)}"
                    )
                vectors.update(
                    (doc_id, vector) for (doc_id, _), vector in zip(to_embed, embeddings)
                )
                logger.info(
                    f"[{self.workspace}] {self.namespace}: embedded {len(to_embed)} "
                    f"in {time.perf_counter() - embedding_start:.1f}s, reused "
                    f"{len(todo) - len(to_embed)} unchanged or primed"
                )

            async with self._flush_lock:
                for i, (doc_id, pending) in enumerate(todo, start=1):
                    if self._pending_vector_docs.get(doc_id) is pending:
                        pending.vector = vectors[doc_id]
                    await _cooperative_yield(i)

    def prime(self, content: str) -> None:
        """Start embedding ``content`` now, ahead of its upsert.

        Goes through the same ``embedding_func`` (and so the same concurrency
        limit) as ``embed_pending``. The vector is used only for a pending doc
        whose ``content`` is byte-identical, so a doc whose text changed in the
        meantime is embedded as usual.
        """
        if content in self._primed:
            return
        future = asyncio.ensure_future(self.embedding_func([content], context="document"))
        # A primed vector nobody takes (discarded, or its doc failed) must not
        # warn "exception was never retrieved"; embed_pending re-embeds on error.
        future.add_done_callback(lambda f: f.cancelled() or f.exception())
        self._primed[content] = future

    def discard_primed(self, contents: list[str]) -> None:
        """Drop primed embeddings that no pending doc took (cancels in-flight ones)."""
        for content in contents:
            future = self._primed.pop(content, None)
            if future is not None:
                future.cancel()

    async def _stored_vectors_for_unchanged(
        self, docs: list[tuple[str, "_PendingPGVectorDoc"]]
    ) -> dict[str, np.ndarray]:
        """Stored vectors of the docs whose stored ``content`` equals the pending one.

        Only for a model-suffixed table: without the suffix several embedding
        models share one table, and a stored vector may come from another model.
        Any read failure just means everything gets embedded.
        """
        if not self.model_suffix:
            return {}
        try:
            rows = await self.db.query(
                f"SELECT id, content, content_vector FROM {self.table_name} "
                f"WHERE workspace=$1 AND id = ANY($2) AND content_vector IS NOT NULL",
                [self.workspace, [doc_id for doc_id, _ in docs]],
                multirows=True,
            )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] {self.namespace}: could not read stored vectors "
                f"for reuse, embedding all: {e}"
            )
            return {}
        stored = {row["id"]: row for row in rows or [] if row}
        dim = self.embedding_func.embedding_dim
        reused: dict[str, np.ndarray] = {}
        for doc_id, pending in docs:
            row = stored.get(doc_id)
            if row is None or row.get("content") != pending.item["content"]:
                continue
            vector = _as_float32_vector(row.get("content_vector"))
            if vector is not None and vector.shape == (dim,):
                reused[doc_id] = vector
        return reused

    async def _flush_pending_vector_ops(self) -> None:
        """Flush buffered PG vector upserts and deletes in one transaction.

        Concurrency:
            All buffer reads/writes and destructive server mutations on
            this storage run under ``self._flush_lock``. Embedding runs first,
            outside that lock (``embed_pending``), so a slow provider does not
            block upserts from other documents' merges; a record replaced
            during that window keeps no stale vector and is embedded below,
            inside the lock, so nothing can interleave between its embedding
            and the PG write.

        Failure handling:
            PG cannot expose per-document statuses, so flush is
            all-or-nothing:
              * If embedding fails the buffers stay intact (next flush
                retries; cached vectors are reused).
              * If ``_run_with_retry`` raises the transaction rolls back
                and the buffers stay intact. Cached vectors stay attached
                to pending docs so the next flush does not re-embed.
              * On success both buffers are cleared.

        Post-finalize / pre-initialize:
            Calling this after ``finalize()`` (``self.db is None``) or
            before ``initialize()`` (``self._flush_lock is None``) with a
            non-empty buffer raises ``RuntimeError`` — silently dropping
            buffered writes would defeat the data-loss visibility that
            ``finalize()`` provides. An empty-buffer call is a no-op.
        """
        if self._flush_lock is None:
            pending_docs = len(self._pending_vector_docs)
            pending_deletes = len(self._pending_vector_deletes)
            if pending_docs or pending_deletes:
                raise RuntimeError(
                    f"[{self.workspace}] PGVectorStorage._flush_pending_vector_ops "
                    f"called before initialize(); {pending_docs} pending upserts "
                    f"and {pending_deletes} pending deletes cannot be flushed"
                )
            return

        await self.embed_pending()

        async with self._flush_lock:
            if not self._pending_vector_docs and not self._pending_vector_deletes:
                return
            if self.db is None:
                pending_docs = len(self._pending_vector_docs)
                pending_deletes = len(self._pending_vector_deletes)
                raise RuntimeError(
                    f"[{self.workspace}] PGVectorStorage._flush_pending_vector_ops "
                    f"called after client release; {pending_docs} pending upserts "
                    f"and {pending_deletes} pending deletes cannot be flushed"
                )

            timing_label = f"{self.workspace} PGVectorStorage.flush[{self.namespace}]"
            total_start = time.perf_counter()
            performance_timing_log(
                "[%s] start upserts=%s deletes=%s max_batch_size=%s",
                timing_label,
                len(self._pending_vector_docs),
                len(self._pending_vector_deletes),
                self._max_batch_size,
            )

            # --- Embedding phase ---------------------------------------------
            docs_to_embed = [
                (doc_id, pending_doc)
                for doc_id, pending_doc in self._pending_vector_docs.items()
                if pending_doc.vector is None
            ]
            if docs_to_embed:
                contents = [
                    pending_doc.item["content"] for _, pending_doc in docs_to_embed
                ]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                logger.info(
                    f"[{self.workspace}] {self.namespace} flush: embedding "
                    f"{len(docs_to_embed)} vectors in {len(batches)} batch(es) "
                    f"(batch_num={self._max_batch_size})"
                )
                embedding_start = time.perf_counter()
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error embedding pending vector ops "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise
                performance_timing_log(
                    "[%s] embedding completed in %.4fs docs=%s batches=%s",
                    timing_label,
                    time.perf_counter() - embedding_start,
                    len(docs_to_embed),
                    len(batches),
                )
                embeddings = np.concatenate(embeddings_list)
                # Explicit check: a count mismatch under `python -O` would
                # silently truncate via zip(), mispairing vectors with docs.
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: "
                        f"expected {len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((_, pending_doc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    pending_doc.vector = embedding
                    await _cooperative_yield(i)

            # --- Build batch tuples ------------------------------------------
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                build_tuple = self._upsert_chunks
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                build_tuple = self._upsert_entities
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                build_tuple = self._upsert_relationships
            else:
                raise ValueError(f"{self.namespace} is not supported")

            batch_values: list[tuple[Any, ...]] = []
            upsert_sql: str | None = None
            for i, (doc_id, pending_doc) in enumerate(
                self._pending_vector_docs.items(), start=1
            ):
                if pending_doc.vector is None:
                    # Should not happen: every pending doc was embedded above
                    # or had a cached vector from a previous lazy embed.
                    raise RuntimeError(
                        f"[{self.workspace}] Pending vector for id={doc_id} "
                        f"missing after embedding phase"
                    )
                # Coerce to float32 ndarray if not already (defensive; the
                # embedding func typically returns float32 but a custom
                # provider may return float64 — pgvector wants float32).
                item = dict(pending_doc.item)
                vector = pending_doc.vector
                if not isinstance(vector, np.ndarray) or vector.dtype != np.float32:
                    vector = np.asarray(vector, dtype=np.float32)
                item["__vector__"] = vector
                upsert_sql, values = build_tuple(item, pending_doc.created_at)
                batch_values.append(values)
                await _cooperative_yield(i)

            pending_delete_ids = list(self._pending_vector_deletes)

            # --- Persistence -------------------------------------------------
            # upsert and delete run as separate, payload/record-bounded phases
            # (mirrors mongo_impl). The two buffers are disjoint -- upsert()
            # discards from pending_deletes and delete() pops from pending_docs
            # -- so phase ordering is irrelevant. Each chunk is its own
            # transaction; both ops are idempotent (ON CONFLICT / ANY($2)), so a
            # mid-flush failure raises with the buffers intact and the next
            # flush replays everything (fail-fast-retain). This trades the old
            # single-transaction atomicity for bounded peak memory / tx duration.
            log_prefix = f"[{self.workspace}] {self.namespace} flush:"

            upsert_batches = (
                _chunk_by_budget(
                    batch_values,
                    _estimate_record_bytes,
                    self._max_upsert_payload_bytes,
                    self._max_upsert_records_per_batch,
                )
                if batch_values and upsert_sql
                else []
            )
            if len(upsert_batches) > 1:
                logger.info(
                    f"{log_prefix} upsert split into {len(upsert_batches)} batches "
                    f"for {len(batch_values)} records "
                    f"(max_payload={self._max_upsert_payload_bytes} batch={self._max_upsert_records_per_batch})"
                )

            # ``or 1`` guards an upsert-only flush: with the delete cap disabled
            # (<= 0) and no pending deletes, the fallback would be 0 and the
            # range() step below would raise even though there is nothing to
            # delete. The empty-list loop then simply no-ops.
            delete_chunk = (
                self._max_delete_records_per_batch
                if self._max_delete_records_per_batch > 0
                else len(pending_delete_ids) or 1
            )
            if pending_delete_ids and len(pending_delete_ids) > delete_chunk:
                logger.info(
                    f"{log_prefix} delete {len(pending_delete_ids)} ids split "
                    f"into chunks (chunk={delete_chunk})"
                )
            delete_sql = (
                f"DELETE FROM {self.table_name} WHERE workspace=$1 AND id = ANY($2)"
            )

            try:
                for batch_index, (sub_batch, estimated_bytes) in enumerate(
                    upsert_batches, start=1
                ):
                    if (
                        len(sub_batch) == 1
                        and self._max_upsert_payload_bytes > 0
                        and estimated_bytes > self._max_upsert_payload_bytes
                    ):
                        logger.warning(
                            f"{log_prefix} single record id={sub_batch[0][1]} "
                            f"estimated {estimated_bytes} bytes exceeds {self._max_upsert_payload_bytes}"
                        )

                    async def _flush_upsert(
                        connection: asyncpg.Connection,
                        _sql: str = upsert_sql,
                        _data: list[tuple] = sub_batch,
                        _batch_index: int = batch_index,
                        _num_batches: int = len(upsert_batches),
                    ) -> None:
                        async with connection.transaction():
                            execute_start = time.perf_counter()
                            await connection.executemany(_sql, _data)
                            performance_timing_log(
                                "[%s] sub-batch %s/%s executemany completed in %.4fs batch_size=%s",
                                timing_label,
                                _batch_index,
                                _num_batches,
                                time.perf_counter() - execute_start,
                                len(_data),
                            )

                    await self.db._run_with_retry(
                        _flush_upsert, timing_label=timing_label
                    )

                for i in range(0, len(pending_delete_ids), delete_chunk):
                    id_slice = pending_delete_ids[i : i + delete_chunk]

                    async def _flush_delete(
                        connection: asyncpg.Connection,
                        _ids: list[str] = id_slice,
                    ) -> None:
                        async with connection.transaction():
                            await connection.execute(delete_sql, self.workspace, _ids)

                    await self.db._run_with_retry(
                        _flush_delete, timing_label=timing_label
                    )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error flushing vector ops "
                    f"(upserts={len(batch_values)}, "
                    f"deletes={len(pending_delete_ids)}): {e}"
                )
                raise

            # Success: clear committed buffers. Cached vectors live on
            # those records and are GC'd with them.
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()
            performance_timing_log(
                "[%s] total complete in %.4fs upserts=%s deletes=%s",
                timing_label,
                time.perf_counter() - total_start,
                len(batch_values),
                len(pending_delete_ids),
            )

    async def query(self, query: str, top_k: int, query_embedding: list[float] = None) -> list[dict[str, Any]]:
        """Nearest rows for ``query``; the chunks namespace fuses vector and BM25 with RRF."""
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding = (await self.embedding_func([query], context="query"))[0]

        # Use positional $4 parameter instead of string-interpolated literal.
        # asyncpg sends the embedding via register_vector binary codec, avoiding
        # per-query text serialization and PostgreSQL text-to-vector parsing.
        vector_cast = (
            "halfvec"
            if getattr(self.db, "vector_index_type", None) == "HNSW_HALFVEC"
            else "vector"
        )
        if self.namespace == "chunks":
            candidates = settings.hybrid_search_chunk_candidates
            w_semantic = settings.hybrid_search_w_semantic
            w_keyword = settings.hybrid_search_w_keyword
            query_cleaned = preprocess_indonesian_fast(query)
            sql = SQL_TEMPLATES["chunks_hybrid_bm25"].format(
                table_name=self.table_name, vector_cast=vector_cast
            )
            params = {
                "workspace": self.workspace,
                "closer_than": 1 - self.cosine_better_than_threshold,
                "candidates": candidates,
                "embedding": embedding,
                "bm25_query": query_cleaned,
                "w_semantic": w_semantic,
                "w_keyword": w_keyword,
            }
        else:
            sql = SQL_TEMPLATES[self.namespace].format(
                table_name=self.table_name, vector_cast=vector_cast
            )
            params = {
                "workspace": self.workspace,
                "closer_than_threshold": 1 - self.cosine_better_than_threshold,
                "top_k": top_k,
                "embedding": embedding,
            }
        return await self.db.query(sql, params=list(params.values()), multirows=True)

    async def index_done_callback(self) -> None:
        await self._flush_pending_vector_ops()

    async def drop_pending_index_ops(self) -> None:
        """Discard buffered upserts/deletes (pipeline aborting on error)."""
        async with self._flush_lock:
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()

    async def get_chunk_ids_by_doc_id(self, doc_id: str) -> list[str]:
        """Every vector-table chunk id for ``doc_id`` — see the KV twin.

        Buffered upserts that have not flushed yet are invisible here; deletion
        unions this with ``doc_status.chunks_list`` so a chunk in flight is not
        missed.
        """
        if not is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
            raise ValueError(
                f"get_chunk_ids_by_doc_id is only valid for the chunk vector "
                f"namespace, not {self.namespace!r}"
            )
        sql = f"SELECT id FROM {self.table_name} WHERE workspace=$1 AND full_doc_id=$2"
        rows = await self.db.query(sql, [self.workspace, doc_id], multirows=True)
        return [str(row["id"]) for row in (rows or []) if row.get("id")]

    async def delete(self, ids: list[str]) -> None:
        """Buffer vector deletes for batched flush.

        A delete cancels any pending upsert for the same id. The actual PG
        delete is performed by ``_flush_pending_vector_ops`` during the next
        ``index_done_callback`` / ``finalize`` call.
        """
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        async with self._flush_lock:
            for doc_id in ids:
                self._pending_vector_docs.pop(doc_id, None)
                self._pending_vector_deletes.add(doc_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for {len(ids)} vectors in {self.namespace}"
        )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID with read-your-writes against the buffer.

        The embedding column is stripped from BOTH the buffered and the
        SQL-fallback result (``__vector__``/``__id__`` from the buffer,
        ``content_vector`` from the row) so the shapes match each other and
        the other vector backends. The raw ``content_vector`` is a pgvector
        value that the asyncpg codec returns as a numpy array, which is not
        JSON-serializable and would break callers that return this dict in an
        API response (e.g. ``/graph/entity/edit``). Callers needing embeddings
        must use ``get_vectors_by_ids``.

        Response shape:
            ``{"id", "content", <payload fields>, "created_at"}`` — no
            embedding column, from either path.
        """
        async with self._flush_lock:
            if id in self._pending_vector_deletes:
                return None
            pending = self._pending_vector_docs.get(id)
            if pending is not None:
                doc = {
                    k: v
                    for k, v in pending.item.items()
                    if k not in ("__id__", "__vector__")
                }
                doc["id"] = id
                doc["created_at"] = int(pending.created_at.timestamp())
                return doc

        query = (
            f"SELECT *, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at "
            f"FROM {self.table_name} WHERE workspace=$1 AND id=$2"
        )
        try:
            result = await self.db.query(query, [self.workspace, id])
            if result:
                row = dict(result)
                # Drop the embedding column: it is a numpy array (pgvector
                # codec) and not JSON-serializable; matches the buffered shape.
                row.pop("content_vector", None)
                return row
            return None
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector docs by ID, preserving caller order.

        Pending deletes return ``None`` in their slot. Pending upserts are
        served from the buffer; remaining ids fall through to a single
        parameterized ``id = ANY($2)`` SQL query (replacing the previous
        string-built ``IN (...)`` form).

        Response shape: same buffered-vs-SQL inconsistency as
        ``get_by_id`` — see that docstring for details.
        """
        if not ids:
            return []

        buffered: dict[str, dict[str, Any] | None] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    buffered[doc_id] = None
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    doc = {
                        k: v
                        for k, v in pending.item.items()
                        if k not in ("__id__", "__vector__")
                    }
                    doc["id"] = doc_id
                    doc["created_at"] = int(pending.created_at.timestamp())
                    buffered[doc_id] = doc
                    continue
                remaining.append(doc_id)

        id_map: dict[str, dict[str, Any]] = {}
        if remaining:
            query = (
                f"SELECT *, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at "
                f"FROM {self.table_name} WHERE workspace=$1 AND id = ANY($2)"
            )
            try:
                results = await self.db.query(
                    query, [self.workspace, remaining], multirows=True
                )
                for record in results or []:
                    if record is None:
                        continue
                    record_dict = dict(record)
                    # Drop the (numpy / non-JSON-serializable) embedding column
                    # so the SQL shape matches the buffered shape.
                    record_dict.pop("content_vector", None)
                    row_id = record_dict.get("id")
                    if row_id is not None:
                        id_map[str(row_id)] = record_dict
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
                )
                return []

        ordered_results: list[dict[str, Any] | None] = []
        for requested_id in ids:
            if requested_id in buffered:
                ordered_results.append(buffered[requested_id])
            else:
                ordered_results.append(id_map.get(str(requested_id)))
        return ordered_results

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vector embeddings by ID, with read-your-writes against the buffer.

        Lazily embeds pending docs whose vector has not been computed yet,
        caches the result on the pending record (so the next flush reuses
        it), and falls through to a parameterized SQL query for ids not in
        the buffer.

        Embedding I/O runs *outside* ``_flush_lock`` so a slow embedding
        provider cannot block concurrent ``upsert`` / ``delete`` / read
        calls on this storage. The lock is re-acquired briefly to cache
        the result, and the pending record's identity is re-checked
        first: if a concurrent ``upsert`` / ``delete`` / ``drop`` replaced
        or removed the record during the embedding window, that ID is
        dropped from the response entirely — we neither cache the stale
        vector on the new/missing record nor return it to the caller, so
        callers cannot observe an embedding that no longer matches the
        current buffer state. Affected callers should treat the missing
        key the same as the existing "id was deleted before the call"
        case and retry if needed.
        """
        if not ids:
            return {}

        result: dict[str, list[float]] = {}
        remaining: list[str] = []
        docs_to_embed: list[tuple[str, _PendingPGVectorDoc]] = []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    if pending.vector is None:
                        docs_to_embed.append((doc_id, pending))
                    else:
                        result[doc_id] = pending.vector.tolist()
                    continue
                remaining.append(doc_id)

        if docs_to_embed:
            contents = [pending_doc.item["content"] for _, pending_doc in docs_to_embed]
            batches = [
                contents[i : i + self._max_batch_size]
                for i in range(0, len(contents), self._max_batch_size)
            ]
            try:
                embeddings_list = await asyncio.gather(
                    *[
                        self.embedding_func(batch, context="document")
                        for batch in batches
                    ]
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error lazily embedding pending vectors "
                    f"(upserts={len(docs_to_embed)}): {e}"
                )
                raise
            embeddings = np.concatenate(embeddings_list)
            if len(embeddings) != len(docs_to_embed):
                raise RuntimeError(
                    f"[{self.workspace}] Embedding count mismatch: "
                    f"expected {len(docs_to_embed)}, got {len(embeddings)}"
                )

            # Re-acquire the lock just long enough to cache results on
            # the same record. The identity check gates BOTH the cache
            # write and the response entry: if the pending record was
            # swapped or removed during the embedding window (concurrent
            # upsert / delete / drop), the just-computed vector no longer
            # matches the current buffer state for this id, so we drop it
            # from the response rather than return a stale embedding.
            async with self._flush_lock:
                for i, ((doc_id, original_pending), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    current = self._pending_vector_docs.get(doc_id)
                    if current is original_pending:
                        current.vector = embedding
                        result[doc_id] = embedding.tolist()
                    await _cooperative_yield(i)

        if not remaining:
            return result

        query = (
            f"SELECT id, content_vector FROM {self.table_name} "
            f"WHERE workspace=$1 AND id = ANY($2)"
        )
        try:
            results = await self.db.query(
                query, [self.workspace, remaining], multirows=True
            )
            for row in results or []:
                if not row or "content_vector" not in row or "id" not in row:
                    continue
                vector_data = row["content_vector"]
                try:
                    if isinstance(vector_data, (list, tuple)):
                        result[row["id"]] = list(vector_data)
                    elif isinstance(vector_data, str):
                        parsed = json.loads(vector_data)
                        if isinstance(parsed, list):
                            result[row["id"]] = parsed
                    elif hasattr(vector_data, "tolist"):
                        result[row["id"]] = vector_data.tolist()
                    elif hasattr(vector_data, "to_list") and callable(
                        vector_data.to_list
                    ):
                        result[row["id"]] = vector_data.to_list()
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"[{self.workspace}] Failed to parse vector data for ID {row['id']}: {e}"
                    )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting vectors: {e}")

        return result

