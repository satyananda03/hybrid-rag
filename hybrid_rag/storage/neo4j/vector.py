"""``Neo4jVectorStorage`` — embeddings and BM25 text in Neo4j, next to the graph.

Chunk vectors live on ``(:HybridRagChunk)`` nodes, entity vectors on the graph's
``(:HybridRagEntity)`` nodes and relation vectors on its ``[:DIRECTED]`` edges (layout
and queries: :mod:`hybrid_rag.storage.neo4j.cypher`). :meth:`Neo4jVectorStorage.query`
runs, for chunks, a vector search and a BM25 search fused with RRF; for entities and
relations a vector search.

Writes are buffered: ``upsert``/``delete`` queue work that ``index_done_callback()``
embeds and flushes in batches. An entity/relation vector can only be written once its
graph element exists, and the graph writes a document's merge at the end of that merge,
possibly after another document's flush. Such a vector therefore stays buffered and is
retried on the next flush (see :data:`_UNMATCHED_TTL`).
"""

from hybrid_rag.core.concurrency import _cooperative_yield
import asyncio
import time
import datetime
from datetime import timezone
from dataclasses import dataclass, field
from typing import Any, Callable, final

import numpy as np
from neo4j import AsyncManagedTransaction  # type: ignore

from hybrid_rag.config import settings
from hybrid_rag.core.base import BaseVectorStorage
from hybrid_rag.core.constants import NameSpace, is_namespace
from hybrid_rag.core.utils import validate_workspace
from hybrid_rag.storage.neo4j import cypher
from hybrid_rag.storage.neo4j.bm25 import preprocess_indonesian_fast
from hybrid_rag.storage.neo4j.connection import get_neo4j_driver
from hybrid_rag.storage.shared import get_data_init_lock, get_namespace_lock
import logging

logger = logging.getLogger(__name__)

# Rows per write transaction: 200 vectors of 1024 floats is ~1.6 MB.
_WRITE_BATCH = 200
# How long an entity/relation vector waits in the buffer for its graph element.
# ponytail: fixed TTL; a vector whose document's graph write comes later than this is
# dropped (warned) — tie it to the document's merge if that ever happens.
_UNMATCHED_TTL = datetime.timedelta(hours=1)


def _as_float32_vector(value: Any) -> np.ndarray | None:
    """A stored vector as a float32 array (Neo4j returns a list of floats)."""
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _as_list(vector: Any) -> list[float]:
    """A vector as the plain list the driver sends."""
    return np.asarray(vector, dtype=np.float32).tolist()


@dataclass
class _PendingVectorDoc:
    """Buffered vector upsert awaiting embedding and batched flush."""

    item: dict[str, Any]
    created_at: datetime.datetime
    vector: np.ndarray | None = None


def _chunk_row(item: dict[str, Any], workspace: str) -> dict[str, Any]:
    created = int(item["__created_at__"].replace(tzinfo=timezone.utc).timestamp())
    return {
        "id": item["__id__"],
        "created_at": created,
        "props": {
            "full_doc_id": item["full_doc_id"],
            "chunk_order_index": item["chunk_order_index"],
            "tokens": item["tokens"],
            "content": item["content"],
            "file_path": item["file_path"],
            # BM25 runs on the contextualized chunk text ("{context}. {child_content}"),
            # stopwords stripped (Sastrawi).
            "content_cleaned": preprocess_indonesian_fast(item["content"]),
            "ws_key": cypher.ws_key(workspace),
            "updated_at": created,
        },
    }


def _entity_row(item: dict[str, Any], workspace: str) -> dict[str, Any]:
    return {"id": item["__id__"], "entity_name": item["entity_name"], "content": item["content"]}


def _relation_row(item: dict[str, Any], workspace: str) -> dict[str, Any]:
    return {
        "id": item["__id__"], "src_id": item["src_id"], "tgt_id": item["tgt_id"],
        "content": item["content"],
    }


def _sorted_pair(row: dict[str, Any]) -> dict[str, Any]:
    """Relation hits as the vector table returned them: endpoints sorted (writers sort them)."""
    src, tgt = sorted((row["src_id"], row["tgt_id"]))
    return {"src_id": src, "tgt_id": tgt, "created_at": row.get("created_at")}


@dataclass(frozen=True)
class _Target:
    """Everything that differs between the chunk, entity and relation namespaces."""

    to_row: Callable[[dict[str, Any], str], dict[str, Any]]
    upsert: str
    delete: str
    by_id: str
    search: str | None  # None: chunks, whose search is cypher.chunk_hybrid_search()
    hit: Callable[[dict[str, Any]], dict[str, Any]]
    graph_owned: bool  # the vector attaches to an element the graph storage creates


_TARGETS = {
    NameSpace.VECTOR_STORE_CHUNKS: _Target(
        _chunk_row, cypher.CHUNK_UPSERT, cypher.CHUNK_DELETE, cypher.CHUNK_BY_ID,
        None, dict, graph_owned=False,
    ),
    NameSpace.VECTOR_STORE_ENTITIES: _Target(
        _entity_row, cypher.ENTITY_UPSERT, cypher.ENTITY_DELETE, cypher.ENTITY_BY_ID,
        cypher.ENTITY_SEARCH,
        lambda r: {"entity_name": r["entity_name"], "created_at": r["created_at"]},
        graph_owned=True,
    ),
    NameSpace.VECTOR_STORE_RELATIONSHIPS: _Target(
        _relation_row, cypher.RELATION_UPSERT, cypher.RELATION_DELETE, cypher.RELATION_BY_ID,
        cypher.RELATION_SEARCH, _sorted_pair, graph_owned=True,
    ),
}


def _target(namespace: str) -> _Target:
    for name, target in _TARGETS.items():
        if is_namespace(namespace, name):
            return target
    raise ValueError(f"{namespace} is not a vector namespace")


@final
@dataclass
class Neo4jVectorStorage(BaseVectorStorage):
    _driver: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        validate_workspace(self.workspace)
        self._validate_embedding_func()
        self._max_batch_size = self.global_config["embedding_batch_num"]
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = config.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold
        self._target = _target(self.namespace)

        # The vector property and index are named after the configured model
        # (cypher.SUFFIX); vectors of another model must never be written into them.
        self.model_suffix = self._generate_collection_suffix()
        if (self.model_suffix or "default") != cypher.SUFFIX:
            raise ValueError(
                f"embedding_func model {self.model_suffix!r} does not match "
                f"EMBEDDING_MODEL/EMBEDDING_DIM ({cypher.SUFFIX!r})"
            )

        # Pending buffers: upsert() and delete() queue work here until
        # _flush_pending_vector_ops() runs from index_done_callback() / finalize().
        self._pending_vector_docs: dict[str, _PendingVectorDoc] = {}
        self._pending_vector_deletes: set[str] = set()
        # Namespace-keyed lock; created in initialize() after workspace is final.
        self._flush_lock = None
        # Serialises embed_pending() runs so two flushes never embed the same
        # pending doc twice; created on first use (binds to the running loop).
        self._embed_lock: asyncio.Lock | None = None
        # Embeddings started by prime() before their doc is upserted, keyed by
        # the exact content; embed_pending() takes them instead of re-embedding.
        self._primed: dict[str, asyncio.Future] = {}

    async def initialize(self):
        async with get_data_init_lock():
            if self._driver is None:
                # One driver for every storage (connection.py); never build one here.
                self._driver = get_neo4j_driver()
            if not self.workspace:
                self.workspace = "default"
            # Indexes are created by `make db-init` (hybrid_rag.db_init).

        if self._flush_lock is None:
            # Keyed by role too: retrieval only reads (get_vectors_by_ids checks
            # the pending buffer under this lock) and must never queue behind
            # an ingestion flush of the same workspace.
            self._flush_lock = get_namespace_lock(
                f"{self.namespace}:{self.global_config['role']}", workspace=self.workspace
            )

    async def finalize(self):
        """Flush pending vector ops, then drop this instance's driver reference.

        Captures regular ``Exception`` from the flush so it can be re-raised as a
        ``RuntimeError`` naming the unflushed buffer counts. ``BaseException``
        (``CancelledError``, ``KeyboardInterrupt``, ``SystemExit``) propagates.

        Entity/relation vectors still waiting for their graph element after the final
        flush are dropped with a warning: there is nothing to attach them to.
        Re-entry after the driver was released still raises if buffers remain.
        """
        if self._driver is None:
            pending_docs = len(self._pending_vector_docs)
            pending_deletes = len(self._pending_vector_deletes)
            if pending_docs or pending_deletes:
                raise RuntimeError(
                    f"[{self.workspace}] Neo4jVectorStorage.finalize() re-entry: "
                    f"driver already released; {pending_docs} pending upserts "
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
            if flush_error is None and self._target.graph_owned and self._pending_vector_docs:
                logger.warning(
                    f"[{self.workspace}] {self.namespace}: dropping "
                    f"{len(self._pending_vector_docs)} vectors whose graph element never appeared"
                )
                self._pending_vector_docs.clear()
            self._driver = None

        pending_docs = len(self._pending_vector_docs)
        pending_deletes = len(self._pending_vector_deletes)
        if flush_error is not None:
            raise RuntimeError(
                f"[{self.workspace}] Neo4jVectorStorage.finalize() flush raised; "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes were left buffered (driver released, data lost)"
            ) from flush_error
        if pending_docs or pending_deletes:
            raise RuntimeError(
                f"[{self.workspace}] Neo4jVectorStorage.finalize() left "
                f"{pending_docs} pending upserts and {pending_deletes} "
                f"pending deletes buffered after final flush attempt"
            )

    # ── Neo4j access ───────────────────────────────────────────────────────
    async def _read(self, query: str, **params) -> list[dict[str, Any]]:
        """Run a read query with ``$workspace`` bound to this instance's workspace."""
        params.setdefault("workspace", self.workspace)
        async with self._driver.session(
            database=settings.neo4j_database or None, default_access_mode="READ"
        ) as session:
            result = await session.run(query, params)
            return [record.data() async for record in result]

    async def _write(self, query: str, **params) -> list[dict[str, Any]]:
        """Run a write query in a managed (retried-by-driver) transaction."""
        params.setdefault("workspace", self.workspace)

        async def work(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(query, params)
            return [record.data() async for record in result]

        async with self._driver.session(database=settings.neo4j_database or None) as session:
            return await session.execute_write(work)

    # ── Buffered writes ────────────────────────────────────────────────────
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vector docs for embedding and batched flush.

        Correctness premise:
            The ingestion pipeline is the only writer and guarantees a single
            writer per workspace. The pending buffer is process-local:
            committed rows are visible to every reader, buffered writes only
            after index_done_callback().
        """
        if not data:
            return

        logger.debug(
            f"[{self.workspace}] Buffering {len(data)} vectors for {self.namespace}"
        )

        current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
        pending_docs: list[tuple[str, _PendingVectorDoc]] = []
        for i, (k, v) in enumerate(data.items(), start=1):
            pending_docs.append(
                (k, _PendingVectorDoc(item={"__id__": k, **v}, created_at=current_time))
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

        A doc whose ``content`` is byte-identical to the one already stored under
        its id reuses that stored vector instead of calling the model: same text and
        same model give the same vector, and the property is per model+dimension.
        """
        if self._flush_lock is None or self._driver is None:
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
                embedding_start = time.perf_counter()
                embeddings = await self._embed([p.item["content"] for _, p in to_embed])
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

    async def _embed(self, contents: list[str]) -> np.ndarray:
        """Embed ``contents`` in ``embedding_batch_num`` batches, one vector per text."""
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        try:
            embeddings_list = await asyncio.gather(
                *[self.embedding_func(batch, context="document") for batch in batches]
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error embedding {len(contents)} vectors for {self.namespace}: {e}"
            )
            raise
        embeddings = np.concatenate(embeddings_list)
        # Explicit check: a count mismatch under `python -O` would silently
        # truncate via zip(), mispairing vectors with docs.
        if len(embeddings) != len(contents):
            raise RuntimeError(
                f"[{self.workspace}] Embedding count mismatch: "
                f"expected {len(contents)}, got {len(embeddings)}"
            )
        return embeddings

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
        self, docs: list[tuple[str, _PendingVectorDoc]]
    ) -> dict[str, np.ndarray]:
        """Stored vectors of the docs whose stored ``content`` equals the pending one.

        Any read failure just means everything gets embedded.
        """
        try:
            rows = await self._read(self._target.by_id, ids=[doc_id for doc_id, _ in docs])
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] {self.namespace}: could not read stored vectors "
                f"for reuse, embedding all: {e}"
            )
            return {}
        stored = {row["id"]: row for row in rows}
        dim = self.embedding_func.embedding_dim
        reused: dict[str, np.ndarray] = {}
        for doc_id, pending in docs:
            row = stored.get(doc_id)
            if row is None or row.get("content") != pending.item["content"]:
                continue
            vector = _as_float32_vector(row.get("vector"))
            if vector is not None and vector.shape == (dim,):
                reused[doc_id] = vector
        return reused

    async def _flush_pending_vector_ops(self) -> None:
        """Flush buffered vector upserts and deletes.

        Concurrency:
            All buffer reads/writes and destructive server mutations on this
            storage run under ``self._flush_lock``. Embedding runs first, outside
            that lock (``embed_pending``), so a slow provider does not block
            upserts from other documents' merges; a record replaced during that
            window keeps no stale vector and is embedded below, inside the lock.

        Failure handling:
            Each batch is its own transaction; upserts (MERGE/SET) and deletes are
            idempotent, so a mid-flush failure raises with the buffers intact and
            the next flush replays everything (fail-fast-retain).

        Entity/relation vectors whose graph element does not exist yet stay
        buffered for the next flush, up to :data:`_UNMATCHED_TTL`.

        Post-finalize / pre-initialize:
            Calling this after ``finalize()`` or before ``initialize()`` with a
            non-empty buffer raises ``RuntimeError``. An empty-buffer call is a no-op.
        """
        if self._flush_lock is None:
            pending_docs = len(self._pending_vector_docs)
            pending_deletes = len(self._pending_vector_deletes)
            if pending_docs or pending_deletes:
                raise RuntimeError(
                    f"[{self.workspace}] Neo4jVectorStorage._flush_pending_vector_ops "
                    f"called before initialize(); {pending_docs} pending upserts "
                    f"and {pending_deletes} pending deletes cannot be flushed"
                )
            return

        await self.embed_pending()

        async with self._flush_lock:
            if not self._pending_vector_docs and not self._pending_vector_deletes:
                return
            if self._driver is None:
                pending_docs = len(self._pending_vector_docs)
                pending_deletes = len(self._pending_vector_deletes)
                raise RuntimeError(
                    f"[{self.workspace}] Neo4jVectorStorage._flush_pending_vector_ops "
                    f"called after driver release; {pending_docs} pending upserts "
                    f"and {pending_deletes} pending deletes cannot be flushed"
                )

            # A record replaced during embed_pending() has no vector yet.
            docs_to_embed = [
                pending for pending in self._pending_vector_docs.values() if pending.vector is None
            ]
            if docs_to_embed:
                embeddings = await self._embed([p.item["content"] for p in docs_to_embed])
                for i, (pending, embedding) in enumerate(zip(docs_to_embed, embeddings), start=1):
                    pending.vector = embedding
                    await _cooperative_yield(i)

            rows = []
            for i, pending in enumerate(self._pending_vector_docs.values(), start=1):
                row = self._target.to_row(
                    {**pending.item, "__created_at__": pending.created_at}, self.workspace
                )
                row["vector"] = _as_list(pending.vector)
                rows.append(row)
                await _cooperative_yield(i)
            delete_ids = list(self._pending_vector_deletes)

            written: set[str] = set()
            try:
                for i in range(0, len(rows), _WRITE_BATCH):
                    result = await self._write(self._target.upsert, rows=rows[i : i + _WRITE_BATCH])
                    written.update(r["id"] for r in result)
                for i in range(0, len(delete_ids), _WRITE_BATCH):
                    await self._write(self._target.delete, ids=delete_ids[i : i + _WRITE_BATCH])
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error flushing vector ops "
                    f"(upserts={len(rows)}, deletes={len(delete_ids)}): {e}"
                )
                raise

            self._pending_vector_deletes.clear()
            self._keep_unmatched(written)

    def _keep_unmatched(self, written: set[str]) -> None:
        """Clear flushed docs; keep the ones whose graph element is not there yet."""
        expired = datetime.datetime.now(timezone.utc).replace(tzinfo=None) - _UNMATCHED_TTL
        waiting = {
            doc_id: pending
            for doc_id, pending in self._pending_vector_docs.items()
            if doc_id not in written and pending.created_at > expired
        }
        dropped = len(self._pending_vector_docs) - len(written) - len(waiting)
        if dropped:
            logger.warning(
                f"[{self.workspace}] {self.namespace}: dropped {dropped} vectors whose "
                f"graph element did not appear within {_UNMATCHED_TTL}"
            )
        if waiting:
            logger.debug(
                f"[{self.workspace}] {self.namespace}: {len(waiting)} vectors wait for their graph element"
            )
        self._pending_vector_docs = waiting

    async def index_done_callback(self) -> None:
        await self._flush_pending_vector_ops()

    async def drop_pending_index_ops(self) -> None:
        """Discard buffered upserts/deletes (pipeline aborting on error)."""
        async with self._flush_lock:
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()

    async def delete(self, ids: list[str]) -> None:
        """Buffer vector deletes for batched flush.

        A delete cancels any pending upsert for the same id. For an entity or a
        relation only its vector goes; the node or edge belongs to the graph storage.
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

    # ── Search ─────────────────────────────────────────────────────────────
    async def query(self, query: str, top_k: int, query_embedding: list[float] = None) -> list[dict[str, Any]]:
        """Nearest rows for ``query``; the chunks namespace fuses vector and BM25 with RRF."""
        if query_embedding is None:
            query_embedding = (await self.embedding_func([query], context="query"))[0]
        params = {
            "embedding": _as_list(query_embedding),
            # cos > threshold, in Neo4j's (1 + cos) / 2 score.
            "min_score": cypher.score_floor(1 - self.cosine_better_than_threshold),
        }
        if self._target.search is None:
            query_cleaned = preprocess_indonesian_fast(query)
            rows = await self._read(
                cypher.chunk_hybrid_search(with_bm25=bool(query_cleaned)),
                candidates=settings.hybrid_search_chunk_candidates,
                w_semantic=settings.hybrid_search_w_semantic,
                w_keyword=settings.hybrid_search_w_keyword,
                bm25_query=cypher.bm25_query(self.workspace, query_cleaned),
                **params,
            )
        else:
            rows = await self._read(self._target.search, top_k=top_k, **params)
        return [self._target.hit(row) for row in rows]

    async def nearest_entities(
        self, embeddings: list, top_k: int, max_distance: float
    ) -> list[list[tuple[str, float]]]:
        """Per embedding, the workspace's closest stored entities as (name, cosine distance).

        Entity resolution candidates only. Vectors still buffered for the next flush
        are not visible, like ``query``.
        """

        async def one(embedding) -> list[tuple[str, float]]:
            rows = await self._read(
                cypher.ENTITY_SEARCH,
                embedding=_as_list(embedding),
                top_k=top_k,
                min_score=cypher.score_floor(max_distance),
            )
            return [(r["entity_name"], cypher.distance(r["score"])) for r in rows]

        return list(await asyncio.gather(*(one(e) for e in embeddings)))

    # ── Reads by id ────────────────────────────────────────────────────────
    async def get_chunk_ids_by_doc_id(self, doc_id: str) -> list[str]:
        """Every stored chunk id for ``doc_id`` — see the KV twin.

        Buffered upserts that have not flushed yet are invisible here; deletion
        unions this with ``doc_status.chunks_list`` so a chunk in flight is not
        missed.
        """
        if not is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
            raise ValueError(
                f"get_chunk_ids_by_doc_id is only valid for the chunk vector "
                f"namespace, not {self.namespace!r}"
            )
        rows = await self._read(cypher.CHUNK_IDS_OF_DOC, doc_id=doc_id)
        return [str(row["id"]) for row in rows if row.get("id")]

    def _buffered(self, doc_id: str, pending: _PendingVectorDoc) -> dict[str, Any]:
        doc = {k: v for k, v in pending.item.items() if k != "__id__"}
        doc["id"] = doc_id
        doc["created_at"] = int(pending.created_at.timestamp())
        return doc

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """One stored doc, without its vector, with read-your-writes against the buffer.

        Response shape: ``{"id", "content", <payload fields>, "created_at"}``.
        Callers needing embeddings must use ``get_vectors_by_ids``.
        """
        docs = await self.get_by_ids([id])
        return docs[0] if docs else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        """Stored docs by id, in caller order; ``None`` for a missing or pending-deleted id."""
        if not ids:
            return []

        buffered: dict[str, dict[str, Any] | None] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    buffered[doc_id] = None
                elif (pending := self._pending_vector_docs.get(doc_id)) is not None:
                    buffered[doc_id] = self._buffered(doc_id, pending)
                else:
                    remaining.append(doc_id)

        stored: dict[str, dict[str, Any]] = {}
        if remaining:
            try:
                rows = await self._read(self._target.by_id, ids=remaining)
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
                )
                return []
            stored = {
                row["id"]: {"id": row["id"], "content": row["content"], **row["fields"]}
                for row in rows
            }
        return [buffered[i] if i in buffered else stored.get(i) for i in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vector embeddings by ID, with read-your-writes against the buffer.

        Lazily embeds pending docs whose vector has not been computed yet,
        caches the result on the pending record (so the next flush reuses
        it), and reads the stored vector for ids not in the buffer.

        Embedding I/O runs *outside* ``_flush_lock``. The lock is re-acquired
        briefly to cache the result, and the pending record's identity is
        re-checked first: if a concurrent ``upsert`` / ``delete`` / ``drop``
        replaced or removed the record during the embedding window, that ID is
        dropped from the response entirely, like an id deleted before the call.
        """
        if not ids:
            return {}

        result: dict[str, list[float]] = {}
        remaining: list[str] = []
        docs_to_embed: list[tuple[str, _PendingVectorDoc]] = []
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
            embeddings = await self._embed([p.item["content"] for _, p in docs_to_embed])
            async with self._flush_lock:
                for i, ((doc_id, original_pending), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    current = self._pending_vector_docs.get(doc_id)
                    if current is original_pending:
                        current.vector = embedding
                        result[doc_id] = embedding.tolist()
                    await _cooperative_yield(i)

        if remaining:
            try:
                rows = await self._read(self._target.by_id, ids=remaining)
                result.update((row["id"], list(row["vector"])) for row in rows if row["vector"])
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting vectors: {e}")
        return result
