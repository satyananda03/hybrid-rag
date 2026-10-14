
from hybrid_rag.core.concurrency import limit_async_func_call

import asyncio
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from typing import (
    Any,
    cast,
    final,
    Optional,
)

from hybrid_rag.storage.shared import (
    get_default_workspace,
    set_default_workspace,
)

from hybrid_rag.core.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocStatusStorage,
    StorageNameSpace,
    StoragesStatus,
)
from hybrid_rag.core.namespace import NameSpace
from hybrid_rag.ingestion.knowledge_graph.extraction import (
    extract_entities,
)
from hybrid_rag.core.exceptions import IndexFlushError
from hybrid_rag.core.utils import (
    Tokenizer,
    get_tokenizer,
    EmbeddingFunc,
    normalize_source_ids_limit_method,
)
import logging

from hybrid_rag.config import settings
from hybrid_rag.ingestion.worker import _PipelineMixin
from hybrid_rag.storage.neo4j import Neo4JStorage
from hybrid_rag.storage.postgres import PGDocStatusStorage, PGKVStorage, PGVectorStorage
from hybrid_rag.ingestion.deletion.deletion import _DeletionMixin
from hybrid_rag.ingestion.insert import _IngestionMixin
from hybrid_rag.ingestion.multimodal.analyze import _AnalyzeMixin
from hybrid_rag.ingestion.multimodal.equation_processor import (
    EquationModalProcessor,
)
from hybrid_rag.ingestion.multimodal.image_processor import (
    ImageModalProcessor,
)
from hybrid_rag.ingestion.multimodal.table_processor import (
    TableModalProcessor,
)

logger = logging.getLogger(__name__)


@final
@dataclass
class IngestionRAG(
    _PipelineMixin,
    _IngestionMixin,
    _AnalyzeMixin,
    _DeletionMixin,
):
    """Graph + vector RAG over PostgreSQL and Neo4j, with multimodal ingestion.

    One object owns the whole flow: text and multimodal content are ingested
    through :meth:`insert_content_list`, chunked parent-child, extracted into a
    knowledge graph, and queried back out. There is no inner engine to reach
    through — the storages, the pipeline, and the multimodal processors all hang
    off this instance.

    Data isolation is per :attr:`workspace`: every storage instance below is
    constructed with it, so two ``IngestionRAG`` objects with different workspaces
    share a database but never each other's rows, nodes, or pipeline state.
    """

    # Workspace
    # ---

    workspace: str = field(default="")
    """Workspace for data isolation. Defaults to empty string if WORKSPACE environment variable is not set."""

    role: str = field(default="ingestion", init=False)
    """Which PostgreSQL pool and vector flush lock this instance's storages use."""

    # Entity extraction
    # ---

    entity_extract_max_gleaning: int = field(
        default=settings.max_gleaning
    )
    """Maximum number of entity extraction attempts for ambiguous content."""

    entity_extract_max_records: int = field(
        default=settings.max_extraction_records
    )
    """Per-response cap on total entity+relationship rows/records."""

    entity_extract_max_entities: int = field(
        default=settings.max_extraction_entities
    )
    """Per-response cap on entity rows/objects."""

    force_llm_summary_on_merge: int = field(
        default=settings.force_llm_summary_on_merge
    )

    # Text chunking
    # ---

    tokenizer: Optional[Tokenizer] = field(default=None)
    """
    Token counter for every token budget; None uses the shared :func:`get_tokenizer`.
    """

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""

    embedding_token_limit: int | None = field(default=None, init=False)
    """Token limit for embedding model. Set automatically from embedding_func.max_token_size in __post_init__."""

    embedding_batch_num: int = field(default=settings.embedding_batch_num)
    """Batch size for embedding computations."""

    embedding_func_max_async: int = field(
        default=settings.embedding_func_max_async
    )
    """Maximum number of concurrent embedding function calls."""

    default_embedding_timeout: float = field(
        default=settings.embedding_timeout
    )

    summary_max_tokens: int = field(
        default=settings.summary_max_tokens
    )
    """Maximum tokens allowed for entity/relation description."""

    summary_context_size: int = field(
        default=settings.summary_context_size
    )
    """Maximum number of tokens allowed per LLM response."""

    summary_length_recommended: int = field(
        default=int(
            settings.summary_length_recommended
        )
    )
    """Recommended length of LLM summary output."""

    llm_model_max_async: int = field(
        default=int(
            settings.extract_llm_max_async
        )
    )
    """Maximum number of concurrent LLM calls."""

    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""

    enable_llm_cache: bool = field(default=True)
    """Enables caching for LLM responses to avoid redundant computations."""

    enable_llm_cache_for_entity_extract: bool = field(default=True)
    """If True, enables caching for entity extraction steps to reduce LLM costs."""

    max_source_ids_per_entity: int = field(
        default=settings.max_source_ids_per_entity
    )
    """Maximum number of source (chunk) ids in entity Graph + VDB."""

    max_source_ids_per_relation: int = field(
        default=settings.max_source_ids_per_relation
    )
    """Maximum number of source (chunk) ids in relation Graph + VDB."""

    source_ids_limit_method: str = field(
        default_factory=lambda: normalize_source_ids_limit_method(
            settings.source_ids_limit_method
        )
    )
    """Strategy for enforcing source_id limits: KEEP (oldest stay) or FIFO."""

    max_file_paths: int = field(
        default=settings.max_file_paths
    )
    """Maximum number of file paths to store in entity/relation file_path field."""

    cosine_better_than_threshold: float = field(
        default=settings.cosine_threshold
    )

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)

    # The multimodal processors are deliberately NOT dataclass fields:
    # ``_build_global_config`` runs ``asdict(self)``, which deep-copies every
    # field, and they hold live storage handles. They are plain attributes set
    # in ``__post_init__``, exactly like the storages.

    def _build_global_config(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self):
        from hybrid_rag.storage.shared import (
            initialize_share_data,
        )

        initialize_share_data()

        # Ensure vector_db_storage_cls_kwargs has required fields
        self.vector_db_storage_cls_kwargs = {
            "cosine_better_than_threshold": self.cosine_better_than_threshold,
            **self.vector_db_storage_cls_kwargs,
        }

        self.tokenizer = self.tokenizer or get_tokenizer()

        # Validate config
        if self.force_llm_summary_on_merge < 3:
            logger.warning(
                f"force_llm_summary_on_merge should be at least 3, got {self.force_llm_summary_on_merge}"
            )
        if self.summary_length_recommended > self.summary_max_tokens:
            logger.warning(
                f"max_total_tokens({self.summary_max_tokens}) should greater than summary_length_recommended({self.summary_length_recommended})"
            )

        # Init Embedding
        # Step 1: Capture embedding_func and max_token_size before applying rate_limit decorator
        original_embedding_func = self.embedding_func
        embedding_max_token_size = None
        if self.embedding_func and hasattr(self.embedding_func, "max_token_size"):
            embedding_max_token_size = self.embedding_func.max_token_size
            logger.debug(
                f"Captured embedding max_token_size: {embedding_max_token_size}"
            )
        self.embedding_token_limit = embedding_max_token_size

        # Fix global_config now
        global_config = self._build_global_config()
        # Restore original EmbeddingFunc object (asdict converts it to dict)
        global_config["embedding_func"] = original_embedding_func

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"IngestionRAG init with param:\n  {_print_config}\n")

        # The multimodal mixins log through ``self.logger``; bind the module
        # logger so they and the pipeline write to the same place.
        self.logger = logger

        # Multimodal runtime state — see the note on the field block above for
        # why these are attributes rather than dataclass fields. Populated by
        # ``_initialize_multimodal`` once the storages are up.
        self.modal_processors: dict[str, Any] = {}

        # Step 2: Bound concurrency on a NEW EmbeddingFunc so the caller's
        # object (possibly shared across instances) is never mutated.
        if self.embedding_func is not None:
            wrapped_func = limit_async_func_call(
                self.embedding_func_max_async,
                # Twice the provider timeout: room for the client's own retry.
                timeout=self.default_embedding_timeout * 2,
            )(self.embedding_func.func)
            # Use dataclasses.replace() to create a new instance, leaving the original unchanged
            self.embedding_func = replace(self.embedding_func, func=wrapped_func)

        # Initialize all storages
        self.key_string_value_json_storage_cls = partial(PGKVStorage, global_config=global_config)
        self.vector_db_storage_cls = partial(PGVectorStorage, global_config=global_config)
        self.graph_storage_cls = partial(Neo4JStorage, global_config=global_config)

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
            workspace=self.workspace,
            global_config=global_config,
            embedding_func=self.embedding_func,
        )

        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_DOCS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.full_entities: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_ENTITIES,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.full_relations: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_RELATIONS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.entity_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.relation_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_RELATION_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_ENTITIES,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = PGDocStatusStorage(
            namespace=NameSpace.DOC_STATUS,
            workspace=self.workspace,
            global_config=global_config,
            embedding_func=None,
        )

        self._storages_status = StoragesStatus.CREATED

    async def initialize_storages(self):
        """Storage initialization must be called one by one to prevent deadlock"""
        if self._storages_status == StoragesStatus.CREATED:
            # Set the first initialized workspace will set the default workspace
            # Allows namespace operation without specifying workspace for backward compatibility
            default_workspace = get_default_workspace()
            if default_workspace is None:
                set_default_workspace(self.workspace)
            elif default_workspace != self.workspace:
                logger.info(
                    f"Creating ingestion instance with workspace='{self.workspace}' "
                    f"while default workspace is set to '{default_workspace}'"
                )

            # Auto-initialize pipeline_status for this workspace
            from hybrid_rag.storage.shared import initialize_pipeline_status

            await initialize_pipeline_status(workspace=self.workspace)

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.full_entities,
                self.full_relations,
                self.entity_chunks,
                self.relation_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    # logger.debug(f"Initializing storage: {storage}")
                    await storage.initialize()

            await self._initialize_multimodal()

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("All storage types initialized")

    async def _initialize_multimodal(self) -> None:
        """Bring up the multimodal side: its status cache and its processors.

        Split out of :meth:`initialize_storages` only for readability — it runs
        as part of the same startup and depends on the storages above already
        being initialized.
        """
        if not self.modal_processors:
            self._initialize_processors()

    def _initialize_processors(self) -> None:
        """Build one processor per modality."""
        self.modal_processors = {
            "image": ImageModalProcessor(rag=self),
            "table": TableModalProcessor(rag=self),
            "equation": EquationModalProcessor(rag=self),
        }
        logger.info(
            f"Multimodal processors ready: {sorted(self.modal_processors)}"
        )

    async def _ensure_multimodal_ready(self) -> dict[str, Any]:
        """Guard used by the ingest/query entry points.

        Storages are normally brought up by ``initialize_storages()`` at
        construction; this catches the case where an instance is used before
        that happened.
        """
        try:
            if self._storages_status != StoragesStatus.INITIALIZED:
                await self.initialize_storages()
            else:
                await self._initialize_multimodal()
            return {"success": True}
        except Exception as e:
            logger.error(f"Multimodal initialization failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def finalize_storages(self):
        """Asynchronously finalize the storages with improved error handling"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            storages = [
                ("full_docs", self.full_docs),
                ("text_chunks", self.text_chunks),
                ("full_entities", self.full_entities),
                ("full_relations", self.full_relations),
                ("entity_chunks", self.entity_chunks),
                ("relation_chunks", self.relation_chunks),
                ("entities_vdb", self.entities_vdb),
                ("relationships_vdb", self.relationships_vdb),
                ("chunks_vdb", self.chunks_vdb),
                ("chunk_entity_relation_graph", self.chunk_entity_relation_graph),
                ("llm_response_cache", self.llm_response_cache),
                ("doc_status", self.doc_status),
            ]

            # Finalize each storage individually to ensure one failure doesn't prevent others from closing
            successful_finalizations = []
            failed_finalizations = []

            for storage_name, storage in storages:
                if storage:
                    try:
                        await storage.finalize()
                        successful_finalizations.append(storage_name)
                        logger.debug(f"Successfully finalized {storage_name}")
                    except Exception as e:
                        error_msg = f"Failed to finalize {storage_name}: {e}"
                        logger.error(error_msg)
                        failed_finalizations.append(storage_name)

            # Log summary of finalization results
            if successful_finalizations:
                logger.info(
                    f"Successfully finalized {len(successful_finalizations)} storages"
                )

            if failed_finalizations:
                logger.error(
                    f"Failed to finalize {len(failed_finalizations)} storages: {', '.join(failed_finalizations)}"
                )
            else:
                logger.debug("All storages finalized successfully")

            self._storages_status = StoragesStatus.FINALIZED

    async def _process_extract_entities(
        self,
        chunk: dict[str, Any],
        pipeline_status=None,
        pipeline_status_lock=None,
        ontology: str | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list:
        try:
            chunk_results = await extract_entities(
                chunk,
                # _build_global_config() is per-RAG-instance; the extraction contract
                # is per-document, so it rides along as an override rather than a field.
                global_config={**self._build_global_config(), "kg_ontology": ontology},
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
                text_chunks_storage=self.text_chunks,
                semaphore=semaphore,
            )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            raise e

    def _index_storages(self) -> list:
        """All storages flushed together by index_done_callback / abort."""
        return [
            storage_inst
            for storage_inst in [
                self.full_docs,
                self.doc_status,
                self.text_chunks,
                self.full_entities,
                self.full_relations,
                self.entity_chunks,
                self.relation_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]

    async def _discard_pending_index_ops(self) -> None:
        """Drop not-yet-flushed buffers on an aborting batch.

        Every still-buffered KG/vector record belongs to a document that will
        be marked FAILED and reprocessed, so dropping the shared buffers is
        safe and stops a poisoned record from being re-flushed by the next
        batch.

        ``full_docs`` and ``doc_status`` are skipped: the concurrent
        ``apipeline_enqueue_documents`` path writes them (``full_docs.upsert ->
        index_done_callback -> doc_status.upsert``) without the lock this
        cleanup holds, so clearing ``full_docs`` mid-window would leave a
        PENDING document with no body. Those writes self-flush, so skipping
        them discards nothing processing-owned.

        The LLM cache is flushed before its buffer is dropped: re-running LLM
        calls is expensive, so whatever can still be written is kept, and only
        an un-writable entry is dropped.

        Best-effort: a flush/clear failure is logged, not raised, so cleanup
        never masks the original abort cause.
        """
        for storage_inst in self._index_storages():
            if storage_inst is self.full_docs or storage_inst is self.doc_status:
                continue
            if storage_inst is self.llm_response_cache:
                try:
                    await cast(StorageNameSpace, storage_inst).index_done_callback()
                except Exception as e:
                    logger.error(f"Failed to persist LLM cache on abort: {e}")
            try:
                await cast(StorageNameSpace, storage_inst).drop_pending_index_ops()
            except Exception as e:
                logger.error(
                    f"Failed to discard pending ops on "
                    f"{type(storage_inst).__name__}: {e}"
                )

    async def _insert_done(self) -> None:
        storages = self._index_storages()

        async def _flush_one(storage_inst) -> None:
            # Wrap each flush so a failure carries the driver name + namespace.
            # The pipeline uses this to abort the batch with an actionable
            # reason instead of misattributing a shared-buffer flush error to
            # whichever document happened to trigger index_done_callback.
            try:
                await cast(StorageNameSpace, storage_inst).index_done_callback()
            except Exception as e:
                namespace = getattr(storage_inst, "final_namespace", None) or getattr(
                    storage_inst, "namespace", ""
                )
                raise IndexFlushError(type(storage_inst).__name__, namespace, e) from e

        # Await every flush to completion (return_exceptions=True) before
        # raising. With the default gather, the first IndexFlushError is
        # propagated while sibling flush coroutines keep running detached —
        # they could commit records or race _discard_pending_index_ops after
        # the abort decision, and a second failing sibling would surface as a
        # "Task exception was never retrieved" warning. Collecting all results
        # first makes teardown deterministic and lets us report every failure.
        results = await asyncio.gather(
            *[_flush_one(inst) for inst in storages], return_exceptions=True
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            # A cooperative cancellation must propagate as-is, not be reported
            # as a storage flush failure (_flush_one's `except Exception` does
            # not catch CancelledError, so it lands here as a result).
            for exc in errors:
                if isinstance(exc, asyncio.CancelledError):
                    raise exc
            for extra in errors[1:]:
                logger.error(f"Additional index flush failure: {extra}")
            raise errors[0]

        log_message = "In memory DB persist to disk"
        logger.info(log_message)


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

