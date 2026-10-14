"""The retriever: one class owning the read-side storages.

``RetrievalRAG`` opens the PostgreSQL and Neo4j handles for one workspace, answers
:meth:`aretrieve`, and closes them again. It shares the storage classes with
``hybrid_rag.ingestion`` but only ever reads — nothing here inserts or deletes.
"""

from dataclasses import asdict, dataclass, field, replace
from functools import partial
from typing import Any, Callable, final, Optional
from hybrid_rag.core.concurrency import limit_async_func_call
from hybrid_rag.storage.neo4j import Neo4JStorage
from hybrid_rag.storage.postgres import PGKVStorage, PGVectorStorage
from hybrid_rag.core.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, StoragesStatus
from hybrid_rag.retrieval.types import QueryParam
from hybrid_rag.core.constants import NameSpace
from hybrid_rag.retrieval.query import kg_query
from hybrid_rag.core.tokenizer import Tokenizer, get_tokenizer
from hybrid_rag.core.embedding import EmbeddingFunc
from hybrid_rag.config import settings
from hybrid_rag.core.types import KnowledgeGraph
import logging

logger = logging.getLogger(__name__)

@final
@dataclass
class RetrievalRAG:
    """Graph + vector retrieval over PostgreSQL and Neo4j.

    Read-only: the storages are opened, queried through :meth:`aretrieve`, and
    closed. Ingestion and deletion live in ``hybrid_rag.ingestion``, which writes
    the same tables; nothing here creates or removes a document.

    Data isolation is per :attr:`workspace`: every storage instance below is
    constructed with it, so two ``RetrievalRAG`` objects with different workspaces
    share a database but never each other's rows, nodes, or pipeline state.
    """

    workspace: str = field(default="")
    """Workspace for data isolation; every storage below is scoped to it."""

    role: str = field(default="retrieval", init=False)
    """Which PostgreSQL pool and vector flush lock this instance's storages use."""

    # Context budget (read from ``global_config`` by the retrieval stages)
    # ---

    max_entity_tokens: int = field(default=settings.max_entity_tokens)
    """Maximum number of tokens for entity in context."""

    max_relation_tokens: int = field(default=settings.max_relation_tokens)
    """Maximum number of tokens for relation in context."""

    related_chunk_number: int = field(default=settings.related_chunk_number)
    """Number of related chunks to grab from single entity or relation."""

    kg_chunk_pick_method: str = field(default=settings.kg_chunk_pick_method)
    """'VECTOR' picks entity/relation chunks by embedding similarity, 'WEIGHT' by frequency."""

    enable_content_headings: bool = field(default_factory=lambda: settings.enable_content_headings)
    """Append each chunk's parent heading path as a `content_headings` field in the chunk JSON sent to the LLM."""

    max_graph_nodes: int = field(default=settings.max_graph_nodes)
    """Maximum number of graph nodes returned by the knowledge-graph endpoints."""

    # Models
    # ---

    tokenizer: Optional[Tokenizer] = field(default=None)
    """Token counter for the context budget; None uses the shared :func:`get_tokenizer`."""

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""

    embedding_batch_num: int = field(default=settings.embedding_batch_num)
    """Batch size for embedding computations."""

    embedding_func_max_async: int = field(default=settings.embedding_func_max_async)
    """Maximum number of concurrent embedding function calls."""

    default_embedding_timeout: float = field(default=settings.embedding_timeout)
    """Per-call embedding timeout in seconds."""

    rerank_model_func: Callable[..., object] | None = field(default=None)
    """Reranks retrieved chunks; carries its own model, host and key. None disables reranking."""

    rerank_model_max_async: int = field(default=settings.rerank_max_async)
    """Maximum number of concurrent rerank calls."""

    default_rerank_timeout: float = field(default=settings.rerank_timeout)
    """Rerank request timeout in seconds."""

    min_rerank_score: float = field(default=settings.min_rerank_score)
    """Minimum rerank score threshold for filtering chunks after reranking."""

    # Storage

    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""

    enable_llm_cache: bool = field(default=True)
    """Cache keyword-extraction responses."""

    cosine_better_than_threshold: float = field(default=settings.cosine_threshold)
    """Vector search drops rows whose cosine similarity is below this."""

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)

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


        if self.rerank_model_func is not None:
            self.rerank_model_func = limit_async_func_call(
                self.rerank_model_max_async,
                self.default_rerank_timeout,
            )(self.rerank_model_func)

        original_embedding_func = self.embedding_func

        # Fix global_config now
        global_config = self._build_global_config()
        # Restore original EmbeddingFunc object (asdict converts it to dict)
        global_config["embedding_func"] = original_embedding_func

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"RetrievalRAG init with param:\n  {_print_config}\n")

        # Some helpers log through ``self.logger``; bind the module logger so
        # everything on this instance writes to the same place.
        self.logger = logger

        # Bound concurrency on a NEW EmbeddingFunc so the caller's object
        # (possibly shared across instances) is never mutated.
        if self.embedding_func is not None:
            wrapped_func = limit_async_func_call(
                self.embedding_func_max_async,
                self.default_embedding_timeout,
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

        self._storages_status = StoragesStatus.CREATED

    async def initialize_storages(self):
        """Storage initialization must be called one by one to prevent deadlock"""
        if self._storages_status == StoragesStatus.CREATED:
            for storage in (
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
            ):
                if storage:
                    # logger.debug(f"Initializing storage: {storage}")
                    await storage.initialize()

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("All storage types initialized")

    async def finalize_storages(self):
        """Asynchronously finalize the storages with improved error handling"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            storages = [
                ("text_chunks", self.text_chunks),
                ("entities_vdb", self.entities_vdb),
                ("relationships_vdb", self.relationships_vdb),
                ("chunks_vdb", self.chunks_vdb),
                ("chunk_entity_relation_graph", self.chunk_entity_relation_graph),
                ("llm_response_cache", self.llm_response_cache),
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

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
        workspace: str = None,
    ) -> KnowledgeGraph:
        """Get knowledge graph for a given label

        Args:
            node_label (str): Label to get knowledge graph for
            max_depth (int): Maximum depth of graph
            max_nodes (int, optional): Maximum number of nodes to return. Defaults to self.max_graph_nodes.
            workspace (str, optional): Workspace to filter graph data.

        Returns:
            KnowledgeGraph: Knowledge graph containing nodes and edges
        """
        # Use self.max_graph_nodes as default if max_nodes is None
        if max_nodes is None:
            max_nodes = self.max_graph_nodes
        else:
            # Limit max_nodes to not exceed self.max_graph_nodes
            max_nodes = min(max_nodes, self.max_graph_nodes)

        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label, max_depth, max_nodes, workspace=workspace
        )

    async def aretrieve(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ) -> dict[str, Any]:
        """Retrieve context for ``query``; never generates an answer.

        Generation lives outside this module — callers assemble their own
        prompt from what comes back here.

        Returns:
            dict: the structured payload ``convert_to_user_format`` builds
            (``status`` / ``message`` / ``data`` / ``metadata``) plus
            ``context``, the assembled context string the retrieval produced.
        """
        logger.debug(f"[aretrieve] Query param: {param}")
        global_config = self._build_global_config()

        try:
            query_result = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                chunks_vdb=self.chunks_vdb,
            )

            await self._query_done()

            if query_result is None:
                logger.info("[aretrieve] Query returned no results.")
                return {
                    "status": "failure",
                    "message": "Query returned no results",
                    "data": {},
                    "metadata": {"failure_reason": "no_results", "mode": "mix"},
                    "context": "",
                }

            result = query_result.raw_data or {}
            result["context"] = query_result.context or ""

            data_section = result.get("data") or {}
            logger.debug(
                "[aretrieve] %d entities, %d relationships, %d chunks",
                len(data_section.get("entities", [])),
                len(data_section.get("relationships", [])),
                len(data_section.get("chunks", [])),
            )
            return result

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "status": "failure",
                "message": f"Retrieval failed: {e}",
                "data": {},
                "metadata": {},
                "context": "",
            }

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()


