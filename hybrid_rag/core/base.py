import re
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    TypedDict,
)
from .embedding import EmbeddingFunc
from .types import KnowledgeGraph

def collection_suffix(model_name: str | None, embedding_dim: int) -> str | None:
    """Vector table suffix for one embedding model, e.g. ``"text_embedding_3_large_3072d"``.

    Each model and dimension gets its own tables, so switching models never mixes
    vectors. None when the model has no name.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    safe_model_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.strip().lower())
    return f"{safe_model_name}_{embedding_dim}d"


class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int



@dataclass
class StorageNameSpace(ABC):
    namespace: str
    workspace: str
    global_config: dict[str, Any]

    async def initialize(self):
        """Initialize the storage"""
        pass

    async def finalize(self):
        """Finalize the storage"""
        pass

    @abstractmethod
    async def index_done_callback(self) -> None:
        """Commit the storage operations after indexing"""

    async def drop_pending_index_ops(self) -> None:
        """Discard any not-yet-flushed buffered index ops.

        Backends that defer writes to ``index_done_callback`` (via an
        in-memory ``_pending_*`` buffer) override this to clear that buffer.
        The pipeline calls it when a batch is aborting on an internal error:
        every still-buffered record belongs to a document that is being
        marked FAILED and fully reprocessed on the next run, so dropping the
        buffer is safe and prevents the poisoned/stale records from being
        re-flushed by the remaining in-flight documents or carried over to
        the next batch. Immediate-write backends keep the default no-op.
        """
        return None

@dataclass
class BaseVectorStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc
    cosine_better_than_threshold: float = field(default=0.2)

    def _validate_embedding_func(self):
        """Validate that embedding_func is provided.

        This method should be called at the beginning of __post_init__
        in all vector storage implementations.

        Raises:
            ValueError: If embedding_func is None
        """
        if self.embedding_func is None:
            raise ValueError(
                "embedding_func is required for vector storage. "
                "Please provide a valid EmbeddingFunc instance."
            )

    def _generate_collection_suffix(self) -> str | None:
        """Generates collection/table suffix from embedding_func.

        Return suffix if model_name exists in embedding_func, otherwise return None.
        Note: embedding_func is guaranteed to exist (validated in __post_init__).

        Returns:
            str | None: Suffix string e.g. "text_embedding_3_large_3072d", or None if model_name not available
        """
        return collection_suffix(
            getattr(self.embedding_func, "model_name", None),
            self.embedding_func.embedding_dim,
        )

    @abstractmethod
    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` rows closest to ``query``.

        Args:
            query: The query string to search for
            top_k: Number of top results to return
            query_embedding: Optional pre-computed embedding for the query;
                skips the embedding call when given.
        """

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vectors for insert/update; written by ``index_done_callback``."""

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        pass

    @abstractmethod
    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """

    @abstractmethod
    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        pass


@dataclass
class BaseKVStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get value by id"""

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get values by ids"""

    @abstractmethod
    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return un-exist keys"""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert data
        """

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """

@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    """All operations related to edges in graph should be undirected."""

    embedding_func: EmbeddingFunc

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The ID of the node to check

        Returns:
            True if the node exists, False otherwise
        """

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            True if the edge exists, False otherwise
        """

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its ID, returning only node properties.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            A dictionary of node properties if found, None otherwise
        """

    @abstractmethod
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            A dictionary of edge properties if found, None otherwise
        """

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node.

        Args:
            source_node_id: The ID of the node to get edges for

        Returns:
            A list of (source_id, target_id) tuples representing edges,
            or None if the node doesn't exist
        """

    @abstractmethod
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Properties of the nodes that exist, keyed by node ID."""

    @abstractmethod
    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Degree of every node in ``node_ids`` (0 when missing)."""

    @abstractmethod
    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Sum of the source and target degrees of every edge pair."""

    @abstractmethod
    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Properties of the edges that exist, keyed by ``(src, tgt)``."""

    @abstractmethod
    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Edges of every node in ``node_ids`` as ``(source, target)`` tuples."""

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert a new node or update an existing node in the graph.

        Args:
            node_id: The ID of the node to insert or update
            node_data: A dictionary of node properties
        """

    @abstractmethod
    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        """Insert or update ``(node_id, node_data)`` pairs in one call."""

    @abstractmethod
    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        """Insert or update ``(source_id, target_id, edge_data)`` triples in one call."""

    @abstractmethod
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert a new edge or update an existing edge in the graph.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node
            edge_data: A dictionary of edge properties
        """

    @abstractmethod
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node IDs to be deleted
        """

    @abstractmethod
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """

    @abstractmethod
    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000, workspace: str = None
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label(entity name) of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return, Defaults to 1000（BFS if possible)
            workspace: Optional workspace filter to restrict graph retrieval

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """

    @abstractmethod
    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
            (Edge is bidirectional for some storage implementation; deduplication must be handled by the caller)
        """

    @abstractmethod
    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """

    @abstractmethod
    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels(entity names) by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first)
        """

    @abstractmethod
    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels(entity names) with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance
        """


class DocStatus(str, Enum):
    """Document processing status.
    Pipeline order: PENDING -> PARSING -> PROCESSING -> PROCESSED | FAILED.
    ANALYZING and PREPROCESSED are no longer written; they stay so rows left in
    them by older deployments still load and get reset.
    """

    PENDING = "pending"
    PARSING = "parsing"  # Phase 1: hand the stored content back (no parser runs)
    ANALYZING = "analyzing"  # Deprecated: multimodal analysis now runs in PROCESSING
    PROCESSING = "processing"  # Phase 2: chunking, multimodal analysis, extraction
    PREPROCESSED = "preprocessed"  # Deprecated
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class DocProcessingStatus:
    """Document processing status data structure"""

    content_summary: str
    """First 100 chars of document content, used for preview"""
    content_length: int
    """Total length of document"""
    file_path: str
    """Canonical basename of the document.

    Always a hint-stripped basename (e.g. ``abc.docx``) or the literal
    ``"unknown_source"`` sentinel; never carries directory components or
    parser ``[hint]`` segments. UI display, filename-based dedup, and
    citation paths all share this value.
    """
    status: DocStatus
    """Current processing status"""
    created_at: str
    """ISO format timestamp when document was created"""
    updated_at: str
    """ISO format timestamp when document was last updated"""
    track_id: str | None = None
    """Tracking ID for monitoring progress"""
    chunks_count: int | None = None
    """Number of chunks after splitting, used for processing"""
    chunks_list: list[str] | None = field(default_factory=list)
    """List of chunk IDs associated with this document, used for deletion"""
    error_msg: str | None = None
    """Error message if failed"""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""
    multimodal_processed: bool | None = field(default=None, repr=False)
    """Internal field: indicates if multimodal processing is complete. Not shown in repr() but accessible for debugging."""
    content_hash: str | None = None
    """MD5 hash of the underlying document content (raw text or source file).

    Used together with file_path basename for duplicate detection. Empty for
    pending_parse records whose content has not been extracted yet.
    """

    def __post_init__(self):
        """
        Handle status conversion based on multimodal_processed field.

        Business rules:
        - If multimodal_processed is False and status is PROCESSED,
          then change status to PREPROCESSED
        - The multimodal_processed field is kept (with repr=False) for internal use and debugging
        """
        # Apply status conversion logic
        if self.multimodal_processed is not None:
            if (
                self.multimodal_processed is False
                and self.status == DocStatus.PROCESSED
            ):
                self.status = DocStatus.PREPROCESSED


@dataclass
class DocStatusStorage(BaseKVStorage, ABC):
    """Base class for document status storage"""

    @abstractmethod
    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching any of the given statuses"""

    @abstractmethod
    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Get document by canonical file basename.

        Used for filename-based deduplication. Callers must pass the canonical
        basename; storage implementations only compare against the canonical
        ``file_path`` persisted by the business layer.

        Args:
            basename: The filename basename to search for (e.g. "report.pdf").

        Returns:
            (doc_id, doc_data) when a matching record exists, otherwise None.
        """

    @abstractmethod
    async def get_doc_by_content_hash(
        self, content_hash: str, exclude_doc_id: str | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        """Get the oldest document with this content_hash field.

        Used for content-hash deduplication of full documents.

        Args:
            content_hash: The content hash value to search for.
            exclude_doc_id: Skip this document, so a document already enqueued
                can ask whether *another* one holds the same content.

        Returns:
            (doc_id, doc_data) when a matching record exists, otherwise None.
        """


class StoragesStatus(str, Enum):
    """Storages status"""

    NOT_CREATED = "not_created"
    CREATED = "created"
    INITIALIZED = "initialized"
    FINALIZED = "finalized"


@dataclass
class DeletionResult:
    """Represents the result of a deletion operation."""

    status: Literal["success", "not_found", "fail"]
    doc_id: str
    message: str
    status_code: int = 200
    file_path: str | None = None