"""The read pool: one :class:`RetrievalRAG` per workspace.

Used by retrieval, the agents' tools and the graph endpoints. Pooling, leases and eviction live in
:mod:`hybrid_rag.workspace_pool`; this module only says how to build an instance.
"""

from hybrid_rag.config import settings
from hybrid_rag.retrieval.factory import build_rag
from hybrid_rag.retrieval.rag import RetrievalRAG
from hybrid_rag.workspace_pool import WorkspacePool

workspace_pool: WorkspacePool[RetrievalRAG] = WorkspacePool(build_rag, max_workspaces=settings.max_workspaces)
