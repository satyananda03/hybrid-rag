"""The write pool: one :class:`IngestionRAG` per workspace.

Used by ingest and delete. Pooling lives in
:mod:`hybrid_rag.workspace_pool`; this module only says how to build an instance.
"""

from hybrid_rag.ingestion.factory import build_rag
from hybrid_rag.ingestion.rag import IngestionRAG
from hybrid_rag.workspace_pool import WorkspacePool

workspace_pool: WorkspacePool[IngestionRAG] = WorkspacePool(build_rag)
