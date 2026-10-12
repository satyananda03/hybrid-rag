"""One RAG instance per workspace, cached — shared by the read and the write pool.

A workspace is the isolation boundary: it is forwarded to every storage, so two
workspaces share the database but never each other's documents, chunks, graph
nodes or pipeline state. Building an instance is expensive (storage handles,
model wrappers), so one is built per workspace on first use and kept until
:meth:`WorkspacePool.shutdown`. An instance is only in-memory objects: the
database connections and the Neo4j driver are shared by every workspace.

The pools themselves live next to their builders:
``hybrid_rag.ingestion.workspace`` (write) and
``hybrid_rag.retrieval.workspace`` (read).
"""

import asyncio
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Generic, Protocol, TypeVar

import logging

logger = logging.getLogger(__name__)


_UNSAFE_WORKSPACE_CHARS = re.compile(r"[^a-zA-Z0-9_\-]")
DEFAULT_WORKSPACE_NAME = "default_workspace"


def sanitize_workspace(ws_name: str) -> str:
    """Reduce a workspace name to what the storage layer accepts.

    The name keys every storage row and graph node, so anything outside
    ``[A-Za-z0-9_-]`` is replaced. An empty result falls back to the default
    name rather than producing an unnamed — and therefore shared — workspace.
    """
    if not ws_name:
        return DEFAULT_WORKSPACE_NAME
    cleaned = _UNSAFE_WORKSPACE_CHARS.sub("_", ws_name.strip())
    return cleaned.strip("_-") or DEFAULT_WORKSPACE_NAME


class _Finalizable(Protocol):
    async def finalize_storages(self) -> None: ...


T = TypeVar("T", bound=_Finalizable)


class WorkspacePool(Generic[T]):
    """Ready-to-use instances, built once per (sanitized) workspace."""

    def __init__(self, build: Callable[[str], Awaitable[T]]) -> None:
        self._build = build
        self._entries: dict[str, T] = {}
        # One lock per workspace so building instance A never blocks a request
        # for instance B.
        self._build_locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def lease(self, workspace: str) -> AsyncIterator[T]:
        """The instance for ``workspace``, built on first use."""
        yield await self._get(sanitize_workspace(workspace))

    async def warm(self, workspace: str) -> None:
        """Build and cache ``workspace`` ahead of its first request."""
        await self._get(sanitize_workspace(workspace))

    async def shutdown(self) -> None:
        instances = list(self._entries.items())
        self._entries.clear()
        self._build_locks.clear()
        for ws, instance in instances:
            logger.info(f"[WorkspacePool] finalizing workspace: {ws}")
            try:
                await instance.finalize_storages()
            except Exception:
                logger.error(f"[WorkspacePool] error finalizing {ws}", exc_info=True)

    async def _get(self, ws: str) -> T:
        instance = self._entries.get(ws)
        if instance is not None:
            return instance
        async with self._build_locks.setdefault(ws, asyncio.Lock()):
            # Someone else may have built it while we waited for the lock.
            instance = self._entries.get(ws)
            if instance is None:
                logger.info(f"[WorkspacePool] initializing instance for workspace: {ws}")
                instance = self._entries[ws] = await self._build(ws)
        return instance
