"""One RAG instance per workspace, cached — shared by the read and the write pool.

A workspace is the isolation boundary: it is forwarded to every storage, so two
workspaces share the database but never each other's documents, chunks, graph
nodes or pipeline state. Building an instance is expensive (storage handles,
model wrappers), so instances are pooled and reused.

Instances are handed out as leases (``async with pool.lease(ws) as rag``). The
pool is bounded and evicts the least-recently-used workspace, finalizing its
storages — but never one that is leased: finalizing an instance mid-ingest or
mid-query would fail that work. When every instance is leased the pool runs over
capacity for a while and shrinks back as the leases end.

The pools themselves live next to their builders:
``hybrid_rag.ingestion.workspace`` (write) and
``hybrid_rag.retrieval.workspace`` (read).
"""

import asyncio
import re
from collections import Counter, OrderedDict
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
    """LRU cache of ready-to-use instances, keyed by (sanitized) workspace."""

    def __init__(self, build: Callable[[str], Awaitable[T]], max_workspaces: int) -> None:
        self._build = build
        self._max = max_workspaces
        self._entries: "OrderedDict[str, T]" = OrderedDict()
        self._leases: Counter[str] = Counter()
        # One lock per workspace so building instance A never blocks a request
        # for instance B; _global_lock only guards the bookkeeping above.
        self._build_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        # Evicted instances are finalized in the background, as before, so the
        # request that triggered the eviction does not wait for it.
        self._finalizing: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lease(self, workspace: str) -> AsyncIterator[T]:
        """The instance for ``workspace``, protected from eviction while held."""
        ws = sanitize_workspace(workspace)
        instance = await self._checkout(ws)
        try:
            yield instance
        finally:
            async with self._global_lock:
                self._leases[ws] -= 1
                if not self._leases[ws]:
                    del self._leases[ws]
                evicted = self._evict_over_capacity()
            self._finalize_in_background(evicted)

    async def warm(self, workspace: str) -> None:
        """Build and cache ``workspace`` ahead of its first request."""
        async with self.lease(workspace):
            pass

    async def shutdown(self) -> None:
        async with self._global_lock:
            instances = list(self._entries.items())
            self._entries.clear()
            self._leases.clear()
            self._build_locks.clear()
        await asyncio.gather(*self._finalizing, return_exceptions=True)
        await self._finalize(instances)

    async def _checkout(self, ws: str) -> T:
        """Fetch or build the instance and count the lease in one locked step,
        so it can never be evicted between the two."""
        instance = await self._lease_cached(ws)
        if instance is not None:
            return instance

        async with self._global_lock:
            build_lock = self._build_locks.setdefault(ws, asyncio.Lock())
        async with build_lock:
            # Someone else may have built it while we waited for the lock.
            instance = await self._lease_cached(ws)
            if instance is not None:
                return instance

            logger.info(f"[WorkspacePool] initializing instance for workspace: {ws}")
            instance = await self._build(ws)
            async with self._global_lock:
                self._entries[ws] = instance
                self._leases[ws] += 1
                evicted = self._evict_over_capacity()
        self._finalize_in_background(evicted)
        return instance

    async def _lease_cached(self, ws: str) -> T | None:
        async with self._global_lock:
            instance = self._entries.get(ws)
            if instance is not None:
                self._entries.move_to_end(ws)
                self._leases[ws] += 1
            return instance

    def _evict_over_capacity(self) -> list[tuple[str, T]]:
        """Drop least-recently-used, unleased workspaces until back at capacity.

        Called with ``_global_lock`` held; the caller finalizes the returned
        instances outside it.
        """
        evicted: list[tuple[str, T]] = []
        idle = [ws for ws in self._entries if not self._leases[ws]]  # LRU first
        while len(self._entries) > self._max and idle:
            victim = idle.pop(0)
            evicted.append((victim, self._entries.pop(victim)))
            self._build_locks.pop(victim, None)
        return evicted

    def _finalize_in_background(self, instances: list[tuple[str, T]]) -> None:
        if instances:
            task = asyncio.create_task(self._finalize(instances))
            self._finalizing.add(task)
            task.add_done_callback(self._finalizing.discard)

    async def _finalize(self, instances: list[tuple[str, T]]) -> None:
        for ws, instance in instances:
            logger.info(f"[WorkspacePool] finalizing workspace: {ws}")
            try:
                await instance.finalize_storages()
            except Exception:
                logger.error(f"[WorkspacePool] error finalizing {ws}", exc_info=True)
