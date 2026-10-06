"""Process-wide state the storage backends and the pipeline share.

Everything here is scoped to one process — the app runs a single worker, so
plain ``asyncio`` primitives are enough:

* ``initialize_share_data`` / ``finalize_share_data`` — bring the state up and
  tear it down; initialization is idempotent.
* ``get_data_init_lock`` — serializes one-time table/index bootstrap.
* ``get_storage_keyed_lock`` — locks a set of keys (entity / relation names)
  so two documents never merge the same graph element at once.
* ``get_namespace_lock`` — a reusable lock per ``workspace``/``namespace``.
* ``get_namespace_data`` / ``initialize_pipeline_status`` — the per-workspace
  ``pipeline_status`` dict the worker uses to signal cancellation.
* ``get_default_workspace`` / ``set_default_workspace`` — the workspace the first
  instance claimed, so namespace helpers called without one still resolve.
"""

import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import logging
from hybrid_rag.core.exceptions import PipelineNotInitializedError

logger = logging.getLogger(__name__)

_initialized = False
_data_init_lock: Optional[asyncio.Lock] = None
_internal_lock: Optional[asyncio.Lock] = None
_shared_dicts: dict[str, dict[str, Any]] = {}
_namespace_locks: dict[str, asyncio.Lock] = {}
_default_workspace: Optional[str] = None


class _KeyedLocks:
    """One ``asyncio.Lock`` per key, dropped once no holder or waiter refers to it."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._refs: Counter[str] = Counter()

    @asynccontextmanager
    async def hold(self, namespace: str, keys: list[str]) -> AsyncIterator[None]:
        # Sorted, so two callers locking overlapping key sets cannot deadlock.
        combined = [f"{namespace}:{key}" for key in sorted(keys)]
        for key in combined:
            self._refs[key] += 1
            self._locks.setdefault(key, asyncio.Lock())
        acquired: list[str] = []
        try:
            for key in combined:
                await self._locks[key].acquire()
                acquired.append(key)
            yield
        finally:
            for key in reversed(acquired):
                self._locks[key].release()
            for key in combined:
                self._refs[key] -= 1
                if not self._refs[key]:
                    del self._refs[key]
                    self._locks.pop(key, None)


_keyed_locks: Optional[_KeyedLocks] = None


def initialize_share_data() -> None:
    """Create the process-wide state. Safe to call more than once."""
    global _initialized, _data_init_lock, _internal_lock, _keyed_locks
    if _initialized:
        return
    _data_init_lock = asyncio.Lock()
    _internal_lock = asyncio.Lock()
    _keyed_locks = _KeyedLocks()
    _shared_dicts.clear()
    _namespace_locks.clear()
    _initialized = True
    logger.debug("[shared] process-local state initialized")


def finalize_share_data() -> None:
    """Drop all process-wide state; the next ``initialize_share_data`` starts fresh."""
    global _initialized, _data_init_lock, _internal_lock, _keyed_locks, _default_workspace
    _data_init_lock = _internal_lock = _keyed_locks = None
    _shared_dicts.clear()
    _namespace_locks.clear()
    _default_workspace = None
    _initialized = False


def _require(value):
    if value is None:
        raise RuntimeError(
            "Shared data not initialized. Call initialize_share_data() before using locks!"
        )
    return value


def get_data_init_lock() -> asyncio.Lock:
    """The lock guarding one-time table/index bootstrap."""
    return _require(_data_init_lock)


def get_storage_keyed_lock(keys: str | list[str], namespace: str = "default"):
    """Context manager holding every key in ``keys`` within ``namespace``."""
    return _require(_keyed_locks).hold(namespace, [keys] if isinstance(keys, str) else keys)


def get_final_namespace(namespace: str, workspace: str | None = None) -> str:
    """``workspace:namespace``, falling back to the default workspace."""
    if workspace is None:
        workspace = _default_workspace
    if workspace is None:
        raise ValueError("Invoke namespace operation without workspace")
    return f"{workspace}:{namespace}" if workspace else namespace


def get_namespace_lock(namespace: str, workspace: str | None = None) -> asyncio.Lock:
    """A reusable lock scoped to one ``workspace``/``namespace`` pair."""
    key = get_final_namespace(namespace, workspace)
    return _namespace_locks.setdefault(key, asyncio.Lock())


async def get_namespace_data(
    namespace: str, first_init: bool = False, workspace: str | None = None
) -> dict[str, Any]:
    """The shared dict for ``namespace``, created on first use.

    ``pipeline_status`` is only created through :func:`initialize_pipeline_status`
    (``first_init=True``); reading it earlier is a programming error.
    """
    final_namespace = get_final_namespace(namespace, workspace)
    async with _require(_internal_lock):
        if final_namespace not in _shared_dicts:
            if final_namespace.rpartition(":")[2] == "pipeline_status" and not first_init:
                raise PipelineNotInitializedError(final_namespace)
            _shared_dicts[final_namespace] = {}
    return _shared_dicts[final_namespace]


async def initialize_pipeline_status(workspace: str | None = None) -> None:
    """Create ``workspace``'s ``pipeline_status`` dict; idempotent."""
    await get_namespace_data("pipeline_status", first_init=True, workspace=workspace)


def set_default_workspace(workspace: str | None = None) -> None:
    """Record the workspace namespace helpers should assume when given none."""
    global _default_workspace
    _default_workspace = workspace or ""


def get_default_workspace() -> Optional[str]:
    """The default workspace, or ``None`` when no instance has claimed one yet."""
    return _default_workspace
