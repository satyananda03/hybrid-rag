"""Concurrency primitives the pipeline runs on.

* :func:`limit_async_func_call` — bounds concurrent embedding calls and gives
  each one a timeout.
* :func:`_cooperative_yield` — hands control back inside long synchronous loops
  so storage batches do not starve the event loop.
"""

import asyncio
from functools import wraps


def limit_async_func_call(max_size: int, timeout: float | None = None):
    """Bound concurrent calls to ``max_size``, with a per-call timeout.

    A plain semaphore: FIFO-fair, no priorities — every caller here queues at
    the same priority.
    """
    sem = asyncio.Semaphore(max_size)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with sem:
                return await asyncio.wait_for(func(*args, **kwargs), timeout)

        return wrapper

    return decorator


async def _cooperative_yield(iteration: int, every: int = 64) -> None:
    """Periodically yield control to the event loop during CPU-heavy async loops.

    Call inside long synchronous-style loops to prevent event loop starvation
    in single-worker deployments. Yields every `every` iterations.
    """
    if iteration > 0 and iteration % every == 0:
        await asyncio.sleep(0)
