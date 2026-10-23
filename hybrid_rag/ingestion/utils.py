"""Ingestion-side helpers — used while a document is being written."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Callable, NoReturn, Optional, TYPE_CHECKING

from langchain_core.exceptions import OutputParserException
from openai import LengthFinishReasonError
from pydantic import ValidationError

if TYPE_CHECKING:
    from hybrid_rag.core.base import BaseKVStorage

logger = logging.getLogger(__name__)

# A model reply that breaks its output schema as a whole (invalid JSON, a missing
# or mistyped field, or JSON cut off at max_tokens because the model ran on).
# Asking again can succeed; infrastructure errors are not in here.
INVALID_OUTPUT_ERRORS = (ValidationError, OutputParserException, LengthFinishReasonError)


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


async def update_chunk_cache_list(
    chunk_id: str,
    text_chunks_storage: "BaseKVStorage",
    cache_keys: list[str],
    cache_scenario: str = "batch_update",
) -> None:
    """Update chunk's llm_cache_list with the given cache keys

    Args:
        chunk_id: Chunk identifier
        text_chunks_storage: Text chunks storage instance
        cache_keys: List of cache keys to add to the list
        cache_scenario: Description of the cache scenario for logging
    """
    if not cache_keys:
        return

    try:
        chunk_data = await text_chunks_storage.get_by_id(chunk_id)
        if chunk_data:
            # Ensure llm_cache_list exists
            if "llm_cache_list" not in chunk_data:
                chunk_data["llm_cache_list"] = []

            # Add cache keys to the list if not already present
            existing_keys = set(chunk_data["llm_cache_list"])
            new_keys = [key for key in cache_keys if key not in existing_keys]

            if new_keys:
                chunk_data["llm_cache_list"].extend(new_keys)

                # Update the chunk in storage
                await text_chunks_storage.upsert({chunk_id: chunk_data})
                logger.debug(
                    f"Updated chunk {chunk_id} with {len(new_keys)} cache keys ({cache_scenario})"
                )
    except Exception as e:
        logger.warning(
            f"Failed to update chunk {chunk_id} with cache references on {cache_scenario}: {e}"
        )


def raise_first_failure(failures: BaseExceptionGroup) -> NoReturn:
    """Re-raise the first failure of an ``asyncio.TaskGroup`` as itself.

    Callers match on the concrete error (cancellation, flush errors) and show
    its message as the document's ``error_msg``; the group wrapper would hide
    both. The failure keeps its own ``__cause__``.
    """
    failure = failures.exceptions[0]
    raise failure from failure.__cause__


def generate_track_id(prefix: str = "upload") -> str:
    """Generate a unique tracking ID with timestamp and UUID

    Args:
        prefix: Prefix for the track ID (e.g., 'upload', 'insert')

    Returns:
        str: Unique tracking ID in format: {prefix}_{timestamp}_{uuid}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    return f"{prefix}_{timestamp}_{unique_id}"


async def safe_vdb_operation_with_exception(
    operation: Callable,
    operation_name: str,
    entity_name: str = "",
    max_retries: int = 3,
    retry_delay: float = 0.2,
    logger_func: Optional[Callable] = None,
    timeout_seconds: float | None = None,
    log_start: bool = False,
    success_log_threshold_seconds: float = 10.0,
) -> None:
    """
    Safely execute vector database operations with retry mechanism and exception handling.

    This function ensures that VDB operations are executed with proper error handling
    and retry logic. If all retries fail, it raises an exception to maintain data consistency.

    Args:
        operation: The async operation to execute
        operation_name: Operation name for logging purposes
        entity_name: Entity name for logging purposes
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        logger_func: Logger function to use for error messages
        timeout_seconds: Optional timeout for a single operation attempt
        log_start: Whether to emit start/success logs for each attempt
        success_log_threshold_seconds: Log successful attempts when duration exceeds this threshold

    Raises:
        Exception: When operation fails after all retry attempts
    """
    log_func = logger_func or logger.warning

    for attempt in range(max_retries):
        start_ts = time.perf_counter()
        attempt_label = f"{attempt + 1}/{max_retries}"
        try:
            if log_start:
                logger.info(
                    "VDB %s start for %s (attempt %s, timeout=%s)",
                    operation_name,
                    entity_name or "<unknown>",
                    attempt_label,
                    f"{timeout_seconds:.1f}s"
                    if timeout_seconds is not None
                    else "none",
                )

            if timeout_seconds is not None and timeout_seconds > 0:
                await asyncio.wait_for(operation(), timeout=timeout_seconds)
            else:
                await operation()

            elapsed = time.perf_counter() - start_ts
            if log_start or elapsed >= success_log_threshold_seconds:
                logger.info(
                    "VDB %s success for %s in %.2fs (attempt %s)",
                    operation_name,
                    entity_name or "<unknown>",
                    elapsed,
                    attempt_label,
                )
            return  # Success, return immediately
        except asyncio.TimeoutError as e:
            elapsed = time.perf_counter() - start_ts
            timeout_msg = (
                f"VDB {operation_name} timeout for {entity_name or '<unknown>'} "
                f"after {elapsed:.2f}s (attempt {attempt_label}, timeout={timeout_seconds}s)"
            )
            if attempt >= max_retries - 1:
                log_func(timeout_msg)
                raise TimeoutError(timeout_msg) from e
            log_func(f"{timeout_msg}, retrying...")
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)
        except Exception as e:
            elapsed = time.perf_counter() - start_ts
            if attempt >= max_retries - 1:
                error_msg = (
                    f"VDB {operation_name} failed for {entity_name or '<unknown>'} "
                    f"after {max_retries} attempts in {elapsed:.2f}s: {e}"
                )
                log_func(error_msg)
                raise Exception(error_msg) from e
            else:
                log_func(
                    f"VDB {operation_name} attempt {attempt + 1} failed for "
                    f"{entity_name or '<unknown>'} after {elapsed:.2f}s: {e}, retrying..."
                )
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)


def create_prefixed_exception(original_exception: Exception, prefix: str) -> Exception:
    """
    Safely create a prefixed exception that adapts to all error types.

    Args:
        original_exception: The original exception.
        prefix: The prefix to add.

    Returns:
        A new exception with the prefix, maintaining the original exception type if possible.
    """
    try:
        # Method 1: Try to reconstruct using original arguments.
        if hasattr(original_exception, "args") and original_exception.args:
            args = list(original_exception.args)
            # Find the first string argument and prefix it. This is safer for
            # exceptions like OSError where the first arg is an integer (errno).
            found_str = False
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    args[i] = f"{prefix}: {arg}"
                    found_str = True
                    break

            # If no string argument is found, prefix the first argument's string representation.
            if not found_str:
                args[0] = f"{prefix}: {args[0]}"

            return type(original_exception)(*args)
        else:
            # Method 2: If no args, try single parameter construction.
            return type(original_exception)(f"{prefix}: {str(original_exception)}")
    except Exception:
        # Method 3: If reconstruction fails for any reason, wrap it in a
        # RuntimeError preserving the original type name and message. This is a
        # defensive catch-all: most known failures already surface as TypeError
        # (e.g. json.JSONDecodeError needs (msg, doc, pos) and
        # openai.APIStatusError/BadRequestError need keyword-only
        # (response, body), so rebuilding from args alone raises TypeError), but
        # an exotic constructor could raise something else (KeyError, a
        # validation error, ...). Catching `Exception` guarantees this helper
        # never raises while prefixing — `KeyboardInterrupt`/`SystemExit` are
        # BaseException and still propagate. The original exception and its full
        # traceback are preserved by the caller's `raise ... from original`.
        return RuntimeError(
            f"{prefix}: {type(original_exception).__name__}: {str(original_exception)}"
        )
