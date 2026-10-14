"""Helpers shared by ingestion, retrieval and storage."""

import logging
import re
from hashlib import md5
from typing import Any, Callable

from hybrid_rag.core.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Set httpx logging level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)


def compute_args_hash(*args: Any) -> str:
    """Compute a hash for the given arguments with safe Unicode handling.

    Args:
        *args: Arguments to hash
    Returns:
        str: Hash string
    """
    # Convert all arguments to strings and join them
    args_str = "".join([str(arg) for arg in args])

    # Use 'replace' error handling to safely encode problematic Unicode characters
    # This replaces invalid characters with Unicode replacement character (U+FFFD)
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        # Handle surrogate characters and other encoding issues
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.

    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    return prefix + compute_args_hash(content)


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def truncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[int]:
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def validate_workspace(workspace: str) -> str:
    """Validate a workspace name before it is used to key stored data.

    Every storage in this project (PostgreSQL and Neo4j) partitions its rows,
    nodes and labels by workspace, so the name ends up embedded in queries and
    identifiers. It must therefore be a single, plain component: no path
    separator, no relative path reference.

    Unlike a sanitizing approach, this validator does not rewrite the name.
    Legitimate names containing dots (e.g. ``"v1.0"``) are accepted unchanged,
    while unsafe names are rejected so the caller fails fast instead of
    silently reading or writing another workspace's data.

    Args:
        workspace: Workspace name from configuration or environment variables.

    Returns:
        The workspace name unchanged when it is valid.

    Raises:
        ValueError: If the workspace contains ``/`` or ``\\``, or is ``"."`` or
            ``".."``.

    Examples:
        >>> validate_workspace("my_workspace")
        'my_workspace'
        >>> validate_workspace("v1.0")
        'v1.0'
        >>> validate_workspace("../../../etc")
        Traceback (most recent call last):
            ...
        ValueError: Invalid workspace name '../../../etc': must not contain path separators ('/', '\\') or be a relative path reference ('.', '..')
    """
    if "/" in workspace or "\\" in workspace or workspace in (".", ".."):
        raise ValueError(
            f"Invalid workspace name {workspace!r}: must not contain path "
            "separators ('/', '\\') or be a relative path reference ('.', '..')"
        )
    return workspace


def normalize_string_list(raw_values: Any, context: str = "") -> list[str]:
    """Return a list of non-empty strings from raw_values.

    Non-string elements are dropped and logged as warnings. If raw_values is
    not a list, an empty list is returned.
    """
    if not isinstance(raw_values, list):
        return []
    result = []
    for i, value in enumerate(raw_values):
        if isinstance(value, str) and value:
            result.append(value)
        else:
            logger.warning(
                "Non-string element dropped from list%s at index %d: %r",
                f" ({context})" if context else "",
                i,
                value,
            )
    return result
