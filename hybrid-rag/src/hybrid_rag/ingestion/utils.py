
"""Ingestion-side helpers — used while a document is being written."""


from typing import Any, Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from hybrid_rag.core.base import BaseKVStorage
from hybrid_rag.core.constants import (
    SOURCE_IDS_LIMIT_METHOD_FIFO,
)
from hybrid_rag.core.utils import (
    Tokenizer,
    _CONTROL_CHAR_PATTERN_ALL,
    _SURROGATE_PATTERN,
    _child_source_span,
    normalize_extracted_info,
    normalize_source_ids_limit_method,
    sanitize_text_for_encoding,
    split_text_by_token_limit,
)
import logging

logger = logging.getLogger(__name__)


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]

def enforce_chunk_token_limit_before_embedding(
    chunking_result: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    tokenizer: Tokenizer,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Hard fallback split before embedding while preserving heading hierarchy."""
    if max_tokens <= 0:
        return list(chunking_result)

    normalized: list[dict[str, Any]] = []

    for dp in chunking_result:
        if not isinstance(dp, dict):
            continue

        content = dp.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        try:
            token_count = len(tokenizer.encode(content))
        except Exception:
            token_count = (
                dp.get("tokens", 0) if isinstance(dp.get("tokens"), int) else 0
            )

        if token_count <= max_tokens:
            ndp = dict(dp)
            ndp["tokens"] = token_count if token_count > 0 else ndp.get("tokens", 0)
            normalized.append(ndp)
            continue

        pieces = split_text_by_token_limit(content, tokenizer, max_tokens)
        if not pieces:
            ndp = dict(dp)
            ndp["tokens"] = token_count
            normalized.append(ndp)
            continue

        base_chunk_id = dp.get("chunk_id")
        parent_span = dp.get("_source_span")
        span_search_from = 0
        total_parts = len(pieces)
        for i, piece in enumerate(pieces, 1):
            new_dp = dict(dp)
            new_dp["content"] = piece
            try:
                new_dp["tokens"] = len(tokenizer.encode(piece))
            except Exception:
                new_dp["tokens"] = max(1, int(len(piece) * 0.5))

            # Shallow-copy preserves the nested heading dict and sidecar block from the source chunk; only the payload (content/token/chunk_id) is rewritten per split slice.
            if isinstance(base_chunk_id, str) and base_chunk_id.strip():
                new_dp["chunk_id"] = f"{base_chunk_id}-s{i:02d}"

            child_span, span_search_from = _child_source_span(
                content, parent_span, piece, span_search_from
            )
            if child_span is not None:
                new_dp["_source_span"] = child_span
            elif "_source_span" in new_dp:
                new_dp.pop("_source_span", None)

            new_dp["split_type"] = "hard_fallback"
            new_dp["split_part"] = i
            new_dp["split_total"] = total_parts
            normalized.append(new_dp)

    # Rebuild order index to keep continuity after splitting.
    for idx, item in enumerate(normalized):
        item["chunk_order_index"] = idx
    return normalized

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

def sanitize_and_normalize_extracted_text(
    input_text: str, remove_inner_quotes=False
) -> str:
    """Santitize and normalize extracted text
    Args:
        input_text: text string to be processed
        is_name: whether the input text is a entity or relation name

    Returns:
        Santitized and normalized text string
    """
    safe_input_text = sanitize_text_for_encoding(input_text)
    if safe_input_text:
        normalized_text = normalize_extracted_info(
            safe_input_text, remove_inner_quotes=remove_inner_quotes
        )
        return normalized_text
    return ""

def strip_control_characters(text: str, replacement_char: str = "") -> str:
    """Remove control/separator chars and surrogates while preserving text.

    Strips the same character classes as :func:`sanitize_text_for_encoding`
    (surrogates via ``_SURROGATE_PATTERN`` and control chars via
    ``_CONTROL_CHAR_PATTERN_ALL`` — including the C0 separators ``\\x1c``-``\\x1f``
    FS/GS/RS/US, while keeping ``\\t``/``\\n``/``\\r``) but deliberately does
    *not* ``html.unescape`` or ``.strip()`` the result.

    This makes it safe for text that carries intentional markup (e.g. sidecar
    block content with ``<table>``/``<drawing>``/``<equation>`` tags, where
    unescaping ``&lt;`` would corrupt the markup) or significant leading/
    trailing whitespace. For control-char-free input it returns the string
    unchanged, so it does not perturb existing content hashes or snapshots.
    """
    if not text:
        return text
    text = _SURROGATE_PATTERN.sub(replacement_char, text)
    return _CONTROL_CHAR_PATTERN_ALL.sub(replacement_char, text)

def merge_source_ids(
    existing_ids: Iterable[str] | None, new_ids: Iterable[str] | None
) -> list[str]:
    """Merge two iterables of source IDs while preserving order and removing duplicates."""

    merged: list[str] = []
    seen: set[str] = set()

    for sequence in (existing_ids, new_ids):
        if not sequence:
            continue
        for source_id in sequence:
            if not source_id:
                continue
            if source_id not in seen:
                seen.add(source_id)
                merged.append(source_id)

    return merged

def apply_source_ids_limit(
    source_ids: Sequence[str],
    limit: int,
    method: str,
    *,
    identifier: str | None = None,
) -> list[str]:
    """Apply a limit strategy to a sequence of source IDs."""

    if limit <= 0:
        return []

    source_ids_list = list(source_ids)
    if len(source_ids_list) <= limit:
        return source_ids_list

    normalized_method = normalize_source_ids_limit_method(method)

    if normalized_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
        truncated = source_ids_list[-limit:]
    else:  # IGNORE_NEW
        truncated = source_ids_list[:limit]

    if identifier and len(truncated) < len(source_ids_list):
        logger.debug(
            "Source_id truncated: %s | %s keeping %s of %s entries",
            identifier,
            normalized_method,
            len(truncated),
            len(source_ids_list),
        )

    return truncated

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
