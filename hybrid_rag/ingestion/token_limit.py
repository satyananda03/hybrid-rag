"""Hard-fallback split that keeps every chunk within the embedding token limit,
while each piece keeps an exact source span inside its parent."""

import bisect
import re
from typing import Any

from hybrid_rag.core.tokenizer import Tokenizer


def split_text_units_for_hard_fallback(text: str) -> list[str]:
    """Split text into sentence/paragraph-like units for fallback chunking."""
    if not text:
        return []
    units: list[str] = []
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        for sentence in re.split(r"(?<=[。！？；.!?])", p):
            s = sentence.strip()
            if s:
                units.append(s)
    return units if units else [text]


def split_text_by_token_limit(
    text: str, tokenizer: Tokenizer, max_tokens: int
) -> list[str]:
    """Split text by token limit with sentence-first, token-window fallback."""
    if not text:
        return []

    try:
        total_tokens = len(tokenizer.encode(text))
    except Exception:
        total_tokens = 0

    if total_tokens > 0 and total_tokens <= max_tokens:
        return [text]

    units = split_text_units_for_hard_fallback(text)
    out: list[str] = []
    cur_parts: list[str] = []
    cur_tokens = 0

    for unit in units:
        try:
            unit_tokens = len(tokenizer.encode(unit))
        except Exception:
            unit_tokens = 0

        # Sentence itself is oversize: token-window split directly.
        if unit_tokens > max_tokens:
            if cur_parts:
                out.append("\n\n".join(cur_parts))
                cur_parts = []
                cur_tokens = 0

            token_ids = tokenizer.encode(unit)
            for start in range(0, len(token_ids), max_tokens):
                piece = tokenizer.decode(token_ids[start : start + max_tokens]).strip()
                if piece:
                    out.append(piece)
            continue

        if cur_parts and cur_tokens + unit_tokens > max_tokens:
            out.append("\n\n".join(cur_parts))
            cur_parts = [unit]
            cur_tokens = unit_tokens
        else:
            cur_parts.append(unit)
            cur_tokens += unit_tokens

    if cur_parts:
        out.append("\n\n".join(cur_parts))

    return [x for x in out if x.strip()]


def _normalized_child_offsets(
    parent_content: str,
    piece: str,
    search_from: int,
) -> tuple[int, int] | None:
    """Locate ``piece`` in ``parent_content`` ignoring all whitespace.

    Returns ``(start, end)`` char offsets into ``parent_content`` for the first
    whitespace-stripped occurrence at/after ``search_from``, or ``None`` if absent.
    Removing every whitespace char (not collapsing runs) keeps the match exact even
    when the two sides space the same characters differently — the same monotonic
    projection sidecar span backfill relies on.
    """
    norm_piece = "".join(piece.split())
    if not norm_piece:
        return None
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    for idx, ch in enumerate(parent_content):
        if ch.isspace():
            continue
        norm_chars.append(ch)
        norm_to_orig.append(idx)
    norm_parent = "".join(norm_chars)
    # First normalized index whose source offset is >= search_from (norm_to_orig is
    # strictly increasing), so repeated pieces resolve forward in order.
    norm_start = bisect.bisect_left(norm_to_orig, search_from)
    pos = norm_parent.find(norm_piece, norm_start)
    if pos < 0:
        return None
    o_start = norm_to_orig[pos]
    o_end = norm_to_orig[pos + len(norm_piece) - 1] + 1
    return o_start, o_end


def _child_source_span(
    parent_content: str,
    parent_span: Any,
    piece: str,
    search_from: int,
) -> tuple[dict[str, int] | None, int]:
    """Locate a hard-split child ``piece`` inside its parent's source span.

    Pieces are usually verbatim substrings of ``parent_content`` (token-window
    slices), so an exact forward ``find`` resolves them precisely. But
    :func:`split_text_by_token_limit` rejoins multiple sentence units with
    ``"\\n\\n"``, so a multi-unit piece is *not* byte-verbatim when the source
    separated those sentences with a single space/newline. In that case we fall
    back to a whitespace-stripped match (the same projection sidecar backfill uses),
    which stays exact because whitespace removal is monotonic. Without this fallback
    the child would lose its span and sidecar backfill would wrongly FAIL the
    document.

    Returns ``(span | None, next_search_from)`` where ``next_search_from`` is a
    ``parent_content`` offset threaded forward by the caller so repeated pieces
    resolve in order.
    """
    if not isinstance(parent_span, dict):
        return None, search_from
    try:
        parent_start = int(parent_span["start"])
        parent_end = int(parent_span["end"])
    except (KeyError, TypeError, ValueError):
        return None, search_from
    if parent_start < 0 or parent_end < parent_start:
        return None, search_from

    search_from = max(0, search_from)

    # Exact: verbatim token-window pieces.
    local_start = parent_content.find(piece, search_from)
    if local_start >= 0:
        local_end = local_start + len(piece)
    else:
        # Whitespace-normalized fallback: multi-unit pieces rejoined with "\n\n".
        offsets = _normalized_child_offsets(parent_content, piece, search_from)
        if offsets is None:
            return None, search_from
        local_start, local_end = offsets

    if parent_start + local_end > parent_end:
        return None, search_from
    return (
        {"start": parent_start + local_start, "end": parent_start + local_end},
        local_end,
    )


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
