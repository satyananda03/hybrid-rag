"""Pipeline-specific helpers for document status, identity, and content.

These helpers are shared by the pipeline mixin (``core/pipeline.py``) and by
other IngestionRAG methods that touch the document ingestion paths (custom-chunks
ingest, deletion, etc.). They are kept out of utils.py because they are tied to
the doc_status / full_docs domain rather than to general text/token utilities.
"""


from pathlib import Path
from typing import Any

from hybrid_rag.core.base import DocProcessingStatus, DocStatusStorage
from hybrid_rag.core.constants import (
    FILE_EXTRACTION_SUMMARY_PREFIX,
)
from hybrid_rag.core.utils import compute_mdhash_id
from hybrid_rag.ingestion.text_cleaning import get_content_summary


PLACEHOLDER_DOCUMENT_SOURCES = {"", "no-file-path", "unknown_source"}


def build_chunks_dict_from_chunking_result(
    chunking_result: list[dict[str, Any]],
    *,
    doc_id: str,
    file_path: str,
) -> dict[str, dict[str, Any]]:
    """Assemble the per-doc chunks dict written into chunks_vdb / text_chunks.

    Resolves a stable ``chunk_key`` for each entry — preferring an explicit
    ``chunk_id`` (auto-prefixed with ``doc_id-`` if not already), falling back
    to a positional ``chunk-NNN`` derived from ``chunk_order_index``, and
    finally hashing on collision so two entries inside one document never
    overwrite each other.
    """
    chunks: dict[str, dict[str, Any]] = {}
    for dp in chunking_result:
        chunk_content = dp.get("content", "")
        if not chunk_content:
            continue
        raw_chunk_id = dp.get("chunk_id", "")
        order = dp.get("chunk_order_index")
        if isinstance(raw_chunk_id, str) and raw_chunk_id.strip():
            chunk_key = (
                raw_chunk_id
                if raw_chunk_id.startswith(f"{doc_id}-")
                else f"{doc_id}-{raw_chunk_id}"
            )
        elif isinstance(order, int):
            chunk_key = f"{doc_id}-chunk-{order:03d}"
        else:
            chunk_key = compute_mdhash_id(f"{doc_id}:{chunk_content}", prefix="chunk-")

        # Hard collision guard (same chunk_id inside one document).
        if chunk_key in chunks:
            chunk_key = compute_mdhash_id(
                f"{doc_id}:{order}:{chunk_content}",
                prefix="chunk-",
            )
        # Preserve any pre-populated cache ids on dp (multimodal chunks
        # arrive with analysis cache ids already attached so document
        # deletion can find them via the per-chunk llm_cache_list).
        existing_cache_list = dp.get("llm_cache_list")
        seed_cache_list: list[str] = []
        if isinstance(existing_cache_list, list):
            seen: set[str] = set()
            for entry in existing_cache_list:
                key = str(entry or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    seed_cache_list.append(key)
        stored_chunk = {k: v for k, v in dp.items() if k != "_source_span"}
        chunks[chunk_key] = {
            **stored_chunk,
            "full_doc_id": doc_id,
            "file_path": file_path,
            "llm_cache_list": seed_cache_list,
        }
    return chunks


def chunk_fields_from_status_doc(
    status_doc: DocProcessingStatus,
) -> tuple[list[str], int]:
    """Return (chunks_list, chunks_count) preserved from a status document.

    Filters out any non-string or empty chunk IDs.  When chunks_count is
    absent or invalid, it is inferred from the length of chunks_list.
    """
    chunks_list: list[str] = []
    if isinstance(status_doc.chunks_list, list):
        chunks_list = [
            chunk_id
            for chunk_id in status_doc.chunks_list
            if isinstance(chunk_id, str) and chunk_id
        ]

    if isinstance(status_doc.chunks_count, int) and status_doc.chunks_count >= 0:
        return chunks_list, status_doc.chunks_count

    return chunks_list, len(chunks_list)


def resolve_doc_file_path(
    status_doc: DocProcessingStatus | None = None,
    content_data: dict[str, Any] | None = None,
) -> str:
    """Resolve the best available document file path.

    Returns the first non-placeholder ``file_path`` from doc_status, then
    full_docs. Both are already canonicalized at write time, so this only
    has to skip placeholder sentinels.
    """
    for source in (
        getattr(status_doc, "file_path", None),
        content_data.get("file_path") if content_data else None,
    ):
        if not isinstance(source, str):
            continue
        candidate = source.strip()
        if candidate and candidate not in PLACEHOLDER_DOCUMENT_SOURCES:
            return candidate
    return "unknown_source"


def normalize_document_file_path(file_path: Any) -> str:
    """Return the canonical basename stored as ``file_path``.

    Strips any supported ``[hint]`` segment so ``abc.docx`` and
    ``abc.[native-iet].docx`` map to the same key. Collapses placeholders to
    ``"unknown_source"``. Idempotent.
    """
    source = str(file_path or "").strip()
    if source in PLACEHOLDER_DOCUMENT_SOURCES:
        return "unknown_source"
    canonical = Path(source).name.strip()
    if canonical in PLACEHOLDER_DOCUMENT_SOURCES:
        return "unknown_source"
    return canonical or "unknown_source"


def has_known_document_source(source_key: str) -> bool:
    return source_key not in PLACEHOLDER_DOCUMENT_SOURCES


def doc_status_field(doc: Any, field: str, default: Any = "") -> Any:
    if isinstance(doc, dict):
        return doc.get(field, default)
    return getattr(doc, field, default)


def read_source_file_basename(data: Any) -> str | None:
    """Read the source-file basename with backward compatibility.

    The ``source_file_name`` → ``source_file`` rename means documents enqueued
    before the change persisted the old key in full_docs content_data /
    doc_status.metadata.  Read through here so resumed/legacy docs still resolve
    their original source basename; write sites always use the new
    ``source_file`` key.  Lives here (not in pipeline.py) so doc_status_fields
    helpers can reuse it without importing back into the pipeline module.
    """
    if not isinstance(data, dict):
        return None
    return data.get("source_file") or data.get("source_file_name")


# Long-lived per-document metadata that must survive every doc_status state
# transition, because backends treat ``metadata`` as an opaque blob and replace
# it wholesale on every upsert.
#
# ``source_file`` is the original source basename, kept separate from the
# canonical ``file_path``.
# ``chunk_opts`` is written when entering PROCESSING (via ``extraction_meta``)
# and records the chunker params actually used, in the same format as the
# chunking log line. Carrying it forward keeps it visible after
# PROCESSING -> FAILED, whose ``metadata_extra`` only carries timing fields.
# ``parse_start_time`` is Unix epoch seconds stamped when entering
# ``_parse_worker`` (mirroring ``process_start_time``), so per-stage durations
# can be derived post-mortem.
# ``parse_stage_skipped`` is always set: documents arrive already parsed, so the
# parse stage does no work and has no duration to pair with a start time.
#
# The order of this tuple is the rendering order of metadata fields in the WebUI
# ``DocumentStatusDetailsDialog`` (carry-over builds the new metadata dict by
# iterating it, and dict / JSON / JSX preserve insertion order all the way to
# the rendered output). Keep fields grouped by stage so the dialog reads
# top-to-bottom along the pipeline.
_DOC_STATUS_METADATA_CARRY_OVER_KEYS: tuple[str, ...] = (
    "source_file",
    "kg_ontology",
    "kg_entity_registry",
    "entity_resolution",
    "entity_resolution_llm",
    "orphan_repair",
    "prune_isolated",
    "save_parent",
    "chunk_opts",
    "parse_start_time",
    "parse_stage_skipped",
    "parse_format",
)


def doc_status_metadata_carry_over(status_doc: Any) -> dict[str, Any]:
    """Return the subset of ``status_doc.metadata`` to preserve across upserts.

    ``doc_status`` storage backends generally treat the ``metadata`` field
    as an opaque blob and **replace** it on every upsert, so callers must
    explicitly carry forward fields they want to keep.  This helper centralises
    the list of fields we always carry; extend
    ``_DOC_STATUS_METADATA_CARRY_OVER_KEYS`` to add more.
    """
    if status_doc is None:
        return {}
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return {}
    carry: dict[str, Any] = {}
    for key in _DOC_STATUS_METADATA_CARRY_OVER_KEYS:
        if key in raw_metadata and raw_metadata[key] not in (None, ""):
            carry[key] = raw_metadata[key]
    return carry


def doc_status_transition_metadata(
    status_doc: Any,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a doc_status ``metadata`` payload that preserves carry-over fields.

    Use at every state-transition upsert site so long-lived metadata survives
    PENDING → PARSING → PROCESSING → PROCESSED / FAILED.
    """
    payload = doc_status_metadata_carry_over(status_doc)
    if extra:
        payload.update(extra)
    return payload


# Long-lived per-document *directives* (written at enqueue time by
# ``_initial_doc_status``).  Unlike the full carry-over list above, this is the
# subset that must survive a *reset* back to PENDING: it deliberately EXCLUDES
# the per-attempt timing / result fields (``parse_*`` / ``chunk_opts`` /
# ``stage_seconds``), which the next attempt regenerates and which would otherwise
# show stale values while the document waits in PENDING.
_DOC_STATUS_METADATA_DIRECTIVE_KEYS: tuple[str, ...] = (
    "source_file",
    "kg_ontology",
    "kg_entity_registry",
    "entity_resolution",
    "entity_resolution_llm",
    "orphan_repair",
    "prune_isolated",
    "save_parent",
)


# Per-attempt metadata produced by a parse/process attempt.  Used as
# the *precise* trigger for normalising an already-PENDING document's metadata
# (see ``apipeline_process_enqueue_documents``' consistency check): only a
# document carrying one of these stale keys is rebuilt to directives-only, so
# unrelated (future / custom) metadata is never dropped just for being
# non-directive.  Covers the timing/skip pairs and the ``extraction_meta``
# fields stamped when entering PROCESSING.  Legacy keys from the removed parser
# engines (``parse_end_time`` / ``parse_engine`` / ``parse_warnings``) and the
# removed ANALYZING stage (``analyzing_*``) stay listed so rows written before
# their removal still get scrubbed on retry.
_DOC_STATUS_METADATA_ATTEMPT_KEYS: frozenset[str] = frozenset(
    {
        "parse_start_time",
        "parse_end_time",
        "parse_stage_skipped",
        "analyzing_start_time",
        "analyzing_end_time",
        "analyzing_stage_skipped",
        "parse_warnings",
        "chunk_opts",
        "process_start_time",
        "process_end_time",
        "stage_seconds",
        "parse_format",
        "parse_engine",
        "process_options",
        "chunk_method",
        "skip_kg",
        "mm_chunks",
        "hard_fallback_split",
        "error_type",
        "error_stage",
    }
)


def doc_status_reset_metadata(status_doc: Any) -> dict[str, Any]:
    """Build the ``metadata`` payload for a reset back to PENDING.

    Keeps only the long-lived processing directives
    (``_DOC_STATUS_METADATA_DIRECTIVE_KEYS``) and drops every per-attempt
    timing/result field, so a document that is interrupted and re-queued does
    not surface stale parse/analyze timings (or warnings / chunk opts) while it
    waits in PENDING.  ``source_file`` is read with legacy ``source_file_name``
    tolerance and normalised onto the new key, mirroring the parse worker's own
    normalisation so a resumed legacy pending_parse doc keeps its source hint.
    """
    payload: dict[str, Any] = {}
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return payload
    # Iterate the directive whitelist so a future addition to
    # _DOC_STATUS_METADATA_DIRECTIVE_KEYS is automatically carried across a
    # reset.  ``source_file`` is read with legacy ``source_file_name``
    # tolerance and normalised onto the new key; all other directives carry
    # verbatim when present and non-blank.
    for key in _DOC_STATUS_METADATA_DIRECTIVE_KEYS:
        if key == "source_file":
            value = read_source_file_basename(raw_metadata)
        else:
            value = raw_metadata.get(key)
        if value not in (None, ""):
            payload[key] = value
    return payload


def doc_status_parse_failure_fields(
    error: BaseException,
    *,
    status_doc: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the FAILED-upsert extras for a parse-stage failure.

    Returns ``(extra_fields, metadata_extra)`` for
    ``_upsert_doc_status_transition``.  Restores the UI fields that the
    pre-deferral enqueue-time error documents carried: a human-readable
    ``content_summary`` (written only when the document has none —
    ``pending_parse`` docs enqueue with an empty summary, while raw
    passthrough docs keep their real one — or when the existing summary
    is itself a ``[File Extraction]``-generated one from a previous
    failed attempt, which the retry reset preserves and would otherwise
    go stale next to the fresh ``error_msg``) plus ``metadata.error_type`` /
    ``metadata.error_stage`` for error classification.  ``error_type``
    keeps the legacy ``file_extraction_error`` value consumers may match
    on; ``error_stage`` distinguishes the parse-worker failure from an
    enqueue-time error document.  Both metadata keys ride
    ``metadata_extra`` and are deliberately NOT in the carry-over /
    directive whitelists, so the next transition (retry reset, PARSING)
    drops them automatically — mirroring how ``error_msg`` is cleared.
    """
    error_text = str(error)
    if not error_text.strip():
        # Some exceptions (e.g. certain httpx transport errors) stringify to
        # an empty message; fall back to the class name so the WebUI never
        # drops the Error Message row for a FAILED document.
        error_text = type(error).__name__
    extra_fields: dict[str, Any] = {"error_msg": error_text}
    current_summary = str(
        doc_status_field(status_doc, "content_summary", "") or ""
    ).strip()
    if not current_summary or current_summary.startswith(
        FILE_EXTRACTION_SUMMARY_PREFIX
    ):
        extra_fields["content_summary"] = (
            FILE_EXTRACTION_SUMMARY_PREFIX + get_content_summary(error_text)
        )
    metadata_extra: dict[str, Any] = {
        "error_type": "file_extraction_error",
        "error_stage": "parse",
    }
    return extra_fields, metadata_extra


def doc_status_metadata_has_attempt_fields(status_doc: Any) -> bool:
    """True when ``status_doc.metadata`` carries any per-attempt field.

    Used to decide whether an already-PENDING document needs its metadata
    normalised to directives-only — avoids a redundant upsert for documents
    whose metadata is already clean (or only holds non-attempt custom fields).
    """
    raw_metadata = doc_status_field(status_doc, "metadata", {})
    if not isinstance(raw_metadata, dict):
        return False
    return not _DOC_STATUS_METADATA_ATTEMPT_KEYS.isdisjoint(raw_metadata)


def compute_text_content_hash(content: str) -> str:
    """MD5 hex digest of text content used for cross-filename dedup."""
    return compute_mdhash_id(content, prefix="")


async def get_existing_doc_by_file_basename(
    doc_status: DocStatusStorage, file_path: Any
) -> tuple[str, Any] | None:
    """Find an existing doc_status record by canonical file basename.

    Inputs are normalized via :func:`normalize_document_file_path` so callers
    may pass either the bare canonical name (``abc.docx``) or a hint-bearing
    variant (``abc.[native-iet].docx``); both resolve to the same logical
    document.
    """
    basename = normalize_document_file_path(file_path)
    if basename == "unknown_source":
        return None
    return await doc_status.get_doc_by_file_basename(basename)


async def get_existing_doc_by_content_hash(
    doc_status: DocStatusStorage, content_hash: str
) -> tuple[str, Any] | None:
    """Find an existing doc_status record by content hash."""
    if not content_hash:
        return None
    return await doc_status.get_doc_by_content_hash(content_hash)


async def get_duplicate_doc_by_content_hash(
    doc_status: DocStatusStorage, content_hash: str, current_doc_id: str
) -> tuple[str, Any] | None:
    """Find another doc_status record with the same content hash."""
    if not content_hash:
        return None
    return await doc_status.get_doc_by_content_hash(
        content_hash, exclude_doc_id=current_doc_id
    )

