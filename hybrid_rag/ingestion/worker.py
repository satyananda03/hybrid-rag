
"""Document ingestion pipeline mixin for the ingestion ``IngestionRAG`` class.

This module isolates the document parse/enqueue/extraction pipeline so that
``rag.py`` stays focused on storage management.
The mixin is wired into the class via multiple inheritance
and relies on attributes/methods that the main class provides
(``self.full_docs``, ``self.doc_status``, ``self.tokenizer``,
``self.parse_native``-related fields, ``self._insert_done``,
``self._process_extract_entities``, etc.).
"""


import asyncio
import contextlib
import inspect
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hybrid_rag.core.base import DocProcessingStatus, DocStatus
from hybrid_rag.core.constants import FULL_DOCS_FORMAT_RAW
from hybrid_rag.core.exceptions import (
    PipelineCancelledException,
    IndexFlushError,
)
from hybrid_rag.storage.shared import get_namespace_data, get_namespace_lock
from hybrid_rag.ingestion.knowledge_graph.extraction import merge_nodes_and_edges
from hybrid_rag.ingestion.knowledge_graph.ontology_schema import load_ontology_schema
from hybrid_rag.ingestion.parse_stage import parse_stored_document
from hybrid_rag.ingestion.multimodal.analyze import resolve_image_sources
from hybrid_rag.ingestion.utils import enforce_chunk_token_limit_before_embedding, strip_control_characters
from hybrid_rag.core.utils import (
    raise_first_failure,
    compute_mdhash_id,
    generate_track_id,
    get_content_summary,
    sanitize_text_for_encoding,
)
from hybrid_rag.config import settings
import logging
from hybrid_rag import progress

from hybrid_rag.ingestion.doc_status_fields import (
    build_chunks_dict_from_chunking_result,
    chunk_fields_from_status_doc,
    compute_text_content_hash,
    doc_status_field,
    doc_status_parse_failure_fields,
    doc_status_transition_metadata,
    get_duplicate_doc_by_content_hash,
    get_existing_doc_by_content_hash,
    get_existing_doc_by_file_basename,
    has_known_document_source,
    normalize_document_file_path,
    doc_status_metadata_has_attempt_fields,
    doc_status_reset_metadata,
    read_source_file_basename,
    resolve_doc_file_path,
)

logger = logging.getLogger(__name__)


# Document statuses the pipeline considers "in-flight or pending" — the
# snapshot each processing run starts from.  Module-level so we don't
# reconstruct the list on every pipeline entry.
_INFLIGHT_DOC_STATUSES = (
    DocStatus.PROCESSING,
    DocStatus.FAILED,
    DocStatus.PENDING,
    DocStatus.PARSING,
    # No longer written (the stage was removed); kept so a row left in it by an
    # older deployment is still picked up and reset instead of stranded.
    DocStatus.ANALYZING,
)


# Backward-compatible source-file reader.  Implementation lives in
# doc_status_fields so reset/normalisation helpers there can reuse it without a
# reverse import into this module; kept as a module-level alias for the
# existing call sites below.
_read_source_file = read_source_file_basename


_ingestion_slots: asyncio.Semaphore | None = None


@asynccontextmanager
async def _ingestion_turn():
    """One of ``MAX_PARALLEL_INGESTION`` process-wide slots for processing a document.

    Process-wide, not per batch: every workspace runs its own pipeline batch, so
    a per-batch limit never capped the total load the model endpoints see. The
    wait is announced on the document's stream so a queued upload does not look
    stuck. Created on first use, like the Textract job slots.
    """
    global _ingestion_slots
    if _ingestion_slots is None:
        _ingestion_slots = asyncio.Semaphore(max(1, settings.max_parallel_ingestion))
    if _ingestion_slots.locked():
        progress.activity("Menunggu antrean pemrosesan: server sedang memproses dokumen lain")
    async with _ingestion_slots:
        yield


@dataclass
class _BatchRunContext:
    """Per-batch shared state for the parse/process worker pipeline.

    Bundles the cross-cutting handles (pipeline_status, locks, queues) so
    worker methods accept a single ``ctx`` argument instead of individually
    plumbed parameters.  ``processed_count`` mutates inside
    each batch and is always read/written under ``pipeline_status_lock``.
    """

    pipeline_status: dict
    pipeline_status_lock: Any
    total_files: int
    q_parse: asyncio.Queue
    q_process: asyncio.Queue
    processed_count: int = 0


@dataclass
class _DocRun:
    """One document's processing state, shared by its text and multimodal halves.

    Written as the halves progress and read by the failure path, which records
    whatever child chunks exist so far.
    """

    process_start_time: int
    t_start: float = 0.0
    chunks: dict[str, Any] = field(default_factory=dict)
    extraction_meta: dict[str, Any] = field(default_factory=dict)
    # Cumulative seconds since the purge at the end of each stage.
    stage_seconds: dict[str, float] = field(default_factory=dict)
    # Chunk embedding started while extraction runs; held so it is not GC'd.
    early_embed: asyncio.Task | None = None

    def mark(self, stage: str) -> None:
        self.stage_seconds[stage] = round(time.monotonic() - self.t_start, 1)


class _PipelineMixin:
    """Mixin providing the document ingestion pipeline methods.

    Designed to be combined as a base of the ingestion ``IngestionRAG`` only.
    Relies on its attributes (``self.full_docs``, ``self.doc_status``,
    ``self.tokenizer``, ``self.parser_*``, ``self.workspace`` ...) and on the
    shared methods ``self._insert_done`` / ``self._process_extract_entities``
    which remain in the main class and are resolved through MRO.
    """

    # ============================================================
    # Public document ingestion API (entry points)
    # ============================================================

    async def apipeline_enqueue_documents(
        self,
        input: str | list[str],
        ids: list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
        multimodal_items: list[list[dict[str, Any]]] | None = None,
        ontology: str | None = None,
        page_offsets: list[list[dict[str, int]]] | None = None,
        save_parent: bool = True,
    ) -> str:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs and remove duplicate contents
        2. Generate document initial status
        3. Filter out already processed documents
        4. Enqueue document in status

        Documents are always enqueued in ``raw`` format: their content arrived
        already extracted (see :mod:`hybrid_rag.ingestion.parse_stage`), so
        there is no engine to pick and nothing to defer to the parse worker.

        Args:
            input: Single document string or list of document strings
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated (from content or file_path)
            file_paths: list of file paths corresponding to each document, used for citation
            track_id: tracking ID for monitoring processing status
            ontology: ontology file the documents are extracted under; stored as
                doc_status metadata ``kg_ontology``

        Returns:
            str: tracking ID for monitoring processing status
        """
        # Fail before anything is written: a typo'd or unparsable ontology would
        # otherwise only surface mid-pipeline, after the document was enqueued.
        if ontology:
            load_ontology_schema(ontology)
        # Generate track_id if not provided
        if track_id is None or track_id.strip() == "":
            track_id = generate_track_id("enqueue")
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # If file_paths is provided, ensure it matches the number of documents
        if file_paths is not None:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            if len(file_paths) != len(input):
                raise ValueError(
                    "Number of file paths must match the number of documents"
                )
            file_paths = [
                path.strip() if isinstance(path, str) else "" for path in file_paths
            ]
            file_paths = [path if path else "unknown_source" for path in file_paths]
        else:
            file_paths = ["unknown_source"] * len(input)


        # 1. Validate ids and build contents
        if ids is not None:
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

        # Canonicalize every input filename once: the stored ``file_path``
        # is hint-stripped and serves UI display, filename dedup, and the
        # deterministic doc_id seed in one go.
        file_paths_canonical = [
            normalize_document_file_path(path) for path in file_paths
        ]
        contents: dict[str, dict[str, Any]] = {}
        source_to_doc_id: dict[str, str] = {}
        content_hash_to_doc_id: dict[str, str] = {}
        duplicate_attempts: list[dict[str, Any]] = []

        def _add_content(
            index: int,
            content: str,
            doc_format: str,
            modal_items: list[dict[str, Any]] | None = None,
            offsets: list[dict[str, int]] | None = None,
        ) -> None:
            file_path_canonical = file_paths_canonical[index]

            body_length = len(content)

            # Compute content hash: skip for pending_parse (content extracted later).
            content_hash: str | None = None
            if doc_format == FULL_DOCS_FORMAT_RAW and (content or "").strip():
                # An empty body carries no identity: hashing it would give every
                # figure-only document the same hash and make the second one look
                # like a duplicate of the first. Filename dedup still applies.
                content_hash = compute_text_content_hash(content)

            known_source = has_known_document_source(file_path_canonical)
            if ids is not None:
                doc_id = ids[index]
            elif known_source:
                doc_id = compute_mdhash_id(file_path_canonical, prefix="doc-")
            elif doc_format == FULL_DOCS_FORMAT_RAW:
                doc_id = compute_mdhash_id(content or "", prefix="doc-")
            else:
                doc_id = compute_mdhash_id(
                    f"{file_path_canonical}-{track_id}-{index}", prefix="doc-"
                )

            if known_source and file_path_canonical in source_to_doc_id:
                duplicate_attempts.append(
                    {
                        "doc_id": doc_id,
                        "original_doc_id": source_to_doc_id[file_path_canonical],
                        "file_path": file_path_canonical,
                        "content_length": body_length,
                        "existing_status": "batch_duplicate",
                        "existing_track_id": "",
                        "duplicate_kind": "filename",
                    }
                )
                return

            if content_hash and content_hash in content_hash_to_doc_id:
                duplicate_attempts.append(
                    {
                        "doc_id": doc_id,
                        "original_doc_id": content_hash_to_doc_id[content_hash],
                        "file_path": file_path_canonical,
                        "content_length": body_length,
                        "existing_status": "batch_duplicate",
                        "existing_track_id": "",
                        "duplicate_kind": "content_hash",
                    }
                )
                return

            if known_source:
                source_to_doc_id[file_path_canonical] = doc_id
            if content_hash:
                content_hash_to_doc_id[content_hash] = doc_id

            content_data: dict[str, Any] = {
                "content": content,
                "file_path": file_path_canonical,
                "parse_format": doc_format,
                # Stored verbatim so process_single_document analyzes exactly
                # what separate_content() produced — see parse_stage.py.
                "multimodal_items": list(modal_items or []),
            }
            if content_hash:
                content_data["content_hash"] = content_hash
            if offsets:
                # Awal tiap halaman di ``content`` (koordinat pasca-sanitasi, dihitung
                # separate_content). Menumpang slot JSONB ``chunk_options`` yang sudah
                # ada supaya sampai ke process_single_document tanpa kolom baru.
                content_data["chunk_options"] = {"page_offsets": list(offsets)}
            contents[doc_id] = content_data

        if multimodal_items is not None and len(multimodal_items) != len(input):
            raise ValueError(
                "Number of multimodal item lists must match the number of documents"
            )
        for i, doc in enumerate(input):
            _add_content(
                i,
                sanitize_text_for_encoding(doc),
                FULL_DOCS_FORMAT_RAW,
                multimodal_items[i] if multimodal_items else None,
                page_offsets[i] if page_offsets else None,
            )

        # 2. Generate document initial status (without content)
        def _initial_doc_status(content_data: dict[str, Any]) -> dict[str, Any]:
            body_text = content_data.get("content", "")
            base: dict[str, Any] = {
                "status": DocStatus.PENDING,
                "content_summary": get_content_summary(body_text),
                "content_length": len(body_text),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data["file_path"],
                "track_id": track_id,
            }
            if content_data.get("content_hash"):
                base["content_hash"] = content_data["content_hash"]
            metadata: dict[str, Any] = {}
            source_file = _read_source_file(content_data)
            if source_file:
                metadata["source_file"] = source_file
            # Which ontology this document is extracted under. Stored here
            # because the pipeline is broken by storage between enqueue and
            # processing — apipeline_process_enqueue_documents takes no arguments,
            # so a per-document value has to survive a write and a read.
            if ontology:
                metadata["kg_ontology"] = ontology
            # Ditulis tanpa penjagaan falsy, beda dari ``ontology``: ``False`` harus
            # ikut tersimpan, bukan menghilang jadi "tidak diset" — pembacanya
            # memakai default ``True``.
            metadata["save_parent"] = bool(save_parent)
            if metadata:
                base["metadata"] = metadata
            return base

        new_docs: dict[str, Any] = {
            id_: _initial_doc_status(content_data)
            for id_, content_data in contents.items()
        }

        # Serialise the dedup-read-then-upsert critical section across
        # concurrent enqueue calls within the same workspace.  Without
        # this, two enqueues for the same content (e.g. /upload during
        # scan's processing phase, or two uploads via /text + /upload)
        # can both read doc_status before either upserts, both miss the
        # content_hash dedup, and both end up writing PENDING rows for
        # the same content — bypassing the dedup that's supposed to
        # land one of them as ``duplicate_kind=content_hash`` FAILED.
        #
        # The lock is workspace-scoped and only spans steps 3-4 below
        # (filter_keys → upserts).  It does NOT block concurrent
        # processing (``apipeline_process_enqueue_documents`` reads
        # doc_status independently) or scan classification
        # (``scanning_exclusive`` already gates concurrent enqueue).
        # Lock order: enqueue_serialize → pipeline_status_lock; no caller
        # holds pipeline_status_lock first then needs enqueue_serialize.
        enqueue_serialize_lock = get_namespace_lock(
            "enqueue_serialize", workspace=self.workspace
        )

        async with enqueue_serialize_lock:
            # 3. Filter out already processed documents
            # Get docs ids
            all_new_doc_ids = set(new_docs.keys())
            # Exclude IDs of documents that are already enqueued.  The previous
            # ``reprocess_existing_non_processed`` flag has been removed: any
            # same-name record (regardless of status) is treated as a duplicate
            # here.  Recovering half-processed documents is now the job of the
            # pipeline's resume logic, which runs in apipeline_process_enqueue_documents
            # rather than this enqueue path.
            unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

            for doc_id in list(unique_new_doc_ids):
                content_data = contents[doc_id]

                # 3a. Filename-based dedup: same basename always treated as duplicate.
                match = await get_existing_doc_by_file_basename(
                    self.doc_status, content_data["file_path"]
                )
                if match:
                    existing_doc_id, existing_doc = match
                    unique_new_doc_ids.discard(doc_id)
                    duplicate_attempts.append(
                        {
                            "doc_id": doc_id,
                            "original_doc_id": existing_doc_id,
                            "file_path": content_data["file_path"],
                            "content_length": new_docs.get(doc_id, {}).get(
                                "content_length", 0
                            ),
                            "existing_status": doc_status_field(
                                existing_doc, "status", "unknown"
                            ),
                            "existing_track_id": doc_status_field(
                                existing_doc, "track_id", ""
                            ),
                            "duplicate_kind": "filename",
                        }
                    )
                    continue

                # 3b. Content-hash dedup: different filename but same body still dupes.
                content_hash = content_data.get("content_hash")
                if not content_hash:
                    continue
                hash_match = await get_existing_doc_by_content_hash(
                    self.doc_status, content_hash
                )
                if hash_match:
                    existing_doc_id, existing_doc = hash_match
                    unique_new_doc_ids.discard(doc_id)
                    duplicate_attempts.append(
                        {
                            "doc_id": doc_id,
                            "original_doc_id": existing_doc_id,
                            "file_path": content_data["file_path"],
                            "content_length": new_docs.get(doc_id, {}).get(
                                "content_length", 0
                            ),
                            "existing_status": doc_status_field(
                                existing_doc, "status", "unknown"
                            ),
                            "existing_track_id": doc_status_field(
                                existing_doc, "track_id", ""
                            ),
                            "duplicate_kind": "content_hash",
                        }
                    )

            # Handle duplicate documents - create trackable records with current track_id
            ignored_ids = list(all_new_doc_ids - unique_new_doc_ids)
            for doc_id in ignored_ids:
                if any(
                    attempt.get("doc_id") == doc_id for attempt in duplicate_attempts
                ):
                    continue
                existing_doc = await self.doc_status.get_by_id(doc_id)
                duplicate_attempts.append(
                    {
                        "doc_id": doc_id,
                        "original_doc_id": doc_id,
                        "file_path": new_docs.get(doc_id, {}).get(
                            "file_path", "unknown_source"
                        ),
                        "content_length": new_docs.get(doc_id, {}).get(
                            "content_length", 0
                        ),
                        "existing_status": (
                            existing_doc.get("status", "unknown")
                            if existing_doc
                            else "unknown"
                        ),
                        "existing_track_id": (
                            existing_doc.get("track_id", "") if existing_doc else ""
                        ),
                        "duplicate_kind": "filename",
                    }
                )

            if duplicate_attempts:
                duplicate_docs: dict[str, Any] = {}
                for index, attempt in enumerate(duplicate_attempts):
                    doc_id = attempt["doc_id"]
                    file_path = attempt.get("file_path") or "unknown_source"
                    duplicate_kind = attempt.get("duplicate_kind") or "filename"
                    logger.warning(
                        f"Duplicate document detected ({duplicate_kind}): "
                        f"{doc_id} ({file_path})"
                    )

                    # Create a new record with unique ID for this duplicate attempt
                    dup_record_id = compute_mdhash_id(
                        f"{doc_id}-{track_id}-{index}-{file_path}", prefix="dup-"
                    )
                    if duplicate_kind == "content_hash":
                        error_prefix = (
                            "Identical content already exists under another filename."
                        )
                    else:
                        error_prefix = "File name already exists."
                    duplicate_docs[dup_record_id] = {
                        "status": DocStatus.FAILED,
                        "content_summary": (
                            f"[DUPLICATE:{duplicate_kind}] Original document: "
                            f"{attempt.get('original_doc_id', doc_id)}"
                        ),
                        "content_length": attempt.get("content_length", 0),
                        "chunks_count": 0,
                        "chunks_list": [],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": file_path,
                        "track_id": track_id,  # Use current track_id for tracking
                        "error_msg": (
                            f"{error_prefix} "
                            f"Original doc_id: {attempt.get('original_doc_id', doc_id)}, "
                            f"Status: {attempt.get('existing_status', 'unknown')}"
                        ),
                        "metadata": {
                            "is_duplicate": True,
                            "duplicate_kind": duplicate_kind,
                            "original_doc_id": attempt.get("original_doc_id", doc_id),
                            "original_track_id": attempt.get("existing_track_id", ""),
                        },
                    }

                # Store duplicate records in doc_status
                if duplicate_docs:
                    await self.doc_status.upsert(duplicate_docs)
                    logger.info(
                        f"Created {len(duplicate_docs)} duplicate document records with track_id: {track_id}"
                    )

            # Filter new_docs to only include documents with unique IDs
            new_docs = {
                doc_id: new_docs[doc_id]
                for doc_id in unique_new_doc_ids
                if doc_id in new_docs
            }

            if not new_docs:
                logger.warning("No new unique documents were found.")
                return

            # 4. Store document content in full_docs and status in doc_status
            full_docs_data = {
                doc_id: {
                    "content": contents[doc_id].get("content", ""),
                    "file_path": contents[doc_id]["file_path"],
                    "parse_format": contents[doc_id].get(
                        "parse_format", FULL_DOCS_FORMAT_RAW
                    ),
                    # The multimodal half must be persisted here or
                    # process_single_document has nothing to analyze: this dict is rebuilt key by key,
                    # so anything added to ``contents`` above is dropped unless it
                    # is also listed here.
                    "multimodal_items": contents[doc_id].get("multimodal_items")
                    or [],
                    "chunk_options": contents[doc_id].get("chunk_options") or {},
                }
                for doc_id in new_docs.keys()
            }
            for doc_id in new_docs.keys():
                if contents[doc_id].get("content_hash"):
                    full_docs_data[doc_id]["content_hash"] = contents[doc_id][
                        "content_hash"
                    ]
            await self.full_docs.upsert(full_docs_data)
            # Persist data to disk immediately
            await self.full_docs.index_done_callback()

            # Store document status (without content)
            await self.doc_status.upsert(new_docs)
            logger.debug(f"Stored {len(new_docs)} new unique documents")

        # No ``request_pending`` nudge: that signalled an in-flight processing
        # loop belonging to a CONCURRENT caller, and there is none — every
        # enqueue here is followed by its own
        # ``apipeline_process_enqueue_documents`` on the same task, serialized
        # by the advisory lock in ``engine``.
        return track_id

    async def apipeline_process_enqueue_documents(self) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Validate document data consistency and fix any issues
        3. Split document content into chunks
        4. Process each chunk for entity and relation extraction
        5. Update the document status
        """
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )

        async with pipeline_status_lock:
            to_process_docs: dict[
                str, DocProcessingStatus
            ] = await self.doc_status.get_docs_by_statuses(list(_INFLIGHT_DOC_STATUSES))

            if not to_process_docs:
                logger.info("No documents to process")
                return

            pipeline_status.update(
                {
                    "job_name": "Default Job",
                    "job_start": datetime.now(timezone.utc).isoformat(),
                    "docs": 0,
                    "batchs": 0,  # Total number of files to be processed
                    "cur_batch": 0,  # Number of files already processed
                    "cancellation_requested": False,  # Initialize cancellation flag
                    "cancellation_reason": None,  # "internal_error" or None (user)
                    "cancellation_detail": None,  # driver + root cause for internal
                }
            )

        try:
            # Validate document data consistency and fix any issues
            to_process_docs = await self._validate_and_fix_document_consistency(
                to_process_docs
            )

            if not to_process_docs:
                logger.info("No valid documents to process after consistency check")
                return

            logger.info(f"Processing {len(to_process_docs)} document(s)")
            pipeline_status["docs"] = len(to_process_docs)
            pipeline_status["batchs"] = len(to_process_docs)
            pipeline_status["cur_batch"] = 0

            # One batch per call, no loop. Upstream wrapped this in a ``while``
            # that re-queried doc_status so a CONCURRENT caller's enqueue could
            # be picked up mid-run — signalled through ``request_pending``, which
            # only a caller that saw ``busy=True`` would ever set. Ingestion here
            # is driven serially by ``engine`` behind a Postgres advisory lock, so
            # there is no second caller to signal: every claimable document is
            # already in this batch and the extra pass was unreachable.
            await self._run_pipeline_batch(
                to_process_docs,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )

        finally:
            stopped_message = "Enqueued document processing pipeline stopped"
            logger.info(stopped_message)
            async with pipeline_status_lock:
                # An internal-error abort leaves the batch quietly — without
                # this the only visible trace would be the generic "stopped"
                # line above. Surface the actionable halt reason BEFORE
                # clearing reason/detail, so _cancellation_label still sees
                # the cause.
                if pipeline_status.get("cancellation_reason") == "internal_error":
                    logger.error(
                        self._internal_halt_message(
                            self._cancellation_label(pipeline_status)
                        )
                    )
                pipeline_status["cancellation_requested"] = (
                    False  # Always reset cancellation flag
                )
                pipeline_status["cancellation_reason"] = None
                pipeline_status["cancellation_detail"] = None

    # ============================================================
    # Pipeline orchestration
    # ============================================================

    async def _run_pipeline_batch(
        self,
        to_process_docs: dict[str, DocProcessingStatus],
        *,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Run one batch of pending documents through the parse → process queues.

        Two cascading layers of queues:
          - Layer 1: Parse stage (no parser runs — see parse_stage.py)
          - Layer 2: Chunking, multimodal analysis and entity / relation
            extraction  (process_single_document)
        """
        total_files = len(to_process_docs)
        pipeline_status["job_name"] = self._format_job_name(
            to_process_docs, total_files
        )

        ctx = _BatchRunContext(
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            total_files=total_files,
            # Unbounded: items are lightweight metadata (the body is re-read
            # from full_docs) and there are at most total_files of them.
            q_parse=asyncio.Queue(),
            q_process=asyncio.Queue(),
        )

        # One parse worker is enough: it parses nothing (Textract runs before
        # enqueue, capped by TEXTRACT_MAX_CONCURRENT_JOBS) and only does a few
        # bookkeeping writes per document, far faster than the process stage.
        workers: list[asyncio.Task] = [asyncio.create_task(self._parse_worker(ctx))]
        # No more process workers than documents: the real cap is the
        # process-wide ingestion slot each document takes (_ingestion_turn).
        for _ in range(max(1, min(settings.max_parallel_ingestion, total_files))):
            workers.append(asyncio.create_task(self._process_worker(ctx)))

        # The workers above are live asyncio tasks; their cancellation MUST be
        # guaranteed even if enqueuing or a queue join raises (e.g. an orchestrator-
        # level storage call fails during a backend outage). Without this try/finally
        # an escape here would orphan the workers — they keep draining the queues
        # after the caller has already returned, so the task looks finished while
        # processing visibly continues.
        try:
            # Hand every pending document to the parse queue. Upstream this
            # loop also read full_docs to pick a per-engine queue; with a single
            # parse pool there is nothing to route on, and the parse worker
            # reads full_docs itself (failing just that document if the row is
            # missing), so the read is gone.
            for doc_id, status_doc in to_process_docs.items():
                await ctx.q_parse.put((doc_id, status_doc))

            await ctx.q_parse.join()
            await ctx.q_process.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        # If the batch aborted on an internal storage error, the shared
        # cross-file flush buffers may still hold records from the documents
        # that were marked FAILED. Discard them now (workers are stopped, so
        # this does not race a flush) so they are neither re-flushed nor
        # carried into the next batch — every affected document is reprocessed
        # on retry. See _discard_pending_index_ops / drop_pending_index_ops.
        async with pipeline_status_lock:
            internal_abort = (
                pipeline_status.get("cancellation_requested", False)
                and pipeline_status.get("cancellation_reason") == "internal_error"
            )
        if internal_abort:
            await self._discard_pending_index_ops()

    async def _validate_and_fix_document_consistency(
        self,
        to_process_docs: dict[str, DocProcessingStatus],
    ) -> dict[str, DocProcessingStatus]:
        """Validate and fix document data consistency by deleting inconsistent entries, but preserve failed documents"""
        inconsistent_docs = []
        failed_docs_to_preserve = []

        # Check each document's data consistency: one existence query for the
        # whole batch, since only whether full_docs has the row matters here,
        # not its (possibly large) body.
        missing_content = await self.full_docs.filter_keys(set(to_process_docs))
        for doc_id, status_doc in to_process_docs.items():
            if doc_id in missing_content:
                # Check if this is a failed document that should be preserved
                if (
                    hasattr(status_doc, "status")
                    and status_doc.status == DocStatus.FAILED
                ):
                    failed_docs_to_preserve.append(doc_id)
                else:
                    inconsistent_docs.append(doc_id)

        # Log information about failed documents that will be preserved
        if failed_docs_to_preserve:
            preserve_message = f"Preserving {len(failed_docs_to_preserve)} failed document entries for manual review"
            logger.info(preserve_message)

            # Remove failed documents from processing list but keep them in doc_status
            for doc_id in failed_docs_to_preserve:
                to_process_docs.pop(doc_id, None)

        # Delete inconsistent document entries(excluding failed documents)
        if inconsistent_docs:
            summary_message = (
                f"Inconsistent document entries found: {len(inconsistent_docs)}"
            )
            logger.info(summary_message)

            for doc_id in inconsistent_docs:
                # No content to process, whether or not its status row can be
                # removed now: take it out of this batch either way.
                status_doc = to_process_docs.pop(doc_id)
                file_path = resolve_doc_file_path(status_doc=status_doc)
                try:
                    await self.doc_status.delete([doc_id])
                    logger.info(f"Deleted inconsistent entry: {doc_id} ({file_path})")
                except Exception as e:
                    # Log deletion failure
                    error_message = f"Failed to delete entry: {doc_id} - {str(e)}"
                    logger.error(error_message)

        # Bring every to-be-processed document into a clean PENDING state.
        # Two cases are handled here so stale per-attempt metadata never
        # survives into the PENDING wait window (where the WebUI would render
        # last attempt's parse/process timings):
        #   * interrupted docs (PROCESSING/PARSING/FAILED, and the legacy
        #     ANALYZING) are reset
        #     to PENDING, clearing error_msg and resetting metadata to the
        #     enqueue-time directives only;
        #   * docs that are ALREADY PENDING but still carry per-attempt fields
        #     (e.g. reset by an older build that preserved them) are normalised
        #     in place to directives-only.
        # In BOTH cases the cleaned metadata is mirrored back onto the in-memory
        # ``status_doc`` so the downstream parse worker — which no longer scrubs
        # stale keys itself — carries the clean dict forward through
        # ``doc_status_transition_metadata`` at every later transition.
        docs_to_reset = {}
        reset_count = 0
        normalized_count = 0

        # Every document left here has its full_docs row (the rest were taken
        # out above).
        resets: list[tuple[str, DocProcessingStatus, bool]] = []
        for doc_id, status_doc in to_process_docs.items():
            status = getattr(status_doc, "status", None)
            is_interrupted = status in (
                DocStatus.PROCESSING,
                DocStatus.FAILED,
                DocStatus.PARSING,
                DocStatus.ANALYZING,  # legacy, see _INFLIGHT_DOC_STATUSES
            )
            # Only normalise an already-PENDING doc when it actually carries a
            # stale per-attempt field — a precise trigger so unrelated/custom
            # metadata on a clean PENDING is never rewritten or dropped.
            needs_pending_normalize = (
                status == DocStatus.PENDING
                and doc_status_metadata_has_attempt_fields(status_doc)
            )
            if is_interrupted or needs_pending_normalize:
                resets.append((doc_id, status_doc, is_interrupted))

        # full_docs is read only for the documents being reset, and only as the
        # file_path fallback; a fresh upload needs no read at all.
        reset_ids = [doc_id for doc_id, _, _ in resets]
        reset_contents = dict(zip(reset_ids, await self.full_docs.get_by_ids(reset_ids)))

        for doc_id, status_doc, is_interrupted in resets:
            content_data = reset_contents[doc_id]
            preserved_chunks_list, preserved_chunks_count = (
                chunk_fields_from_status_doc(status_doc)
            )
            resolved_file_path = resolve_doc_file_path(
                status_doc=status_doc,
                content_data=content_data,
            )
            # Directives-only metadata: drop per-attempt timing/result fields,
            # keep source_file (legacy source_file_name
            # tolerant).
            reset_metadata = doc_status_reset_metadata(status_doc)
            docs_to_reset[doc_id] = {
                "status": DocStatus.PENDING,
                "content_summary": status_doc.content_summary,
                "content_length": status_doc.content_length,
                "chunks_count": preserved_chunks_count,
                "chunks_list": preserved_chunks_list,
                "created_at": status_doc.created_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": resolved_file_path,
                "track_id": getattr(status_doc, "track_id", ""),
                "content_hash": getattr(status_doc, "content_hash", None),
                "error_msg": "",
                "metadata": reset_metadata,
            }

            # Mirror onto the in-memory status_doc so workers carry it forward.
            status_doc.status = DocStatus.PENDING
            status_doc.file_path = resolved_file_path
            status_doc.metadata = reset_metadata
            if is_interrupted:
                reset_count += 1
            else:
                normalized_count += 1

        # Update doc_status storage if there are documents to reset
        if docs_to_reset:
            await self.doc_status.upsert(docs_to_reset)

            reset_message = (
                f"Reset {reset_count} documents from "
                "PARSING/PROCESSING/FAILED to PENDING status"
                + (
                    f"; normalized {normalized_count} PENDING document(s) "
                    "with stale metadata"
                    if normalized_count
                    else ""
                )
            )
            logger.info(reset_message)

        return to_process_docs

    @staticmethod
    def _format_job_name(
        to_process_docs: dict[str, DocProcessingStatus],
        total_files: int,
    ) -> str:
        """Build the ``job_name`` shown in pipeline_status for one batch."""
        first_doc = next(iter(to_process_docs.values()))
        first_doc_path = first_doc.file_path
        if first_doc_path:
            path_prefix = first_doc_path[:20] + (
                "..." if len(first_doc_path) > 20 else ""
            )
        else:
            path_prefix = "unknown_source"
        return f"{path_prefix}[{total_files} files]"

    # ============================================================
    # Cascading queue workers (Layer 1 -> 2 -> 3)
    # ============================================================

    async def _parse_worker(self, ctx: _BatchRunContext) -> None:
        """Layer 1 worker: consume (doc_id, status_doc) and emit parsed data.

        Marks PARSING, hands the stored content back (nothing is parsed here —
        see :mod:`hybrid_rag.ingestion.parse_stage`), and either
        short-circuits via ``_mark_duplicate_after_parse`` or hands off to
        ``q_process``. Writes FAILED on exception.
        """
        while True:
            item = await ctx.q_parse.get()
            try:
                doc_id_w, status_doc_w = item
                file_path_w = getattr(status_doc_w, "file_path", "unknown_source")
                # Boundary cancellation check: skip parsing the next queued doc
                # without invoking the engine, mark it FAILED with a friendly
                # "User cancelled" message, and let the finally task_done()
                # drain the queue so q.join() in _run_pipeline_batch returns.
                if await self._cancellation_requested(
                    ctx.pipeline_status, ctx.pipeline_status_lock
                ):
                    await self._mark_doc_cancelled_in_stage(
                        doc_id=doc_id_w,
                        status_doc=status_doc_w,
                        file_path=file_path_w,
                        stage_label="parse",
                        pipeline_status=ctx.pipeline_status,
                    )
                    continue
                content_data_w = await self.full_docs.get_by_id(doc_id_w)
                if not content_data_w:
                    raise Exception(
                        f"Document content not found in full_docs for doc_id: {doc_id_w}"
                    )
                if isinstance(status_doc_w.metadata, dict):
                    source_file_w = _read_source_file(status_doc_w.metadata)
                    if source_file_w:
                        # Normalize the legacy ``source_file_name`` onto the new
                        # key in the in-memory status metadata so the carry-over
                        # allowlist (which no longer lists ``source_file_name``)
                        # preserves it through the PARSING upsert below. Without
                        # this, a retry after a parse failure — before full_docs
                        # is rewritten — would no longer resolve the hinted
                        # source file. Idempotent when the new key already exists.
                        status_doc_w.metadata["source_file"] = source_file_w
                        if not _read_source_file(content_data_w):
                            content_data_w["source_file"] = source_file_w
                # Stamp parse_start_time on the in-memory status_doc so
                # carry-over (_DOC_STATUS_METADATA_CARRY_OVER_KEYS) writes it
                # into doc_status here and preserves it across every
                # subsequent state transition for stage-duration analysis.
                if not isinstance(status_doc_w.metadata, dict):
                    status_doc_w.metadata = {}
                # Stale per-attempt fields (*_stage_skipped / stage_seconds)
                # from a prior failed/retried attempt are already
                # scrubbed when the document is brought to PENDING in
                # _validate_and_fix_document_consistency (the single cleanup
                # point), so they are not carried into this PARSING upsert.
                status_doc_w.metadata["parse_start_time"] = int(time.time())
                await self._upsert_doc_status_transition(
                    doc_id=doc_id_w,
                    status=DocStatus.PARSING,
                    status_doc=status_doc_w,
                    file_path=file_path_w,
                )
                log_message = f"Parsing: {doc_id_w}"
                logger.info(log_message)

                # Nothing is parsed here: the content arrived already extracted.
                parsed_data_w = parse_stored_document(
                    doc_id_w, file_path_w, content_data_w
                )

                # Downstream this body feeds content_summary / content_length on
                # doc_status and the duplicate-check length, so strip C0 control
                # chars (incl. NUL, which breaks PostgreSQL text writes) before
                # they can reach doc_status. No-op for already-clean content.
                parsed_data_w["content"] = strip_control_characters(
                    parsed_data_w["content"]
                )

                # No parser runs here, so there is no duration to pair with
                # parse_start_time — record that the stage was a no-op instead.
                # Persisted right below (before the doc enters q_process) so
                # doc_status reflects the parse end immediately; carry-over
                # keeps it visible across every later transition.
                status_doc_w.metadata["parse_stage_skipped"] = True

                # Stamp parse_format at the PARSING upsert instead of deferring
                # to PROCESSING; carry-over preserves it across
                # PROCESSING → PROCESSED.
                status_doc_w.metadata["parse_format"] = FULL_DOCS_FORMAT_RAW

                # ``content_hash`` is computed once at enqueue and never patched
                # afterwards (nothing re-extracts the body), so ``status_doc_w``
                # already carries the value the duplicate check needs.
                if await self._mark_duplicate_after_parse(
                    doc_id=doc_id_w,
                    status_doc=status_doc_w,
                    file_path=file_path_w,
                    content_hash=status_doc_w.content_hash,
                    content_length=len(parsed_data_w.get("content", "")),
                    content_data=content_data_w,
                ):
                    continue

                # Compute content-derived fields here while the parse worker
                # still holds the body, and stamp them on status_doc so they
                # are persisted at the PARSING transition below. The process
                # stage re-reads the body from full_docs by doc_id instead of
                # carrying it through q_process,
                # keeping large documents out of those in-memory buffers. Parse
                # has already persisted the parsed body to full_docs (raw
                # format), so the re-read is guaranteed to find it.
                parsed_content_w = parsed_data_w.get("content", "") or ""
                status_doc_w.content_summary = get_content_summary(parsed_content_w)
                status_doc_w.content_length = len(parsed_content_w)

                # Persist the parse-stage outcome to doc_status now, before the
                # doc waits in q_process, so parse_stage_skipped reflects the
                # actual end of the stage instead of only landing at the
                # PROCESSING transition via carry-over. content_hash is already
                # refreshed and duplicates are filtered out by this point.
                await self._upsert_doc_status_transition(
                    doc_id=doc_id_w,
                    status=DocStatus.PARSING,
                    status_doc=status_doc_w,
                    file_path=file_path_w,
                )

                # q_process carries no body: process_single_document re-reads
                # it from full_docs by doc_id.
                await ctx.q_process.put((doc_id_w, status_doc_w))
            except PipelineCancelledException:
                # Cancellation raised from inside the parse engine (future-
                # proofing — engines don't currently call _raise_if_cancelled,
                # but if they do, route through the same friendly message
                # path as the boundary check above instead of the generic
                # except block below.
                await self._mark_doc_cancelled_in_stage(
                    doc_id=doc_id_w,
                    status_doc=status_doc_w,
                    file_path=getattr(status_doc_w, "file_path", "unknown_source"),
                    stage_label="parse",
                    pipeline_status=ctx.pipeline_status,
                )
            except Exception as e:
                logger.error(f"Parse worker failed: {e}")
                # Mirror the pre-deferral enqueue-time error documents:
                # content_summary (when empty) + metadata error_type /
                # error_stage, so the WebUI list and detail views describe the
                # failure instead of showing a blank row.
                extra_fields_w, metadata_extra_w = doc_status_parse_failure_fields(
                    e, status_doc=status_doc_w
                )
                try:
                    await self._upsert_doc_status_transition(
                        doc_id=doc_id_w,
                        status=DocStatus.FAILED,
                        status_doc=status_doc_w,
                        file_path=getattr(status_doc_w, "file_path", "unknown_source"),
                        extra_fields=extra_fields_w,
                        metadata_extra=metadata_extra_w,
                    )
                except Exception as upsert_err:
                    # The storage backend may be unavailable too (e.g. the same
                    # outage that failed the parse). Don't re-raise — that would
                    # take down the worker — but log so the doc stuck in PARSING
                    # is diagnosable instead of failing silently.
                    logger.error(
                        f"Failed to record FAILED status for {doc_id_w}: {upsert_err}"
                    )
            finally:
                ctx.q_parse.task_done()

    async def _process_worker(self, ctx: _BatchRunContext) -> None:
        """Layer 2 worker: dispatch each ready document to single-doc processing."""
        while True:
            item = await ctx.q_process.get()
            try:
                doc_id_w, status_doc_w = item
                await self.process_single_document(
                    doc_id=doc_id_w,
                    status_doc=status_doc_w,
                    ctx=ctx,
                )
            except Exception as e:
                # process_single_document handles its own per-doc failures; an
                # escape here means even the FAILED-status write failed (e.g.
                # the doc_status backend is down). Do NOT let the worker die —
                # that strands the remaining queued items and hangs
                # q_process.join() forever, wedging the pipeline busy. Route it
                # to the batch-abort path (same flag as IndexFlushError) and
                # keep draining so the batch winds down cleanly. CancelledError
                # is a BaseException, not caught here, so a normal worker
                # cancellation at batch end still propagates.
                logger.error(f"Unhandled error in process worker; aborting batch: {e}")
                logger.error(traceback.format_exc())
                async with ctx.pipeline_status_lock:
                    ctx.pipeline_status["cancellation_requested"] = True
                    ctx.pipeline_status["cancellation_reason"] = "internal_error"
                    ctx.pipeline_status["cancellation_detail"] = (
                        f"process worker unhandled error: {e}"
                    )
            finally:
                ctx.q_process.task_done()

    # ============================================================
    # Single-document state machine
    # ============================================================

    async def process_single_document(
        self,
        *,
        doc_id: str,
        status_doc: DocProcessingStatus,
        ctx: _BatchRunContext,
    ) -> None:
        """Single-document state machine: chunking → KG extraction → merge.

        Always invoked from ``_process_worker`` once ``_parse_worker`` has stored
        the parsed body in full_docs. Drives the PROCESSING → PROCESSED state
        machine, with FAILED fallbacks at both the extract and merge stage
        boundaries.
        """
        # Dokumen FAILED lama bisa ikut terangkut dalam batch yang sama; progresnya
        # harus mendarat di stream miliknya sendiri, bukan milik dokumen pemicu.
        # ``workspace`` ikut karena ``doc_id`` itu hash isi berkas: tanpa itu, tenant
        # lain yang mengunggah berkas yang sama berbagi satu kanal progres.
        progress.current_doc_id.set(doc_id)
        progress.current_workspace.set(self.workspace)
        file_path = resolve_doc_file_path(status_doc=status_doc)
        current_file_number = 0
        file_extraction_stage_ok = False
        run = _DocRun(process_start_time=int(time.time()))
        content_data: dict[str, Any] | None = None
        chunk_results: list = []
        mm_chunks: dict[str, Any] = {}

        def get_failed_chunk_snapshot() -> tuple[list[str], int]:
            if run.chunks:
                chunk_ids = list(run.chunks.keys())
                return chunk_ids, len(chunk_ids)
            return chunk_fields_from_status_doc(status_doc)

        async with _ingestion_turn():
            try:
                # Resolve file_path from full_docs before honoring a queued
                # cancellation so corrupted doc_status placeholders do not
                # get written back again during retry/cancel flows.
                content_data = await self.full_docs.get_by_id(doc_id)
                if content_data:
                    file_path = resolve_doc_file_path(
                        status_doc=status_doc,
                        content_data=content_data,
                    )
                    status_doc.file_path = file_path

                # Check for cancellation before starting document processing.
                # file_path is resolved before this check so queued documents
                # do not lose their source path on early cancellation.
                await self._raise_if_cancelled(
                    ctx.pipeline_status, ctx.pipeline_status_lock
                )

                async with ctx.pipeline_status_lock:
                    ctx.processed_count += 1
                    current_file_number = ctx.processed_count
                    ctx.pipeline_status["cur_batch"] = ctx.processed_count

                    log_message = (
                        f"Extracting stage {current_file_number}/"
                        f"{ctx.total_files}: {file_path}"
                    )
                    logger.info(log_message)
                    logger.info(f"Processing d-id: {doc_id}")

                # Resume guard: if content was already extracted, purge stale
                # chunks + KG before rebuilding.
                await self._purge_stale_extraction_if_resuming(
                    doc_id=doc_id,
                    status_doc=status_doc,
                    file_path=file_path,
                    content_data=content_data,
                )

                # Text and multimodal halves run side by side: neither reads the
                # other's output, so no model call sees a different input. A
                # failure in either cancels the other at once (structured
                # concurrency) instead of letting it run on for a document that
                # is already failed. Image sources are resolved first so a
                # missing image fails the document before any model call.
                # ``extract_gate`` is shared by both extractions so the
                # extraction model never sees more concurrent calls than one
                # extract_entities call would allow on its own.
                multimodal_items = resolve_image_sources(
                    (content_data or {}).get("multimodal_items") or []
                )
                extract_gate = asyncio.Semaphore(max(1, self.llm_model_max_async))
                run.t_start = time.monotonic()

                async def multimodal_side() -> tuple[dict, list]:
                    try:
                        return await self._analyze_resolved_items(
                            multimodal_items, file_path, doc_id,
                            extract_semaphore=extract_gate,
                        )
                    finally:
                        run.mark("multimodal")

                try:
                    async with asyncio.TaskGroup() as group:
                        text_side = group.create_task(
                            self._process_text_side(
                                doc_id=doc_id,
                                status_doc=status_doc,
                                file_path=file_path,
                                content_data=content_data,
                                ctx=ctx,
                                run=run,
                                extract_gate=extract_gate,
                                has_multimodal=bool(multimodal_items),
                            )
                        )
                        mm_side = (
                            group.create_task(multimodal_side())
                            if multimodal_items
                            else None
                        )
                except* Exception as failures:
                    raise_first_failure(failures)

                # Multimodal results join after the text side's source_id
                # redirect: their chunk ids are already final and must not be
                # remapped. Text first, as always, so the merge input is unchanged.
                chunk_results = text_side.result()
                if mm_side is not None:
                    mm_chunks, mm_chunk_results = mm_side.result()
                    chunk_results.extend(mm_chunk_results)
                file_extraction_stage_ok = True

            except Exception as e:
                await self._finalize_doc_failure(
                    doc_id=doc_id,
                    status_doc=status_doc,
                    file_path=file_path,
                    error=e,
                    stage_label="extract",
                    current_file_number=current_file_number,
                    total_files=ctx.total_files,
                    failed_chunks_snapshot=get_failed_chunk_snapshot(),
                    pending_tasks=[],
                    metadata_extra={
                        "process_start_time": run.process_start_time,
                        "process_end_time": int(time.time()),
                    },
                    pipeline_status=ctx.pipeline_status,
                )

            # Concurrency is controlled by keyed lock for individual
            # entities and relationships.
            if file_extraction_stage_ok:
                try:
                    await self._raise_if_cancelled(
                        ctx.pipeline_status, ctx.pipeline_status_lock
                    )

                    # Merge the nodes/edges extracted from the parent chunks
                    # into the graph (source_ids already redirected to children).
                    progress.stage("merge")
                    await merge_nodes_and_edges(
                        chunk_results=chunk_results,
                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                        entity_vdb=self.entities_vdb,
                        relationships_vdb=self.relationships_vdb,
                        global_config=self._build_global_config(),
                        full_entities_storage=self.full_entities,
                        full_relations_storage=self.full_relations,
                        doc_id=doc_id,
                        pipeline_status=ctx.pipeline_status,
                        pipeline_status_lock=ctx.pipeline_status_lock,
                        llm_response_cache=self.llm_response_cache,
                        entity_chunks_storage=self.entity_chunks,
                        relation_chunks_storage=self.relation_chunks,
                        current_file_number=current_file_number,
                        total_files=ctx.total_files,
                        file_path=file_path,
                    )
                    run.mark("merge")

                    # If another in-flight document already triggered an abort
                    # (e.g. a storage flush error set cancellation_requested),
                    # do not mark PROCESSED or re-run _insert_done here: the
                    # shared flush buffer is being torn down, so re-flushing
                    # would just re-raise the same error. Bail out as cancelled
                    # so this document is FAILED and retried on the next run.
                    await self._raise_if_cancelled(
                        ctx.pipeline_status, ctx.pipeline_status_lock
                    )

                    # Flush (embedding + vector/KV writes) before PROCESSED, so a
                    # PROCESSED document is searchable and a flush failure leaves
                    # it FAILED without ever reporting PROCESSED.
                    await self._insert_done()
                    run.mark("embed")
                    logger.info(f"[stage_seconds] {doc_id}: {run.stage_seconds}")

                    process_end_time = int(time.time())
                    await self._upsert_doc_status_transition(
                        doc_id=doc_id,
                        status=DocStatus.PROCESSED,
                        status_doc=status_doc,
                        file_path=file_path,
                        extra_fields={
                            "chunks_count": len(run.chunks) + len(mm_chunks),
                            "chunks_list": list(run.chunks.keys())
                            + list(mm_chunks.keys()),
                        },
                        metadata_extra={
                            "process_start_time": run.process_start_time,
                            "process_end_time": process_end_time,
                            "stage_seconds": run.stage_seconds,
                            **run.extraction_meta,
                        },
                    )

                    log_message = (
                        f"Completed processing file "
                        f"{current_file_number}/{ctx.total_files}: "
                        f"{file_path}"
                    )
                    logger.info(log_message)

                except Exception as e:
                    # A storage flush failure (raised by _insert_done) is not
                    # attributable to this document: index_done_callback flushes
                    # a buffer shared across concurrently-processed files. We
                    # cannot tell whose record failed, so continuing risks
                    # marking other in-flight files PROCESSED with missing data.
                    # Abort the whole batch via the cooperative cancellation
                    # flag, tagging it as an internal error with the driver name
                    # and root cause so it is distinguishable from a user cancel.
                    if isinstance(e, IndexFlushError):
                        async with ctx.pipeline_status_lock:
                            ctx.pipeline_status["cancellation_requested"] = True
                            ctx.pipeline_status["cancellation_reason"] = (
                                "internal_error"
                            )
                            ctx.pipeline_status["cancellation_detail"] = (
                                f"{e.storage_name}[{e.namespace}]: {e.__cause__}"
                            )
                        logger.error(
                            f"Aborting pipeline batch due to storage flush error: {e}"
                        )
                    await self._finalize_doc_failure(
                        doc_id=doc_id,
                        status_doc=status_doc,
                        file_path=file_path,
                        error=e,
                        stage_label="merge",
                        current_file_number=current_file_number,
                        total_files=ctx.total_files,
                        failed_chunks_snapshot=get_failed_chunk_snapshot(),
                        pending_tasks=[],
                        metadata_extra={
                            "process_start_time": run.process_start_time,
                            "process_end_time": int(time.time()),
                            **run.extraction_meta,
                        },
                        pipeline_status=ctx.pipeline_status,
                    )

    async def _process_text_side(
        self,
        *,
        doc_id: str,
        status_doc: DocProcessingStatus,
        file_path: str,
        content_data: dict[str, Any] | None,
        ctx: _BatchRunContext,
        run: _DocRun,
        extract_gate: asyncio.Semaphore,
        has_multimodal: bool,
    ) -> list:
        """The text half of a document: chunk → contextualize → store → extract.

        Runs beside the multimodal half in ``process_single_document``. Returns
        the extraction results with source_ids already redirected from parent
        to child chunks, ready to be merged.
        """
        # The parsed body is not carried through q_process (it would pin large
        # documents in memory); it is re-read from full_docs.
        content = (content_data or {}).get("content") or ""

        # Chunking is always contextualized parent-child: split into
        # large parent chunks (enough context for entity extraction)
        # and small child chunks (precise for retrieval), then rewrite
        # each child to carry its parent's context. Only children are
        # persisted; parents feed extraction and are then discarded,
        # with source_ids redirected onto the children further below.
        from hybrid_rag.ingestion.chunking import (
            chunking_parent_child,
            contextualize_children,
            redirect_source_ids,
        )
        from hybrid_rag.ingestion.chunking.parent_child import (
            PARENT_CHUNK_SIZE,
            PARENT_CHUNK_OVERLAP,
            CHILD_CHUNK_SIZE,
            CHILD_CHUNK_OVERLAP,
        )

        # Recorded on ``doc_status.metadata['chunk_opts']`` via
        # ``extraction_meta`` so admin/list APIs can see the chunk
        # sizes a document was actually built with. Parent-child reads
        # its sizes from the environment, so there is nothing
        # per-document to resolve here.
        chunk_opts_str: str = (
            f"parent={PARENT_CHUNK_SIZE}/{PARENT_CHUNK_OVERLAP}, "
            f"child={CHILD_CHUNK_SIZE}/{CHILD_CHUNK_OVERLAP}"
        )

        progress.stage("chunk")
        logger.info(f"Chunking parent-child, doc_id: {doc_id}")
        page_offsets = ((content_data or {}).get("chunk_options") or {}).get("page_offsets")
        parent_result, child_result = chunking_parent_child(
            self.tokenizer, content, page_offsets=page_offsets
        )

        # === Parent-Child storage & extraction === #
        # Parent dipakai untuk ekstraksi entitas, dan — bila ``save_parent`` —
        # ikut disimpan ke ``text_chunks`` sebagai bukti utuh yang bisa
        # dikembalikan retrieval (``QueryParam.return_parent``). Tidak pernah
        # masuk ``chunks_vdb``: parent tidak di-embed dan tidak pernah dicari.
        save_parent = (getattr(status_doc, "metadata", None) or {}).get(
            "save_parent", True
        )
        for _p in parent_result:
            _idx = _p["chunk_order_index"]
            _p["chunk_id"] = f"parent-{_idx:03d}"
            # Negatif dengan sengaja: sitasi dokumen dibuka lewat
            # ``chunk_order_index = <angka>`` yang selalu >= 0 (lihat
            # agents/auditor/tools/retrievers.lookup), jadi baris parent tidak
            # akan pernah ikut terpilih di sana dan sitasi tetap milik child.
            _p["chunk_order_index"] = -1 - _idx
        _parent_chunks_dict = build_chunks_dict_from_chunking_result(parent_result, doc_id=doc_id, file_path=file_path)

        # Extraction reads only the parents, so it starts now and runs beside
        # contextualize instead of waiting for it. Parents are stored first:
        # extraction records its LLM cache keys onto their text_chunks rows
        # (update_chunk_cache_list skips a row that does not exist yet), and a
        # later upsert of the parent rows would wipe that list again.
        if save_parent and _parent_chunks_dict:
            await self.text_chunks.upsert(_parent_chunks_dict)
        extraction = asyncio.create_task(
            self._process_extract_entities(
                _parent_chunks_dict,
                ctx.pipeline_status,
                ctx.pipeline_status_lock,
                ontology=(getattr(status_doc, "metadata", None) or {}).get("kg_ontology"),
                semaphore=extract_gate,
            )
        )
        primed: list[str] = []

        def prime_child(child: dict[str, Any]) -> None:
            primed.append(child["content"])
            self.chunks_vdb.prime(child["content"])

        try:
            # Contextualize children using parent context (LLM). Runs on
            # its own model and concurrency cap — see chunking/parent_child.
            progress.stage("contextualize")
            # Each child is embedded as soon as its context is written, not
            # after the slowest child: the embedding queue is the bottleneck
            # and would otherwise sit idle through contextualize.
            chunking_result = await contextualize_children(
                parent_result, child_result, on_contextualized=prime_child
            )
            if inspect.isawaitable(chunking_result):
                chunking_result = await chunking_result
            if not isinstance(chunking_result, (list, tuple)):
                raise TypeError(
                    f"chunking_func must return a list or tuple of dicts, "
                    f"got {type(chunking_result)}"
                )
            run.mark("contextualize")

            run.extraction_meta = {
                "parse_format": FULL_DOCS_FORMAT_RAW,
                "chunk_method": "contextualized_parent_child",
                # Mirrors the chunking start log line (params portion only,
                # without the strategy prefix or file path) so admins can
                # see the actual chunker params used.  Carried across
                # transitions via ``_DOC_STATUS_METADATA_CARRY_OVER_KEYS``.
                "chunk_opts": chunk_opts_str,
            }

            # Final hard guard before embedding: split any oversize
            # chunk while preserving heading hierarchy metadata.
            if (
                self.embedding_token_limit is not None
                and self.embedding_token_limit > 0
            ):
                original_chunk_count = len(chunking_result)
                chunking_result = enforce_chunk_token_limit_before_embedding(
                    chunking_result=chunking_result,
                    tokenizer=self.tokenizer,
                    max_tokens=self.embedding_token_limit,
                )
                if len(chunking_result) != original_chunk_count:
                    logger.info(
                        "Applied hard fallback split before embedding for "
                        f"d-id: {doc_id}, chunks {original_chunk_count} -> {len(chunking_result)} "
                        f"(limit={self.embedding_token_limit})"
                    )
                    # Compact "pre -> post" summary mirrors the log
                    # middle segment.  Field is only present when a
                    # hard split actually occurred, so its presence
                    # alone signals the trigger.
                    run.extraction_meta["hard_fallback_split"] = (
                        f"{original_chunk_count} -> {len(chunking_result)}"
                    )


            chunks = build_chunks_dict_from_chunking_result(chunking_result, doc_id=doc_id, file_path=file_path)
            run.chunks = chunks

            if not chunks:
                logger.warning("No document chunks to process")

            run.process_start_time = int(time.time())

            await self._raise_if_cancelled(
                ctx.pipeline_status, ctx.pipeline_status_lock
            )

            # `chunks` here = child chunks (contextualized) — these get stored
            _child_chunks_dict = chunks

            # Tali parent→child dibangun di sini, bukan di chunker: ini titik
            # pertama setelah ``enforce_chunk_token_limit_before_embedding``
            # memecah child, dan ``dict(dp)`` di sana sudah membawa
            # ``parent_index`` ikut ke tiap pecahan. Peta berbasis posisi list
            # dari chunker akan meleset persis di sini.
            _pc_map: dict[str, list[str]] = defaultdict(list)
            for _ckey, _c in _child_chunks_dict.items():
                _pi = _c.get("parent_index")
                if _pi is None:
                    continue  # chunk multimodal: tidak lahir dari parent
                _pkey = f"{doc_id}-parent-{_pi:03d}"
                _pc_map[_pkey].append(_ckey)
                if save_parent:
                    _c.setdefault("sidecar", {})["parent_id"] = _pkey
            logger.info(
                f"[parent-child] {len(_pc_map)} parent -> "
                f"{sum(len(v) for v in _pc_map.values())} child (save_parent={save_parent})"
            )

            # The multimodal half is already running alongside this one. Its
            # stage marker stays at its old position so the progress bar
            # keeps following STAGES order.
            if has_multimodal:
                progress.stage("multimodal")

            # Text chunks only: multimodal ids are not known yet. The
            # PROCESSED write below records both halves; a crash in between
            # is covered by the resume purge, which also looks chunks up by
            # full_doc_id.
            doc_status_write = (
                self._upsert_doc_status_transition(
                    doc_id=doc_id,
                    status=DocStatus.PROCESSING,
                    status_doc=status_doc,
                    file_path=file_path,
                    extra_fields={
                        "chunks_count": len(_child_chunks_dict),
                        "chunks_list": list(_child_chunks_dict.keys()),
                    },
                    metadata_extra={
                        "process_start_time": run.process_start_time,
                        **run.extraction_meta,
                    },
                )
            )
            if not has_multimodal:
                progress.skip("multimodal", "Tidak ada gambar/tabel yang perlu dibaca")
            progress.stage("store_chunks")
            writes = [
                doc_status_write,
                self.chunks_vdb.upsert(_child_chunks_dict),
                self.text_chunks.upsert(_child_chunks_dict),
            ]
            await asyncio.gather(*writes)
            # The child chunks are final now: embed them while extraction and merge
            # run instead of after them. The flush embeds whatever is still missing.
            run.early_embed = asyncio.create_task(self._embed_chunks_early(primed))

            # Stage 2: extraction from PARENT chunks (already running), then
            # redirect source_ids
            progress.stage("extract")
            logger.info("Extracting Entity-Relationship")
            chunk_results = await extraction
        except BaseException:
            # Contextualize or storing failed (or we were cancelled): the
            # document fails, so its extraction must not run on.
            extraction.cancel()
            self.chunks_vdb.discard_primed(primed)
            with contextlib.suppress(BaseException):
                await extraction
            raise
        run.mark("extract_text")

        # ★ CRUCIAL: redirect source_ids from parent → child
        redirect_source_ids(chunk_results, _pc_map)
        return chunk_results

    async def _embed_chunks_early(self, primed: list[str]) -> None:
        """Best effort: a failure here only means the flush embeds these chunks."""
        try:
            await self.chunks_vdb.embed_pending()
        except Exception as e:
            logger.warning(f"Early chunk embedding failed, the flush will retry: {e}")
        finally:
            # Primed vectors no pending chunk took (a child re-split by the
            # token guard has different text) are not needed any more.
            self.chunks_vdb.discard_primed(primed)

    async def _purge_stale_extraction_if_resuming(
        self,
        *,
        doc_id: str,
        status_doc: DocProcessingStatus,
        file_path: str,
        content_data: dict[str, Any] | None,
    ) -> None:
        """If the document already has extracted content, purge stale chunks
        and KG contributions before re-running chunking + entity extraction.

        Mutates ``status_doc.chunks_list`` / ``chunks_count`` to reflect the
        purge so subsequent state-machine upserts don't write back stale IDs.
        """
        content_already_extracted = isinstance(content_data, dict) and bool(
            (content_data.get("content") or "").strip()
        )
        if not content_already_extracted:
            return

        # Order-preserving dedup; keep a list so it satisfies the storage delete
        # contract (``delete(ids: list[str])``) when passed down to purge.
        # ``chunks_list`` alone is not enough: a FAILED run records only its text
        # chunks, and multimodal chunks are stored before chunks_list mentions
        # them — so also ask the chunk tables by ``full_doc_id``, like deletion.
        stored_chunk_ids = list(
            dict.fromkeys(
                chunk_id
                for chunk_id in [
                    *(status_doc.chunks_list or []),
                    *await self._chunk_ids_from_storage(doc_id),
                ]
                if isinstance(chunk_id, str) and chunk_id
            )
        )
        if not stored_chunk_ids:
            return

        log_message = (
            f"[resume] {doc_id}: purging "
            f"{len(stored_chunk_ids)} chunk(s) and "
            f"associated KG entries from a previous run before rebuilding"
        )
        logger.info(log_message)
        await self._purge_doc_chunks_and_kg(
            doc_id,
            stored_chunk_ids,
        )
        # The status_doc carries chunks_list / chunks_count from the prior
        # run; clear them so subsequent state-machine upserts don't write
        # back stale IDs.
        status_doc.chunks_list = []
        status_doc.chunks_count = 0

    # ============================================================
    # doc_status state-machine helpers (shared by all layers)
    # ============================================================

    async def _upsert_doc_status_transition(
        self,
        doc_id: str,
        status: DocStatus,
        status_doc: DocProcessingStatus,
        file_path: str,
        *,
        extra_fields: dict[str, Any] | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        """Single source of truth for doc_status state-transition upserts.

        Mirrors the field set used at every PARSING / PROCESSING /
        PROCESSED / FAILED transition.  ``extra_fields`` carries
        ``chunks_count`` / ``chunks_list`` / ``error_msg``; ``metadata_extra``
        is forwarded to ``doc_status_transition_metadata`` so carry-over
        fields (e.g. ``source_file``) survive every state change.
        """
        payload: dict[str, Any] = {
            "status": status,
            "content_summary": status_doc.content_summary,
            "content_length": status_doc.content_length,
            "created_at": status_doc.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "file_path": file_path,
            "track_id": status_doc.track_id,
            "content_hash": status_doc.content_hash,
            "metadata": doc_status_transition_metadata(
                status_doc, extra=metadata_extra
            ),
        }
        if extra_fields:
            payload.update(extra_fields)
        await self.doc_status.upsert({doc_id: payload})

    async def _raise_if_cancelled(
        self,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> None:
        """Raise ``PipelineCancelledException`` if the user has requested cancel."""
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled")

    @staticmethod
    def _cancellation_label(pipeline_status: dict) -> str:
        """Human-readable cancel cause: internal error (with detail) vs user.

        Callers building cancellation messages use this so an internal abort
        (e.g. a storage flush failure) is not mislabeled as a user cancel.
        """
        if pipeline_status.get("cancellation_reason") == "internal_error":
            detail = pipeline_status.get("cancellation_detail") or "unknown"
            return f"Cancelled by internal error: {detail}"
        return "User cancelled"

    @staticmethod
    def _internal_halt_message(label: str) -> str:
        """Actionable halt message for an internal-error abort.

        Shared by the loop-top cancellation handler and the finally cleanup so
        the same wording surfaces whichever exit path the batch takes.
        """
        return (
            f"Pipeline halted on internal storage error ({label}). Resolve the "
            f"storage issue and restart processing; affected documents remain "
            f"queued (PENDING/FAILED)."
        )

    async def _cancellation_requested(
        self,
        pipeline_status: dict,
        pipeline_status_lock,
    ) -> bool:
        """Read-only cancellation check.

        Use this when a worker wants to branch on the flag (e.g. drain a queue
        item) instead of raising. Callers that prefer the exception style
        should use :meth:`_raise_if_cancelled` instead.
        """
        async with pipeline_status_lock:
            return bool(pipeline_status.get("cancellation_requested", False))

    async def _mark_doc_cancelled_in_stage(
        self,
        *,
        doc_id: str,
        status_doc: DocProcessingStatus,
        file_path: str,
        stage_label: str,
        pipeline_status: dict,
    ) -> None:
        """Mark a queued document FAILED with a 'User cancelled' message.

        Used by the PARSE and ANALYZE workers, which do not have the
        chunks-snapshot / pending-tasks bookkeeping that
        :meth:`_finalize_doc_failure` carries for the PROCESS stage. Also
        flushes the LLM response cache so any cache_ids written by completed
        sibling tasks (e.g. successful multimodal items inside a doc that is
        being cancelled) survive a server restart.
        """
        error_msg = (
            f"{self._cancellation_label(pipeline_status)} during "
            f"{stage_label}: {file_path}"
        )
        logger.warning(error_msg)
        if self.llm_response_cache:
            try:
                await self.llm_response_cache.index_done_callback()
            except Exception as persist_error:
                logger.error(f"Failed to persist LLM cache: {persist_error}")
        try:
            await self._upsert_doc_status_transition(
                doc_id=doc_id,
                status=DocStatus.FAILED,
                status_doc=status_doc,
                file_path=file_path,
                extra_fields={"error_msg": error_msg},
            )
        except Exception as exc:
            logger.error(f"Failed to mark cancelled doc {doc_id} as FAILED: {exc}")

    async def _finalize_doc_failure(
        self,
        *,
        doc_id: str,
        status_doc: DocProcessingStatus,
        file_path: str,
        error: BaseException,
        stage_label: str,
        current_file_number: int,
        total_files: int,
        failed_chunks_snapshot: tuple[list[str], int],
        pending_tasks: list[asyncio.Task],
        metadata_extra: dict[str, Any],
        pipeline_status: dict,
    ) -> None:
        """Common epilogue for an extract / merge stage failure.

        Logs the error (or cancellation), cancels any pending stage tasks,
        flushes the LLM response cache, and writes a FAILED status row that
        preserves the failed chunks snapshot and processing-time metadata.
        """
        if isinstance(error, PipelineCancelledException):
            cancel_label = self._cancellation_label(pipeline_status)
            # The cancel exceptions raised by the merge/summary stages hardcode a
            # generic "User cancelled during <stage>" message. When the batch was
            # actually aborted by an internal error (e.g. a storage outage), that
            # mislabels the cause. Swap the generic prefix for the reason-aware
            # label so doc_status records "Cancelled by internal error: <detail>
            # during <stage>" rather than "User cancelled during <stage>".
            raw = str(error)
            if raw.startswith("User cancelled"):
                doc_error_msg = f"{cancel_label}{raw[len('User cancelled') :]}"
            elif raw:
                doc_error_msg = f"{cancel_label}: {raw}"
            else:
                doc_error_msg = cancel_label
            if stage_label == "merge":
                error_msg = (
                    f"{cancel_label} during merge {current_file_number}/"
                    f"{total_files}: {file_path}"
                )
            else:
                error_msg = (
                    f"{cancel_label} {current_file_number}/{total_files}: {file_path}"
                )
            logger.warning(error_msg)
        else:
            doc_error_msg = str(error)
            logger.error(traceback.format_exc())
            if stage_label == "merge":
                error_msg = (
                    f"Merging stage failed in document "
                    f"{current_file_number}/{total_files}: {file_path}"
                )
            else:
                error_msg = (
                    f"Failed to extract document "
                    f"{current_file_number}/{total_files}: {file_path}"
                )
            logger.error(error_msg)

        for task in pending_tasks:
            if task and not task.done():
                task.cancel()

        if self.llm_response_cache:
            try:
                await self.llm_response_cache.index_done_callback()
            except Exception as persist_error:
                logger.error(f"Failed to persist LLM cache: {persist_error}")

        failed_chunks_list, failed_chunks_count = failed_chunks_snapshot
        await self._upsert_doc_status_transition(
            doc_id=doc_id,
            status=DocStatus.FAILED,
            status_doc=status_doc,
            file_path=file_path,
            extra_fields={
                "error_msg": doc_error_msg,
                "chunks_count": failed_chunks_count,
                "chunks_list": failed_chunks_list,
            },
            metadata_extra=metadata_extra,
        )

    # ============================================================
    # Parser internals
    # ============================================================

    async def _mark_duplicate_after_parse(
        self,
        doc_id: str,
        status_doc: DocProcessingStatus,
        file_path: str,
        content_hash: str | None,
        content_length: int,
        content_data: dict[str, Any] | None = None,
    ) -> bool:
        """Mark post-parse content duplicates and stop further processing."""
        if not content_hash:
            return False

        match = await get_duplicate_doc_by_content_hash(
            self.doc_status, content_hash, doc_id
        )
        if not match:
            return False

        original_doc_id, original_doc = match
        original_track_id = doc_status_field(original_doc, "track_id", "")
        original_status = doc_status_field(original_doc, "status", "unknown")
        now = datetime.now(timezone.utc).isoformat()
        message = (
            "Identical content already exists under another filename. "
            f"Original doc_id: {original_doc_id}, Status: {original_status}"
        )

        await self.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.FAILED,
                    "content_summary": (
                        f"[DUPLICATE:content_hash] Original document: {original_doc_id}"
                    ),
                    "content_length": content_length,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": status_doc.created_at,
                    "updated_at": now,
                    "file_path": file_path,
                    "track_id": status_doc.track_id,
                    "content_hash": content_hash,
                    "error_msg": message,
                    "metadata": doc_status_transition_metadata(
                        status_doc,
                        extra={
                            "is_duplicate": True,
                            "duplicate_kind": "content_hash",
                            "original_doc_id": original_doc_id,
                            "original_track_id": original_track_id,
                        },
                    ),
                }
            }
        )
        try:
            await self.full_docs.delete([doc_id])
            await self.full_docs.index_done_callback()
        except Exception as e:
            logger.warning(f"Failed to remove duplicate full_docs entry {doc_id}: {e}")

        # Upstream also moved the source file into an ``__parsed__/`` archive
        # here, so a directory rescan would not re-ingest it. Documents reach
        # this pipeline from object storage, never from a watched directory, so
        # there is no file on disk to archive.
        warning = f"Duplicate content skipped after parsing: {file_path}"
        logger.warning(warning)
        return True
