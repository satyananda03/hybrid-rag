"""The pipeline's parse stage — a no-op by construction.

Documents reach this pipeline already parsed. ``ingestion/parser/`` (AWS
Textract) turns a file into a content list, and ``ingestion/insert.py`` hands
the text half to ``apipeline_enqueue_documents``. By the time a document enters
the parse stage there is nothing left to parse: the stage exists only to hand
the stored content back and stamp the bookkeeping the later stages read.

Upstream this was an engine registry (native / legacy / mineru / docling) plus
routing to pick one per document from file suffixes, filename hints and env
rules. Parsing now happens outside this module (``ingestion/parser/amazon_textract.py``),
so the registry, the per-engine concurrency knobs and the ``parse_engine``
bookkeeping are all gone.
"""


from typing import Any

from hybrid_rag.core.constants import FULL_DOCS_FORMAT_RAW


def parse_stored_document(
    doc_id: str,
    file_path: str,
    content_data: dict[str, Any],
) -> dict[str, Any]:
    """Return the document's already-parsed content as the parse-stage payload.

    Content is always supplied verbatim at insert time (``insert_content_list``
    stores it as ``FULL_DOCS_FORMAT_RAW``), so nothing is read from or written
    to disk here and ``parse_stage_skipped`` is always set — there is no parse
    duration to record.
    """
    return {
        "doc_id": doc_id,
        "file_path": file_path,
        "parse_format": FULL_DOCS_FORMAT_RAW,
        "content": content_data.get("content") or "",
        # The image / table / equation blocks separated out at enqueue. Carried
        # through so process_single_document can caption them without re-parsing.
        "multimodal_items": content_data.get("multimodal_items") or [],
        "parse_stage_skipped": True,
    }
