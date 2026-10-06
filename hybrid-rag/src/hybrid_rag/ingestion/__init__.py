"""Ingestion — file in, indexed knowledge out; and deletion, its inverse.

    rag.py          IngestionRAG: the storages plus the mixins below
    factory.py      build an IngestionRAG from settings
    workspace.py    the write pool: one IngestionRAG per workspace
    pipeline.py     the outer boundary: parse_file() (file → content list)
    insert.py       _IngestionMixin — insert_content_list, the stage orchestrator
    worker.py       _PipelineMixin — the enqueue / parse / process loop
    parse_stage.py  the pipeline's parse stage (a no-op: content arrives parsed)

The stages themselves:

    parser/           AWS Textract turns a file into a content list
    multimodal/       image / table / equation captioning (VLM)
    chunking/         contextualized parent-child chunking (the text stage)
    knowledge_graph/  entity/relation extraction and merge; embeddings and BM25
                      happen in ``hybrid_rag.storage``
    deletion/         remove a document and rebuild what it shared

No module here calls back into the orchestrator — stages are called, they do
not call.
"""


# Public API, resolved on first access so importing a submodule stays cheap.
_EXPORTS = {
    "IngestionRAG": "hybrid_rag.ingestion.rag",
    "build_rag": "hybrid_rag.ingestion.factory",
    "build_pipeline": "hybrid_rag.ingestion.factory",
    "workspace_pool": "hybrid_rag.ingestion.workspace",
    "parse_file": "hybrid_rag.ingestion.pipeline",
    "SUPPORTED_INGEST_EXTENSIONS": "hybrid_rag.ingestion.pipeline",
    "delete_document": "hybrid_rag.ingestion.deletion.pipeline",
    "convert_office_to_pdf": "hybrid_rag.ingestion.parser.office_to_pdf",
    "DocStatus": "hybrid_rag.core.base",
    "DeletionResult": "hybrid_rag.core.base",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)
