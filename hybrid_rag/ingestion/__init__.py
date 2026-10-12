"""Ingestion — file in, indexed knowledge out; and deletion, its inverse."""

_EXPORTS = {
    "IngestionRAG": "hybrid_rag.ingestion.rag",
    "build_rag": "hybrid_rag.ingestion.factory",
    "build_pipeline": "hybrid_rag.ingestion.factory",
    "workspace_pool": "hybrid_rag.ingestion.workspace",
    "parse_file": "hybrid_rag.ingestion.pipeline",
    "SUPPORTED_INGEST_EXTENSIONS": "hybrid_rag.ingestion.pipeline",
    "delete_document": "hybrid_rag.ingestion.deletion.pipeline",
    "DocStatus": "hybrid_rag.core.base",
    "DeletionResult": "hybrid_rag.core.base",
}
__all__ = list(_EXPORTS)

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(_EXPORTS[name]), name)