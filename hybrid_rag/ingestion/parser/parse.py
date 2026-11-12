"""The one door into parsing: ``parse_file`` picks the parser backend and yields its content list.

Two backends, one contract (``pdf/contract.py``): text items stamped with ``layout``, every
``img_path`` a hosted URL. What differs is only where the file is parsed:

* ``builtin``  — the parsers of this package, in this process (``router.py``).
* ``external`` — the parser server; this process only uploads the file and waits (``external.py``).

The backend comes from ``PARSER_BACKEND``; ``parser_backend`` overrides it for one call. A workspace
should stay on one backend: the two read a page with different engines, the text is not identical,
and a document's id is derived from its content.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from hybrid_rag.config import PARSER_BACKENDS, settings

#: What both backends accept. Kept here, not derived from ``router.PARSERS``, so reading it does not
#: import the builtin parsers and their dependencies.
SUPPORTED_INGEST_EXTENSIONS = frozenset({
    "pdf", "docx", "pptx", "doc", "odt", "rtf", "xlsx", "jpg", "jpeg", "png", "md", "markdown", "txt", "json",
})


@asynccontextmanager
async def parse_file(
    file_path: str | Path,
    entity_registry: str | None = None,
    parser_backend: str | None = None,
) -> AsyncIterator[list[dict]]:
    """Parse one file into a content list, ready for ``rag.insert_content_list``.

    ``parser_backend``: ``"builtin"`` or ``"external"``; None follows ``PARSER_BACKEND``.
    ``entity_registry``: pass the same registry as to ``insert_content_list``. Only the builtin
    backend uses it (acronyms for slide transcription); the parser server holds no domain knowledge.
    """
    backend = parser_backend or settings.parser_backend
    if backend not in PARSER_BACKENDS:
        raise ValueError(f"parser_backend harus salah satu dari {PARSER_BACKENDS}, bukan {backend!r}")
    # Imported on use: a deployment on the external backend never loads the builtin parsers.
    if backend == "external":
        from hybrid_rag.ingestion.parser import external as implementation
    else:
        from hybrid_rag.ingestion.parser import router as implementation
    async with implementation.parse_file(file_path, entity_registry) as content_list:
        yield content_list
