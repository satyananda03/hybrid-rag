"""Input for tuning a detector by hand: a document's parsed content, parsed only once.

Detectors run as plain scripts, where ``hybrid_rag`` is not importable, so the repository
root joins ``sys.path`` only when a PDF really has to be parsed.
"""

import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / ".cache" / "legal_debug"


def pdf_argument() -> Path:
    """The document named on the command line."""
    if len(sys.argv) != 2:
        sys.exit(f"Pakai: python3 {sys.argv[0]} <file.pdf atau content_list.json>")
    return Path(sys.argv[1])


def load_content_list(path: Path) -> list[dict]:
    """A content list ``.json`` as is; a PDF through the parser, read from the cache after the first time."""
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    cached = CACHE / f"{hashlib.md5(path.read_bytes()).hexdigest()}.json"
    if not cached.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(_parse(path), ensure_ascii=False, default=str), encoding="utf-8")
    return json.loads(cached.read_text(encoding="utf-8"))


def load_text(path: Path) -> str:
    return text_of(load_content_list(path))


def text_of(content_list: list[dict]) -> str:
    """The text blocks joined as ``separate_content`` joins them."""
    return "\n\n".join(item["text"] for item in content_list if item.get("type", "text") == "text" and item.get("text", "").strip())


def _parse(path: Path) -> list[dict]:
    """The parser the pipeline uses; figures stay local, nothing reaches S3 hosting, the DB or an LLM."""
    sys.path.insert(0, str(ROOT))
    from hybrid_rag.ingestion.parser.router import _parse_pdf

    with tempfile.TemporaryDirectory() as tmp:
        return asyncio.run(_parse_pdf(path, Path(tmp)))
