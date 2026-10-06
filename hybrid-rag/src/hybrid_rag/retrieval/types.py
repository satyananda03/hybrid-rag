"""What retrieval takes in and hands back."""

from dataclasses import dataclass
from typing import Any

from hybrid_rag.config import settings


@dataclass
class QueryParam:
    """Retrieval budget for one query. Defaults come from ``.env``."""

    top_k: int = settings.top_k
    """Entities (from low-level keywords) and relations (from high-level ones) to retrieve."""

    chunk_top_k: int = settings.chunk_top_k
    """Text chunks kept after reranking."""

    max_entity_tokens: int = settings.max_entity_tokens
    """Token budget for the entity block of the context."""

    max_relation_tokens: int = settings.max_relation_tokens
    """Token budget for the relation block of the context."""

    enable_rerank: bool = settings.rerank_by_default
    """Rerank retrieved chunks; a warning is logged if no rerank model is configured."""

    return_parent: bool = False
    """Kembalikan teks parent, bukan child, sebagai isi tiap chunk hasil.

    Pencarian tetap berjalan di atas child (presisi tidak berubah) — yang diganti
    hanya ``content`` setelah rerank, dan child seinduk dilebur jadi satu parent.
    ``chunk_id`` / ``chunk_order_index`` / ``file_path`` tetap milik child, jadi
    bentuk sitasi tidak berubah sama sekali. Chunk dari dokumen yang di-ingest
    dengan ``save_parent=False`` lewat apa adanya.
    """


@dataclass
class QueryResult:
    """The assembled context string and the structured data behind it."""

    context: str
    raw_data: dict[str, Any]
