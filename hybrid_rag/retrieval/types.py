"""What retrieval takes in and hands back."""

from dataclasses import dataclass
from typing import Any, Literal

from hybrid_rag.config import settings

CitationStyle = Literal["location", "document"]


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

    Untuk pemanggil yang butuh bukti utuh (satu pasal penuh), bukan sekadar kalimat
    yang paling mirip. Pencarian tetap berjalan di atas child (presisi tidak
    berubah) — yang diganti hanya ``content`` setelah rerank, child seinduk dilebur
    jadi satu parent, dan hasilnya dipotong ke ``PARENT_TOP_K``. ``chunk_id`` /
    ``file_path`` tetap milik child; lokasi sitasi mengikuti teks parent yang
    ditampilkan. Chunk dari dokumen yang di-ingest dengan ``save_parent=False``
    lewat apa adanya.
    """

    include_citations: bool = True
    """Sertakan sitasi: ``reference_id`` di tiap chunk, Reference Document List di
    context (satu entri per file + lokasi), dan ``data.references``. False untuk
    pemanggil yang hanya butuh teks bukti; tidak ada lookup lokasi sama sekali."""

    citation_style: CitationStyle = "location"
    """How sources read. ``location``: the file and where in it (``a.pdf, hal. 3``). ``document``:
    the citation the chunker gave the chunk (``Pasal 5, PERATURAN PRESIDEN REPUBLIK INDONESIA NOMOR 39 TAHUN 2023``), else the
    file name; never a page."""

    keywords: tuple[list[str], list[str]] | None = None
    """``(high_level, low_level)`` already extracted for this query (what ``extract_keywords_only``
    returns). Set: extraction is skipped; the empty-keyword fallbacks and stages 1-4 run as usual.
    None (default): keywords are extracted from the query. Set by the cross-workspace search so every
    workspace is searched with the same keywords from one extraction."""

    return_chunk_only: bool = False
    """Context hanya berisi ``<document_chunks>`` (+ references bila ada), tanpa blok
    entitas/relasi. Murni format output: retrieval (KG + vektor) tetap sama, dan
    ``raw_data`` tetap memuat entitas/relasi."""


@dataclass
class QueryResult:
    """The assembled context string and the structured data behind it."""

    context: str
    raw_data: dict[str, Any]
