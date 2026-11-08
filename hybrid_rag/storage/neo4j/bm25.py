"""BM25 (sparse) chunk search: Indonesian text normalisation + Neo4j's Lucene index.

The same :func:`preprocess_indonesian_fast` runs on the stored chunk text (written by
``Neo4jVectorStorage`` as ``content_cleaned``) and on the query, so both sides share one
vocabulary; the index itself only splits on whitespace (``cypher.SCHEMA``).
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from hybrid_rag.storage.neo4j import cypher
from hybrid_rag.storage.neo4j.connection import read

logger = logging.getLogger(__name__)

KEEP = frozenset({"tidak", "belum", "ya", "sudah", "ada", "tanpa", "dua", "kecuali", "nggak", "ok"})

stopwords = None

def get_stopwords() -> frozenset[str]:
    """Lazy singleton: stopword Sastrawi dikurangi ``KEEP``; kosong kalau Sastrawi tidak terpasang."""
    global stopwords
    if stopwords is None:
        try:
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
            stopwords = frozenset(StopWordRemoverFactory().get_stop_words()) - KEEP
            logger.info("Sastrawi stopwords loaded for BM25 preprocessing")
        except ImportError:
            stopwords = frozenset()
            logger.warning(
                "Sastrawi not found. Stopword removal skipped, but basic cleaning (lowercase/punctuation) will still apply."
            )
    return stopwords

def preprocess_indonesian_fast(text: str) -> str:
    # Lowercasing
    cleaned_text = text.lower()
    # Hapus semua tanda baca (hanya menyisakan huruf, angka, dan spasi)
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
    # Hapus stopword (split() juga membuang spasi ganda sisa penghapusan tanda baca)
    sw = get_stopwords()
    return " ".join(w for w in cleaned_text.split() if w not in sw)


@dataclass(frozen=True)
class Neo4jBM25Search:
    """BM25 over one workspace's chunk nodes.

    Read-only: the chunk node, ``content_cleaned`` included, is written by
    ``Neo4jVectorStorage``. Hits have the shape of the vector search's
    (``id, content, file_path, created_at, score``), best first.
    """

    workspace: str

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        cleaned = preprocess_indonesian_fast(query)
        if not cleaned:  # only stopwords: nothing to match, and "()" would not parse
            return []
        return await read(
            cypher.CHUNK_BM25_SEARCH,
            bm25_query=cypher.bm25_query(self.workspace, cleaned),
            top_k=top_k,
        )
