import re
import logging

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
            logger.info("PostgreSQL, Sastrawi stopwords loaded for BM25 preprocessing")
        except ImportError:
            stopwords = frozenset()
            logger.warning(
                "PostgreSQL, Sastrawi not found. Stopword removal skipped, but basic cleaning (lowercase/punctuation) will still apply."
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