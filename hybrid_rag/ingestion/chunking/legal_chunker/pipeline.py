"""Connects the regulation detectors. The only module that knows how they fit together."""
from typing import NamedTuple
from hybrid_rag.ingestion.chunking.legal_chunker.articles import (
    MIN_ARTICLES,
    Article,
    find_article_candidates,
    select_articles,
)
from hybrid_rag.ingestion.chunking.legal_chunker.body_end import detect_body_end
from hybrid_rag.ingestion.chunking.legal_chunker.headers import detect_repeated_headers
from hybrid_rag.ingestion.chunking.legal_chunker.regulation_name import detect_regulation_name

class Regulation(NamedTuple):
    """What the detectors found in a regulation's text."""
    articles: list[Article]
    end: int
    name: str | None
    def citation(self, article: Article) -> str:
        """``Pasal 5, PERATURAN PRESIDEN REPUBLIK INDONESIA NOMOR 37 TAHUN 2026``; the article alone when the name is unknown."""
        return f"{article.title}, {self.name}" if self.name else article.title

def analyze_regulation(content: str, document_name: str | None = None) -> Regulation | None:
    """The structure of ``content``, or ``None`` when it does not follow the regulation pattern."""
    candidates = find_article_candidates(content)
    if not candidates:
        return None
    end = detect_body_end(content, candidates[0].end)
    articles = select_articles(candidate for candidate in candidates if candidate.start < end)
    if len(articles) < MIN_ARTICLES:
        return None
    return Regulation(articles, end, detect_regulation_name(content, document_name))

def remove_regulation_headers(content_list: list[dict]) -> list[dict]:
    """``content_list`` without its running headers, when it is a regulation; other documents pass unchanged."""
    texts = [item for item in content_list if item.get("type", "text") == "text"]
    if any(item.get("layout", "document") != "document" for item in texts):
        return content_list
    if analyze_regulation("\n\n".join(item.get("text", "") for item in texts)) is None:
        return content_list
    headers = detect_repeated_headers(content_list)
    return [item for index, item in enumerate(content_list) if index not in headers]