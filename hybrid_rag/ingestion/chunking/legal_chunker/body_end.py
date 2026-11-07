"""Finds where the articles of a regulation end.

Tune: ``python3 hybrid_rag/ingestion/chunking/legal_chunker/body_end.py file.pdf``
"""

import re

#: What follows the articles: signature block, Penjelasan (repeats their numbers), Lampiran
#: (quotes them). UU use ``Disahkan di`` where other regulations use ``Ditetapkan di``.
_BODY_END = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]+)?(?:Ditetapkan di|Disahkan di|Diundangkan di|PENJELASAN|LAMPIRAN)\b"
)


def detect_body_end(content: str, after: int) -> int:
    """Where the text after the articles begins, searching from ``after``; the end of ``content`` if nothing marks it."""
    match = _BODY_END.search(content, after)
    return match.start() if match else len(content)


def _print_report(content: str) -> None:
    from articles import find_article_candidates

    candidates = find_article_candidates(content)
    if not candidates:
        print("Tidak ada kandidat Pasal.")
        return
    end = detect_body_end(content, candidates[0].end)
    print(f"Akhir batang tubuh: karakter {end} dari {len(content)}")
    print("Tidak ada penanda: seluruh sisa teks ikut Pasal terakhir." if end == len(content) else f"Penanda: {content[end:end + 70]!r}")
    print(f"Teks sebelumnya: {content[max(0, end - 200):end]!r}")


if __name__ == "__main__":
    from debug_input import load_text, pdf_argument

    _print_report(load_text(pdf_argument()))
