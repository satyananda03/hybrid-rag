"""Finds the Pasal headings of a regulation in flat text.

Reads text only, never ``text_level`` or anything else a parser decides: "Pasal 5" is in
the text whichever parser produced it. Nothing here requires ``Pasal 1`` either, as an
upload can be an excerpt and a parser can lose a heading.

Tune: ``python3 hybrid_rag/ingestion/chunking/legal_chunker/articles.py file.pdf``
"""

import re
from typing import Iterable, NamedTuple

#: Fewer articles than this is a document that merely mentions some, not a regulation.
MIN_ARTICLES = 3
#: Largest step between two articles. Tolerates lost headings; rejects a stray ``Pasal 70``
#: or an OCR misread, which would otherwise stop the rest from following it.
MAX_NUMBER_JUMP = 5
#: The same number twice this close is a page-bottom catchword, then the real heading
#: (Perpres 39/2019: ``Pasal 33``, page header, ``Pasal 33``, 43 characters apart).
CATCHWORD_WINDOW = 200

#: ``Pasal N`` ending a line. A citation wrapped there goes on in lowercase on the next line
#: (``dalam\nPasal 38\nhuruf b``), but a heading can be followed by a list item
#: (UU 59/2024: ``Pasal 7\n\na. selayang``). A catchword's dots (``Pasal 2...``) do not match.
_HEADING_LINE = re.compile(r"(?m)(?:#{1,6}[ \t]+)?Pasal[ \t]+(\d{1,3})([A-Z]?)[ \t]*$(?!\s*(?![a-z]\.)[a-z])")
#: ``Pasal N`` starting a line, merged with its one-sentence text
#: (BRIN 2/2024: ``Pasal 45 Peraturan Badan ini mulai berlaku``). Lowercase is a citation (``Pasal 4 ayat (1)``).
_HEADING_WITH_TEXT = re.compile(r"(?m)^[ \t]*Pasal[ \t]+(\d{1,3})([A-Z]?)[ \t]+(?=[A-Z(])")
#: Lines that look like a heading, to report the forms the patterns above do not know (``PasaI 5``).
_LOOKS_LIKE_HEADING = re.compile(r"(?m)^[ \t]*Pas[a-zA-Z0-9]{1,3}[ \t]*\S{1,4}\b.{0,34}$")
_CITATION = re.compile(r"\s*Pasal\s+\d+\s+(?:ayat|huruf|angka|dan|atau|sampai)\b")


class Candidate(NamedTuple):
    """A line that may be a ``Pasal`` heading: ``content[start:end]``."""

    start: int
    end: int
    number: int
    suffix: str  # the ``A`` of an inserted ``Pasal 12A``


class Article(NamedTuple):
    """A ``Pasal`` heading: ``content[start:end]``, and the articles its text covers."""

    start: int
    end: int
    title: str  # "Pasal 5", or "Pasal 24 - Pasal 25" when the heading of Pasal 25 was lost


def find_article_candidates(content: str) -> list[Candidate]:
    """Every line of ``content`` that may be a heading, in document order."""
    matches = [match for match in _HEADING_LINE.finditer(content) if _opens_heading(content, match.start())]
    matches += _HEADING_WITH_TEXT.finditer(content)
    return sorted(Candidate(match.start(), match.end(), int(match[1]), match[2]) for match in matches)


def select_articles(candidates: Iterable[Candidate]) -> list[Article]:
    """The candidates that form the regulation's ascending article numbers, titled."""
    return _titled(_ascending_run(candidates))


def find_near_misses(content: str, end: int, candidates: Iterable[Candidate]) -> list[str]:
    """Lines before ``end`` that look like a heading but no pattern accepted. Only for tuning."""
    known = {candidate.start for candidate in candidates}
    return [
        match[0].strip()
        for match in _LOOKS_LIKE_HEADING.finditer(content, 0, end)
        if match.start() not in known and not _CITATION.match(match[0])
    ]


def _opens_heading(content: str, start: int) -> bool:
    """Alone on its line, or after a title it was merged with (``KETENTUAN PERALIHAN Pasal 28``).

    A citation at the end of a line follows a lowercase word (``dimaksud dalam Pasal 13``).
    """
    before = content[content.rfind("\n", 0, start) + 1 : start]
    if not before.strip():
        return True
    return before[-1].isspace() and before.split()[-1][0].isupper()


def _ascending_run(candidates: Iterable[Candidate]) -> list[Candidate]:
    """The run may start at any number, but a candidate only anchors it once the next one follows.

    So one stray line up front cannot become the start; after that, what does not follow is skipped.

    ponytail: a jump over ``MAX_NUMBER_JUMP`` ends the run, and the articles after it fall
    under the last one. Restart the run there if that shows up in real documents.
    """
    run: list[Candidate] = []
    for candidate in candidates:
        if run and _is_catchword(run[-1], candidate):
            run[-1] = candidate
        elif run and _follows(run[-1], candidate):
            run.append(candidate)
        elif len(run) < 2:
            run = [candidate]
    return run


def _is_catchword(previous: Candidate, candidate: Candidate) -> bool:
    return previous[2:] == candidate[2:] and candidate.start - previous.start <= CATCHWORD_WINDOW


def _follows(previous: Candidate, candidate: Candidate) -> bool:
    return previous[2:] < candidate[2:] and candidate.number - previous.number <= MAX_NUMBER_JUMP


def _titled(run: list[Candidate]) -> list[Article]:
    """A gap in the numbers means a lost heading, whose text sits under the article before it.

    The title says so instead of guessing where that article starts, and names each end as
    ``Pasal N`` because BM25 and embedding search match the text, so ``Pasal 25`` must be in it.
    """
    articles = []
    for current, following in zip(run, [*run[1:], None]):
        title = f"Pasal {current.number}{current.suffix}"
        if following and following.number - current.number > 1:
            title += f" - Pasal {following.number - 1}"
        articles.append(Article(current.start, current.end, title))
    return articles


def _print_report(content: str) -> None:
    from body_end import detect_body_end

    candidates = find_article_candidates(content)
    if not candidates:
        print("Tidak ada kandidat Pasal.")
        return
    end = detect_body_end(content, candidates[0].end)
    inside = [candidate for candidate in candidates if candidate.start < end]
    articles = select_articles(inside)
    accepted = {article.start for article in articles}
    print(f"Pasal diterima : {len(articles)} ({'lolos' if len(articles) >= MIN_ARTICLES else 'bukan peraturan'})")
    print(f"Urutan         : {articles[0].title} .. {articles[-1].title}" if articles else "Urutan         : -")
    print(f"Nomor bolong   : {[article.title for article in articles if ' - ' in article.title] or '-'}")
    print("Ditolak (catchword atau di luar urutan):")
    for candidate in inside:
        if candidate.start not in accepted:
            print(f"  {content[candidate.start:candidate.start + 60]!r}")
    print("Nyaris cocok   :")
    for line in find_near_misses(content, end, inside):
        print(f"  {line!r}")


if __name__ == "__main__":
    from debug_input import load_text, pdf_argument

    _print_report(load_text(pdf_argument()))
