"""Finds running headers in a content list by how they repeat, not by what a parser labels them."""
import re
from collections import Counter
#: A header was the first block of its page in all 10 documents measured; one more tolerates a stray block.
TOP_BLOCKS = 2
#: A header is a short block; it can span lines (``PRESIDEN`` / ``REPUBLIK INDONESIA``), so this counts words.
MAX_WORDS = 8
#: Headers were on 86-100% of the pages measured, any other top block on at most 14%.
MIN_PAGE_SHARE = 0.6
#: Fewer repeats than this is not a pattern, however short the document.
MIN_PAGES = 3

def detect_repeated_headers(content_list: list[dict]) -> set[int]:
    """Indexes of the items in ``content_list`` that are running headers."""
    tops = page_tops(content_list)
    threshold = max(MIN_PAGES, MIN_PAGE_SHARE * len(tops))
    counts = count_pages(tops)
    return {index for blocks in tops.values() for index, key in blocks if counts[key] >= threshold}

def page_tops(content_list: list[dict]) -> dict[object, list[tuple[int, str]]]:
    """Per page, its first ``TOP_BLOCKS`` text blocks that are short enough: ``(index, key)``."""
    tops: dict[object, list[tuple[int, str]]] = {}
    seen: Counter = Counter()
    for index, item in enumerate(content_list):
        text = item.get("text", "") if item.get("type", "text") == "text" else ""
        if not text.strip():
            continue
        page = item.get("page_idx")
        blocks = tops.setdefault(page, [])
        seen[page] += 1
        if seen[page] <= TOP_BLOCKS and len(text.split()) <= MAX_WORDS and (key := header_key(text)):
            blocks.append((index, key))
    return tops

def header_key(text: str) -> str:
    """Ignores case and punctuation, not digits: without them ``Pasal 1`` and ``Pasal 2`` would look repeated."""
    return re.sub(r"[^\w]+", " ", text).strip().casefold()

def count_pages(tops: dict[object, list[tuple[int, str]]]) -> Counter:
    """On how many pages each key is a top block."""
    return Counter(key for blocks in tops.values() for key in {key for _, key in blocks})

def print_report(content_list: list[dict]) -> None:
    tops = page_tops(content_list)
    headers = detect_repeated_headers(content_list)
    print(f"Halaman: {len(tops)}, ambang: {max(MIN_PAGES, MIN_PAGE_SHARE * len(tops)):.1f} halaman")
    print(f"Blok dibuang: {len(headers)}")
    for text, count in Counter(content_list[index]["text"].strip() for index in headers).most_common():
        print(f"  {count}x {text!r}")
    print("Kandidat teratas:")
    for key, count in count_pages(tops).most_common(8):
        print(f"  {count}/{len(tops)} {key!r}")

if __name__ == "__main__":
    from debug_input import load_content_list, pdf_argument
    print_report(load_content_list(pdf_argument()))