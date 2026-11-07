"""Finds the name a regulation is cited by: from its cover, else from its file name."""
import re
from pathlib import Path

COVER_CHARS = 3000
EARLIEST_YEAR, LATEST_YEAR = 1945, 2100
COVER = re.compile(
    r"((?:UNDANG-UNDANG|(?:PERATURAN|KETETAPAN|KEPUTUSAN) [A-Z ]+?)(?: REPUBLIK INDONESIA)?) "
    r"NOMOR ([0-9OolIL]{1,4}) TAHUN ([0-9OolIL]{4})\b"
)
OCR_DIGITS = str.maketrans("OolIL", "00111")

def detect_regulation_name(content: str, document_name: str | None = None) -> str | None:
    """``PERATURAN PRESIDEN REPUBLIK INDONESIA NOMOR 37 TAHUN 2026`` from the cover, else the file name without its extension."""
    return name_from_cover(content) or name_from_file(document_name)

def name_from_cover(content: str) -> str | None:
    match = COVER.search(" ".join(content[:COVER_CHARS].split()))
    if match is None:
        return None
    kind, number, year = match[1], match[2].translate(OCR_DIGITS), match[3].translate(OCR_DIGITS)
    if not number.isdigit() or not EARLIEST_YEAR <= int(year) <= LATEST_YEAR:
        return None
    return f"{kind} NOMOR {number} TAHUN {year}"

def name_from_file(document_name: str | None) -> str | None:
    return Path(document_name).stem if document_name else None

if __name__ == "__main__":
    from debug_input import load_text, pdf_argument
    pdf = pdf_argument()
    content = load_text(pdf)
    print(f"Dari sampul   : {name_from_cover(content)}")
    print(f"Dari nama file: {name_from_file(pdf.name)}")
    print(f"Dipakai       : {detect_regulation_name(content, pdf.name)}")
    print(f"Awal teks     : {content[:300]!r}")