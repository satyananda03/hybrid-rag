"""Navigasi graf Block Textract + geometri bounding box (koordinat ternormalisasi 0-1)."""

ROTATED_ABOVE_DEGREES = 45.0
#: Dua LINE sebaris bila tumpang tindih vertikalnya melebihi porsi ini dari tinggi yang lebih kecil.
SAME_ROW_OVERLAP = 0.5


def build_block_map(blocks: list) -> dict:
    return {b["Id"]: b for b in blocks}


def get_children(block: dict, block_map: dict, rel_type: str = "CHILD") -> list:
    ids = []
    for rel in block.get("Relationships", []):
        if rel["Type"] == rel_type:
            ids.extend(rel["Ids"])
    return [block_map[i] for i in ids if i in block_map]


def get_plain_text(block: dict, block_map: dict) -> str:
    """Rekursive Loop sampai WORD/LINE, untuk merangkai teks.

    ``LAYOUT_TABLE`` (fallback tanpa TABLE) tetap urutan Textract: urutan baca tabel
    butuh uji tersendiri.
    """
    if block["BlockType"] == "WORD":
        return block.get("Text", "")
    if block["BlockType"] == "LINE":
        return _line_text(block, block_map)
    children = get_children(block, block_map)
    if block["BlockType"] != "LAYOUT_TABLE":
        children = lines_in_reading_order(children)
    parts = []
    for child in children:
        txt = get_plain_text(child, block_map)
        if txt:
            parts.append(txt)
    return " ".join(parts).strip()


def lines_in_reading_order(blocks: list) -> list:
    """LINE per barisan dari atas, tiap barisan dari kiri; selain LINE dikembalikan apa adanya.

    AWS tidak menjanjikan urutan ``Ids``. Anak blok LAYOUT nyatanya urut ``Top``, sehingga
    label yang sedikit lebih rendah jatuh setelah teksnya (``bertugas: KEDUA``).
    """
    if any(block["BlockType"] != "LINE" for block in blocks):
        return blocks
    rows: list[list[dict]] = []
    for line in sorted(blocks, key=block_top):
        if rows and _same_row(rows[-1][-1], line):
            rows[-1].append(line)
        else:
            rows.append([line])
    return [line for row in rows for line in sorted(row, key=_left)]


def _line_text(line: dict, block_map: dict) -> str:
    """``Text`` Textract, kecuali baris mendatar yang katanya diberi rotasi.

    Textract mengurutkan kata baris seperti itu menurut ``Top``, sehingga teksnya teracak
    (``Surabaya di Ditetapkan``), jadi disusun ulang dari kiri.
    """
    words = get_word_blocks(line, block_map)
    if not any(_is_rotated(word) for word in words) or _is_taller_than_wide(line):
        return line.get("Text", "")
    return " ".join(word.get("Text", "") for word in sorted(words, key=_left))


def block_top(block: dict) -> float:
    return block["Geometry"]["BoundingBox"]["Top"]


def _left(block: dict) -> float:
    return block["Geometry"]["BoundingBox"]["Left"]


def _same_row(first: dict, second: dict) -> bool:
    box_a, box_b = first["Geometry"]["BoundingBox"], second["Geometry"]["BoundingBox"]
    overlap = min(box_a["Top"] + box_a["Height"], box_b["Top"] + box_b["Height"]) - max(box_a["Top"], box_b["Top"])
    return overlap > SAME_ROW_OVERLAP * min(box_a["Height"], box_b["Height"])


def get_word_blocks(block: dict, block_map: dict) -> list:
    """Rekursive Loop mengumpulkan semua block WORD dari suatu parent block."""
    if block["BlockType"] == "WORD":
        return [block]
    words = []
    for child in get_children(block, block_map):
        words.extend(get_word_blocks(child, block_map))
    return words


def _is_rotated(word: dict) -> bool:
    return abs(word.get("Geometry", {}).get("RotationAngle", 0.0)) > ROTATED_ABOVE_DEGREES


def _is_taller_than_wide(block: dict) -> bool:
    bbox = block["Geometry"]["BoundingBox"]
    return bbox["Height"] > bbox["Width"]


def is_vertical_text(block: dict, block_map: dict, ratio_threshold: float = 0.5) -> bool:
    """Sebagian besar katanya berdiri: berotasi dan lebih tinggi daripada lebar.

    Textract juga memberi ``RotationAngle`` 90/270 pada teks mendatar, jadi rotasi saja
    membuang isi dokumen. Bentuk saja juga tidak cukup: ``:`` atau ``1.`` mendatar pun
    lebih tinggi daripada lebar.

    ponytail: bbox ternormalisasi ke ukuran halaman, jadi kata vertikal 1-2 huruf bisa
    terbaca lebar. Hitung per piksel dari ukuran halaman bila itu muncul di dokumen nyata.
    """
    words = get_word_blocks(block, block_map)
    if not words:
        return False
    standing = sum(1 for word in words if _is_rotated(word) and _is_taller_than_wide(word))
    return standing / len(words) > ratio_threshold


def bbox_to_pixels(bbox: dict, img_w: int, img_h: int):
    left = bbox["Left"] * img_w
    top = bbox["Top"] * img_h
    width = bbox["Width"] * img_w
    height = bbox["Height"] * img_h
    return left, top, left + width, top + height


def bbox_iou(b1: dict, b2: dict) -> float:
    ax1, ay1 = b1["Left"], b1["Top"]
    ax2, ay2 = ax1 + b1["Width"], ay1 + b1["Height"]
    bx1, by1 = b2["Left"], b2["Top"]
    bx2, by2 = bx1 + b2["Width"], by1 + b2["Height"]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = b1["Width"] * b1["Height"]
    area_b = b2["Width"] * b2["Height"]
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def bbox_overlap_ratio(bbox1: dict, bbox2: dict) -> float:
    """Menghitung seberapa persen sebuah bbox1 tumpang tindih dengan bbox2."""
    ax1, ay1 = bbox1["Left"], bbox1["Top"]
    ax2, ay2 = ax1 + bbox1["Width"], ay1 + bbox1["Height"]
    bx1, by1 = bbox2["Left"], bbox2["Top"]
    bx2, by2 = bx1 + bbox2["Width"], by1 + bbox2["Height"]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = bbox1["Width"] * bbox1["Height"]
    return inter_area / area1 if area1 > 0 else 0.0
