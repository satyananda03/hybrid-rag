"""Invariants kode untuk hybrid_rag — bukan konfigurasi.

Semua nilai yang bisa diatur lewat ``.env`` sudah pindah ke
``app/core/config.py`` sebagai default field ``Settings``. Yang tersisa di sini
sengaja TIDAK dijadikan setting: pemisah/format data yang mengubah arti baris
yang sudah tersimpan bila diganti (``GRAPH_FIELD_SEP``), batas panjang nama
entitas, dan sentinel status. Menjadikannya knob env hanya mengundang perubahan
yang merusak data lama.
"""

DEFAULT_ENTITY_NAME_MAX_LENGTH = 256
# Max UTF-8 byte length for entity identifiers: a name within the character
# limit can still exceed it with multi-byte characters. MUST stay <= the
# VARCHAR(512) of entity_name / source_id / target_id in storage/postgres/sql.py.
DEFAULT_ENTITY_NAME_MAX_BYTES = 512

DEFAULT_MAX_SECTION_CONTEXT_TOKENS = 256
# Per-level character cap for each heading in that breadcrumb. Must stay below
# 1/3 of DEFAULT_MAX_SECTION_CONTEXT_TOKENS so the collapsed two-level form
# (first + leaf, plus separator/ellipsis) always fits within the token budget.
DEFAULT_HEADING_LEVEL_MAX_CHARS = 80
# Separator for: description, source_id and relation-key fields(Can not be changed after data inserted)
GRAPH_FIELD_SEP = "<SEP>"

SOURCE_IDS_LIMIT_METHOD_KEEP = "KEEP"
SOURCE_IDS_LIMIT_METHOD_FIFO = "FIFO"
VALID_SOURCE_IDS_LIMIT_METHODS = {
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
}
DEFAULT_FILE_PATH_MORE_PLACEHOLDER = "truncated"

FULL_DOCS_FORMAT_RAW = "raw"
# Prefix marking a doc_status content_summary as GENERATED from a file
# extraction error (enqueue-time error documents and parse-stage FAILED
# upserts). Doubles as the match sentinel that lets a later failure replace
# a stale generated summary while real raw-document summaries are preserved —
# keep every producer on this constant so the match never drifts.
FILE_EXTRACTION_SUMMARY_PREFIX = "[File Extraction]"

