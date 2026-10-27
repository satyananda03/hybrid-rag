"""Invariants kode untuk hybrid_rag — bukan konfigurasi.

Semua nilai yang bisa diatur lewat ``.env`` sudah pindah ke
``app/core/config.py`` sebagai default field ``Settings``. Yang tersisa di sini
sengaja TIDAK dijadikan setting: pemisah/format data yang mengubah arti baris
yang sudah tersimpan bila diganti (``GRAPH_FIELD_SEP``), batas panjang nama
entitas, dan sentinel status. Menjadikannya knob env hanya mengundang perubahan
yang merusak data lama.
"""

from typing import Iterable


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

# Entity types written by the database-schema pipeline (ingestion/db_schema).
# Stored on nodes, so renaming one orphans the rows already written. Never plain
# "table": that type belongs to the multimodal pipeline (entity_resolution).
DB_ENTITY_DATABASE = "db_database"
DB_ENTITY_SCHEMA = "db_schema"
DB_ENTITY_TABLE = "db_table"
DB_ENTITY_COLUMN = "db_column"
DB_ENTITY_CONCEPT = "db_business_concept"
DB_ENTITY_DOMAIN = "db_domain"
DB_ENTITY_VALUE = "db_value"
DB_SCHEMA_ENTITY_TYPES = frozenset({
    DB_ENTITY_DATABASE, DB_ENTITY_SCHEMA, DB_ENTITY_TABLE, DB_ENTITY_COLUMN,
    DB_ENTITY_CONCEPT, DB_ENTITY_DOMAIN, DB_ENTITY_VALUE,
})

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


# All namespace should not be changed
class NameSpace:
    KV_STORE_FULL_DOCS = "full_docs"
    KV_STORE_TEXT_CHUNKS = "text_chunks"
    KV_STORE_LLM_RESPONSE_CACHE = "llm_response_cache"
    KV_STORE_FULL_ENTITIES = "full_entities"
    KV_STORE_FULL_RELATIONS = "full_relations"
    KV_STORE_ENTITY_CHUNKS = "entity_chunks"
    KV_STORE_RELATION_CHUNKS = "relation_chunks"

    VECTOR_STORE_ENTITIES = "entities"
    VECTOR_STORE_RELATIONSHIPS = "relationships"
    VECTOR_STORE_CHUNKS = "chunks"

    GRAPH_STORE_CHUNK_ENTITY_RELATION = "chunk_entity_relation"

    DOC_STATUS = "doc_status"


def is_namespace(namespace: str, base_namespace: str | Iterable[str]):
    if isinstance(base_namespace, str):
        return namespace.endswith(base_namespace)
    return any(is_namespace(namespace, ns) for ns in base_namespace)
