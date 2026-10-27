"""Database-schema ingestion — structured knowledge, no LLM extraction.

A schema YAML (the ``db_schema_examples.yaml`` format: ``name``, ``description``,
``schemas.<schema>.tables.<table>`` with columns, primary and foreign keys) is
already structured, so the graph is built deterministically instead of being
extracted: the database, every schema, table and column become entities, and
foreign keys become ``references`` relations carrying their JOIN condition.
Optional ``business_concepts`` (ratios, distinct counts, hidden filters: rules no
single column states) become entities with their synonyms as aliases, linked by
``uses_column`` to every column they read. Their text also lands as chunks (one
overview per database and schema, one card per table and per concept), so the
three retrieval paths of ``mix`` mode all reach it.

Modules, one job each:

- ``spec``: the YAML format as pydantic models; the only place input is validated.
- ``records``: renders a spec into chunks and node/edge records. Pure.
- ``ingest``: ``_DbSchemaMixin.ainsert_db_schema``, which writes those records.
- ``pipeline``: ``ingest_db_schema``, the entry point that leases the workspace.

One database is one document: ``doc_id`` derives from the database name, so a
re-ingest replaces the previous version and ``delete_document`` removes it. The
document ingestion path is not involved at any point.
"""
