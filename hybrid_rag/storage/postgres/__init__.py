"""PostgreSQL backends — KV and doc-status (vector + BM25 search live in Neo4j).

Split by role. The storage classes are re-exported here.

    connection.py  the asyncpg pool, schema bootstrap, batching helpers
    kv.py          PGKVStorage
    doc_status.py  PGDocStatusStorage
    sql.py         TABLES + SQL_TEMPLATES
"""

from hybrid_rag.storage.postgres.connection import ClientManager, PostgreSQLDB
from hybrid_rag.storage.postgres.doc_status import PGDocStatusStorage
from hybrid_rag.storage.postgres.kv import PGKVStorage

__all__ = [
    "ClientManager",
    "PGDocStatusStorage",
    "PGKVStorage",
    "PostgreSQLDB",
]
