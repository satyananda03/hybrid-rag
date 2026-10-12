"""PostgreSQL backends — KV, vector (+BM25), and doc-status.

Split by role. The three storage classes are re-exported here.

    connection.py  the asyncpg pool, schema bootstrap, batching helpers
    kv.py          PGKVStorage
    vector.py      PGVectorStorage — embeddings and the BM25 column
    doc_status.py  PGDocStatusStorage
    sql.py         TABLES + SQL_TEMPLATES
    bm25.py        Indonesian stopword stripping for the BM25 column
"""

from hybrid_rag.storage.postgres.connection import ClientManager, PostgreSQLDB
from hybrid_rag.storage.postgres.doc_status import PGDocStatusStorage
from hybrid_rag.storage.postgres.kv import PGKVStorage
from hybrid_rag.storage.postgres.vector import PGVectorStorage

__all__ = [
    "ClientManager",
    "PGDocStatusStorage",
    "PGKVStorage",
    "PGVectorStorage",
    "PostgreSQLDB",
]
