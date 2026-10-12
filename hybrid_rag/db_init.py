"""Create the database schema: ``make db-init`` (``python -m hybrid_rag.db_init``).

The runtime runs no DDL, so run this:

* on first setup, against an empty database;
* after changing ``EMBEDDING_MODEL`` or ``EMBEDDING_DIM``: vector tables are named
  after the model (``hybrid_rag_vdb_chunks_<model>_<dim>d``), so a new model needs
  new tables, and its documents must be reingested;
* after a schema change in the code (a new table, column or index).

Idempotent. Needs a database role allowed to ``CREATE SCHEMA`` and
``CREATE EXTENSION``; the application's role does not.
"""

import asyncio
import logging
from hybrid_rag.config import settings
from hybrid_rag.core.base import collection_suffix
from hybrid_rag.core.namespace import NameSpace
from hybrid_rag.storage.neo4j import create_schema
from hybrid_rag.storage.neo4j.connection import close_neo4j_driver
from hybrid_rag.storage.postgres.connection import PostgreSQLDB, _pool_config
from hybrid_rag.storage.postgres.vector import PGVectorStorage, vector_table

logger = logging.getLogger(__name__)

VECTOR_NAMESPACES = (
    NameSpace.VECTOR_STORE_CHUNKS,
    NameSpace.VECTOR_STORE_ENTITIES,
    NameSpace.VECTOR_STORE_RELATIONSHIPS,
)

async def init_schema() -> None:
    """PostgreSQL schema, extension, tables and indexes, then the Neo4j constraints."""
    db = PostgreSQLDB(_pool_config())
    await db.bootstrap()  # before initdb(): the pool needs the vector extension
    await db.initdb()
    try:
        await db.check_tables()
        suffix = collection_suffix(settings.embedding_model, settings.embedding_dim)
        for namespace in VECTOR_NAMESPACES:
            table, base_table = vector_table(namespace, suffix)
            await PGVectorStorage.setup_table(db, table, base_table, settings.embedding_dim)
            logger.info(f"db-init: {table} ready")
    finally:
        await db.pool.close()
    try:
        await create_schema()
    finally:
        await close_neo4j_driver()
    logger.info("db-init: done")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(init_schema())