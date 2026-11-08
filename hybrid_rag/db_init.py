"""Create the database schema: ``make db-init`` (``python -m hybrid_rag.db_init``).

The runtime runs no DDL, so run this:

* on first setup, against an empty database;
* after changing ``EMBEDDING_MODEL`` or ``EMBEDDING_DIM``: the Neo4j vector property and
  indexes are named after the model (``embedding_<model>_<dim>d``), so a new model needs
  new indexes, and its documents must be reingested;
* after a schema change in the code (a new table, column or index).

Idempotent. Needs a PostgreSQL role allowed to ``CREATE SCHEMA`` (no extension is
needed) and Neo4j >= 2026.09 (vector and full-text ``SEARCH``).
"""

import asyncio
import logging
from hybrid_rag.storage.neo4j import create_schema
from hybrid_rag.storage.neo4j.connection import close_neo4j_driver
from hybrid_rag.storage.postgres.connection import PostgreSQLDB, _pool_config

logger = logging.getLogger(__name__)

async def init_schema() -> None:
    """PostgreSQL schema, tables and indexes, then the Neo4j constraints and search indexes."""
    db = PostgreSQLDB(_pool_config())
    await db.bootstrap()  # before initdb(): the pool's search_path names the schema
    await db.initdb()
    try:
        await db.check_tables()
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