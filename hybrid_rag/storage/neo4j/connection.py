"""The process-wide Neo4j driver every Neo4j storage shares."""

import logging
from functools import lru_cache
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from hybrid_rag.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_neo4j_driver() -> AsyncDriver:
    """Return the process-wide Neo4j driver, building it on first use."""
    if not settings.neo4j_uri:
        raise ValueError("NEO4J_URI belum diset — Neo4j graph storage tidak bisa jalan.")
    logger.info("Neo4j: membuka driver ke %s", settings.neo4j_uri)
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
        max_connection_pool_size=settings.neo4j_max_connection_pool_size,
        connection_timeout=settings.neo4j_connection_timeout,
        connection_acquisition_timeout=settings.neo4j_connection_acquisition_timeout,
        max_transaction_retry_time=settings.neo4j_max_transaction_retry_time,
        max_connection_lifetime=settings.neo4j_max_connection_lifetime,
        liveness_check_timeout=settings.neo4j_liveness_check_timeout,
        keep_alive=settings.neo4j_keep_alive,
    )


async def read(query: str, **params: Any) -> list[dict[str, Any]]:
    """Run a read-only query on the shared driver; the rows as dicts."""
    async with get_neo4j_driver().session(
        database=settings.neo4j_database or None, default_access_mode="READ"
    ) as session:
        result = await session.run(query, params)
        return [record.data() async for record in result]


async def close_neo4j_driver() -> None:
    """Close the driver. Call once when the application stops."""
    if get_neo4j_driver.cache_info().currsize == 0:
        return
    driver = get_neo4j_driver()
    get_neo4j_driver.cache_clear()
    await driver.close()
    logger.info("Neo4j: driver ditutup")
