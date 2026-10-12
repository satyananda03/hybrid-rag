"""Fixtures shared by the test modules.

Tests are unit tests by default: no network, no database, no model. Those that need
the live PostgreSQL from ``.env`` are marked ``integration`` and skipped unless
asked for (``make test-integration``).
"""

import asyncio

import pytest
from langchain_core.runnables import RunnableLambda
from PIL import Image

from hybrid_rag import providers
from hybrid_rag.config import settings


class FakeLLM:
    """A chat model whose structured output is always empty."""

    def with_structured_output(self, model):
        return RunnableLambda(lambda _: model(entities=[], relationships=[]))


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a fake ``get_llm`` hook and drop the chains cached on the real one."""
    from hybrid_rag.ingestion.knowledge_graph import chain

    monkeypatch.setitem(providers._hooks, "get_llm", lambda **_: FakeLLM())
    chain._build_chains.cache_clear()
    chain._extract_llm.cache_clear()
    yield
    chain._build_chains.cache_clear()
    chain._extract_llm.cache_clear()


@pytest.fixture
def make_png(tmp_path):
    """``make_png("a.png", (900, 900))`` writes a blank PNG and returns its path."""

    def make(name, size):
        path = tmp_path / name
        Image.new("RGB", size, "white").save(path, "PNG")
        return path

    return make


@pytest.fixture(scope="session")
def pg():
    """``pg(sql, *args)`` runs one read-only query on the PostgreSQL from ``.env``.

    Skips the test when the database cannot be reached.
    """
    import asyncpg
    from pgvector.asyncpg import register_vector

    async def fetch(sql, *args):
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_database,
            timeout=5,
        )
        try:
            await conn.execute("SET default_transaction_read_only = on")
            await register_vector(conn)
            return await conn.fetch(sql, *args)
        finally:
            await conn.close()

    try:
        asyncio.run(fetch("SELECT 1"))
    except Exception as e:
        pytest.skip(f"PostgreSQL not reachable: {type(e).__name__}: {e}")
    return lambda sql, *args: asyncio.run(fetch(sql, *args))
