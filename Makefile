.PHONY: db-init test test-integration

db-init:
	uv run python -m hybrid_rag.db_init

# Unit tests: no network, database or model.
test:
	uv run --extra ingestion pytest

# Tests that read the live PostgreSQL from .env (read-only). Run `make db-init` first.
test-integration:
	uv run --extra ingestion pytest -m integration
