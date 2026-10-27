"""Database-schema ingestion: spec validation, record rendering, and the write flow.

Offline: storages are in-memory fakes and the KG merge is stubbed, so this checks
what ``ainsert_db_schema`` decides and writes, not Postgres/Neo4j themselves.
"""

import asyncio
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_rag.core.base import DocStatus
from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.core.tokenizer import get_tokenizer
from hybrid_rag.ingestion.db_schema import ingest
from hybrid_rag.ingestion.db_schema.ingest import _DbSchemaMixin, schema_document
from hybrid_rag.ingestion.db_schema.records import _MAX_VALUE_NODES, build_records
from hybrid_rag.ingestion.db_schema.spec import load_spec

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "db_schema_examples.yaml"
TOKENIZER = get_tokenizer()


def _column(name, **extra):
    return {"name": name, "type": "INTEGER", "description": f"kolom {name}", **extra}


def _two_schemas() -> dict:
    """Same table name in two schemas, plus a foreign key across them."""
    return {
        "name": "dwh",
        "description": "Gudang data.",
        "schemas": {
            "sales": {"name": "sales", "tables": {"orders": {
                "name": "orders", "description": "Pesanan.", "primary_keys": ["id"],
                "columns": [_column("id", is_primary_key=True), _column("customer_id")],
                "foreign_keys": [{
                    "name": "orders_customer_fkey", "constrained_columns": ["customer_id"],
                    "referred_schema": "crm", "referred_table": "customers", "referred_columns": ["id"],
                }],
            }}},
            "crm": {"name": "crm", "tables": {
                "customers": {"name": "customers", "description": "Pelanggan.",
                              "primary_keys": ["id"], "columns": [_column("id")], "foreign_keys": []},
                "orders": {"name": "orders", "description": "Pesanan CRM.",
                           "primary_keys": ["id"], "columns": [_column("id")], "foreign_keys": []},
            }},
        },
        "business_concepts": [{
            "name": "Pelanggan Aktif", "synonyms": ["Active Customers", "Customer Aktif"],
            "description": "Pelanggan yang pernah memesan.",
            "sql": "COUNT(DISTINCT c.id)", "filters": ["o.id IS NOT NULL"],
            "from": "crm.customers c JOIN sales.orders o ON o.customer_id = c.id",
            "columns": ["crm.customers.id", "sales.orders.customer_id"],
        }],
    }


def _records(source, max_tokens=8192):
    spec = load_spec(source)
    doc_id, file_path = schema_document(spec.name)
    return build_records(spec, doc_id=doc_id, file_path=file_path, tokenizer=TOKENIZER, max_tokens=max_tokens)


def test_example_schema_renders_every_table_column_and_foreign_key():
    records = _records(EXAMPLE)
    values = sum(
        len(c.enum_values) for _, t in load_spec(EXAMPLE).iter_tables() for c in t.columns
        if len(c.enum_values) <= _MAX_VALUE_NODES
    )

    types = Counter(r[0]["entity_type"] for r in records.nodes.values())
    assert types == {
        "db_database": 1, "db_schema": 1, "db_domain": 8, "db_table": 22, "db_column": 299,
        "db_value": values, "db_business_concept": 9,
    }
    relations = Counter(r[0]["relation_type"] for r in records.edges.values())
    assert relations == {
        "contains_schema": 1, "contains_domain": 8, "covers_table": 22, "contains_table": 22,
        "has_column": 299, "has_value": values, "references": 12, "uses_column": 19,
    }
    assert {"database:iccn_data_lake", "schema:iccn_dwh"} <= records.nodes.keys()

    for rec in [r[0] for r in records.nodes.values()] + [r[0] for r in records.edges.values()]:
        assert rec["description"].strip()
        assert set(rec["source_id"].split(GRAPH_FIELD_SEP)) <= records.chunks.keys()
    for src, tgt in records.edges:
        assert src in records.nodes and tgt in records.nodes  # the merge drops dangling edges
    assert [c["chunk_order_index"] for c in records.chunks.values()] == list(range(len(records.chunks)))

    views = records.nodes["iccn_dwh.fact_berita.views"][0]
    assert "fact_berita" in records.chunks[views["source_id"]]["content"]


def test_multiple_schemas_keep_same_named_tables_apart_and_join_across():
    records = _records({**_two_schemas(), "name": "sales"})

    assert {"sales.orders", "crm.orders", "crm.customers"} <= records.nodes.keys()
    assert {"database:sales", "schema:sales"} <= records.nodes.keys()  # same name, still two nodes
    fk = records.edges[("crm.customers", "sales.orders")][0]
    assert fk["properties"]["join_on"] == ["sales.orders.customer_id = crm.customers.id"]
    assert records.stats == {
        "schemas": 2, "tables": 3, "columns": 4, "foreign_keys": 1, "business_concepts": 1, "domains": 0, "values": 0,
    }


def test_business_concept_carries_its_rule_and_links_every_column_with_its_table_card():
    records = _records(_two_schemas())

    concept = records.nodes["concept:Pelanggan Aktif"][0]
    assert concept["entity_type"] == "db_business_concept"
    assert concept["raw_name"].split(GRAPH_FIELD_SEP) == ["Active Customers", "Customer Aktif"]
    assert concept["properties"]["from"] == "crm.customers c JOIN sales.orders o ON o.customer_id = c.id"
    card = records.chunks[concept["source_id"]]["content"]
    assert "COUNT(DISTINCT c.id)" in card and "- o.id IS NOT NULL" in card and "Customer Aktif" in card

    for column in ("crm.customers.id", "sales.orders.customer_id"):
        edge = records.edges[("concept:Pelanggan Aktif", column)][0]
        assert edge["relation_type"] == "uses_column"
        table_card = records.nodes[column][0]["source_id"]
        assert set(edge["source_id"].split(GRAPH_FIELD_SEP)) == {concept["source_id"], table_card}
    assert "Pelanggan Aktif" in next(iter(records.chunks.values()))["content"]  # database overview
    assert "raw_name" not in records.nodes["crm.customers"][0]


def test_enum_values_and_domains_become_nodes_and_every_node_has_a_display_name():
    d = _two_schemas()
    d["domains"] = [{"name": "Penjualan", "description": "Pesanan dan pelanggan."}]
    orders = d["schemas"]["sales"]["tables"]["orders"]
    orders.update(label="Pesanan Penjualan", domain="Penjualan")
    orders["columns"] += [
        _column("status", type="TEXT", enum_values=["BARU", "LUNAS"]),
        _column("kota", type="TEXT", enum_values=[f"kota {i}" for i in range(_MAX_VALUE_NODES + 1)]),
    ]
    records = _records(d)

    value = records.nodes["value:sales.orders.status=LUNAS"][0]
    assert value["entity_type"] == "db_value" and value["properties"]["display_name"] == "LUNAS"
    assert value["source_id"] == records.nodes["sales.orders.status"][0]["source_id"]  # the table card
    assert records.edges[("sales.orders.status", "value:sales.orders.status=LUNAS")][0]["relation_type"] == "has_value"
    assert not any(name.startswith("value:sales.orders.kota=") for name in records.nodes)  # over the cap
    assert "kota 50" in records.chunks[value["source_id"]]["content"]  # still listed on the card

    domain = records.nodes["domain:Penjualan"][0]
    assert domain["entity_type"] == "db_domain" and "sales.orders" in domain["description"]
    assert records.edges[("domain:Penjualan", "sales.orders")][0]["relation_type"] == "covers_table"
    assert "Penjualan" in next(iter(records.chunks.values()))["content"]  # database overview

    assert records.nodes["sales.orders"][0]["properties"]["display_name"] == "Pesanan Penjualan"
    assert records.nodes["crm.orders"][0]["properties"]["display_name"] == "orders"
    assert all(r[0]["properties"]["display_name"] for r in records.nodes.values())


def _concept(d: dict) -> dict:
    return d["business_concepts"][0]


def test_oversized_card_is_split_with_its_heading_and_every_column_once():
    records = _records(EXAMPLE, max_tokens=600)

    parts = [c["content"] for c in records.chunks.values() if c["content"].startswith("# Tabel iccn_dwh.detail_program_ip_event")]
    assert len(parts) > 1 and all("(bagian " in p.split("\n", 1)[0] for p in parts)
    columns = [r[0] for r in records.nodes.values()
               if r[0]["entity_type"] == "db_column" and r[0]["properties"]["table_name"] == "detail_program_ip_event"]
    assert all(f"- {c['properties']['column_name']} (" in records.chunks[c["source_id"]]["content"] for c in columns)


@pytest.mark.parametrize("break_it", [
    lambda d: d["schemas"]["sales"]["tables"]["orders"].pop("columns"),
    lambda d: d["schemas"]["sales"]["tables"]["orders"].update(primary_keys=["nope"]),
    lambda d: d["schemas"]["sales"]["tables"]["orders"]["foreign_keys"][0].update(referred_columns=["nope"]),
    lambda d: d["schemas"]["crm"].update(name="sales"),
    lambda d: d["schemas"]["crm"]["tables"]["orders"].update(domain="nope"),
    lambda d: d.update(domains=[{"name": "Penjualan"}, {"name": "Penjualan"}]),
    lambda d: d["schemas"]["crm"]["tables"]["orders"]["columns"][0].update(enum_values=["A", "A"]),
    lambda d: _concept(d)["columns"].append("sales.orders.nope"),
    lambda d: _concept(d).update(time_column="sales.orders.nope"),
    lambda d: _concept(d).update(sql="", filters=[]),
    lambda d: _concept(d).pop("from"),
    lambda d: _concept(d).update(synonyms=[f"istilah {i}" for i in range(11)]),
    lambda d: d["business_concepts"].append({**_concept(d), "synonyms": []}),
    lambda d: d["business_concepts"].append({**_concept(d), "name": "Lain", "synonyms": ["active customers"]}),
])
def test_invalid_schema_is_rejected_before_anything_is_written(break_it):
    data = _two_schemas()
    break_it(data)
    with pytest.raises(ValueError):
        load_spec(data)


# Write flow


class _Store:
    def __init__(self):
        self.rows = {}

    async def upsert(self, data):
        self.rows.update(data)

    async def get_by_id(self, id_):
        return self.rows.get(id_)


class _Graph:
    def __init__(self, names=()):
        self.names = set(names)

    async def get_nodes_batch(self, ids):
        return {i: {} for i in ids if i in self.names}


class _Rag(_DbSchemaMixin):
    workspace = "ws"
    tokenizer = TOKENIZER
    embedding_token_limit = 8192
    entities_vdb = relationships_vdb = full_entities = full_relations = None
    llm_response_cache = entity_chunks = relation_chunks = None

    def __init__(self, graph_names=()):
        self.doc_status, self.text_chunks, self.chunks_vdb = _Store(), _Store(), _Store()
        self.chunk_entity_relation_graph = _Graph(graph_names)
        self.deleted, self.discarded = [], False

    def _build_global_config(self):
        return {}

    async def _insert_done(self):
        pass

    async def _discard_pending_index_ops(self):
        self.discarded = True

    async def adelete_by_doc_id(self, doc_id):
        self.deleted.append(doc_id)
        self.doc_status.rows.pop(doc_id, None)
        return SimpleNamespace(status="success", message="")


@pytest.fixture
def merged(monkeypatch):
    calls = []

    async def fake_merge(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(ingest, "merge_nodes_and_edges", fake_merge)
    return calls


def test_ingest_writes_chunks_merges_and_marks_processed(merged):
    rag = _Rag()
    result = asyncio.run(rag.ainsert_db_schema(load_spec(_two_schemas())))

    row = rag.doc_status.rows[result["doc_id"]]
    assert result["status"] == "processed" and row["status"] == DocStatus.PROCESSED
    assert row["chunks_list"] == list(rag.text_chunks.rows) == list(rag.chunks_vdb.rows)
    assert merged[0]["entity_resolver"] is None and merged[0]["doc_id"] == result["doc_id"]


def test_reingest_is_a_noop_when_unchanged_and_a_replace_when_changed(merged):
    rag = _Rag()
    data = _two_schemas()
    asyncio.run(rag.ainsert_db_schema(load_spec(data)))

    assert asyncio.run(rag.ainsert_db_schema(load_spec(data)))["status"] == "unchanged"
    assert rag.deleted == []

    data["description"] = "Gudang data, versi baru."
    assert asyncio.run(rag.ainsert_db_schema(load_spec(data)))["status"] == "processed"
    assert rag.deleted == [schema_document("dwh")[0]]


def test_names_owned_by_another_document_are_refused(merged):
    rag = _Rag(graph_names={"crm.customers"})
    with pytest.raises(ValueError, match="crm.customers"):
        asyncio.run(rag.ainsert_db_schema(load_spec(_two_schemas())))
    assert rag.doc_status.rows == {} and merged == []


def test_failed_write_discards_buffers_and_stays_failed(monkeypatch):
    async def broken_merge(**kwargs):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(ingest, "merge_nodes_and_edges", broken_merge)
    rag = _Rag()
    with pytest.raises(RuntimeError):
        asyncio.run(rag.ainsert_db_schema(load_spec(_two_schemas())))

    row = rag.doc_status.rows[schema_document("dwh")[0]]
    assert row["status"] == DocStatus.FAILED and "neo4j down" in row["error_msg"]
    assert rag.discarded
