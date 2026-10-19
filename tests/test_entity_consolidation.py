"""Entity resolution Fase 3: aliases on canonical nodes, the verdict cache, consolidation planning."""

import asyncio

import numpy as np
import pytest

from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.ingestion.knowledge_graph.entity_adjudication import (
    EntityAdjudicator,
    PairVerdictStore,
    Profile,
)
from hybrid_rag.ingestion.knowledge_graph.entity_consolidation import EntityConsolidator
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import EntityResolver, rewrite
from hybrid_rag.ingestion.knowledge_graph.extraction import _ontology_properties
from hybrid_rag.storage.postgres.sql import SQL_TEMPLATES
from test_entity_adjudication import Judge, survivor_rule

SEP = GRAPH_FIELD_SEP


# --- aliases -------------------------------------------------------------------------


def test_vdb_text_is_unchanged_without_aliases():
    assert EntityResolver.vdb_content("Kota Malang", "Kota di Jawa Timur.") == "Kota Malang\nKota di Jawa Timur."
    assert EntityResolver.vdb_content("Kota Malang", "Kota di Jawa Timur.", "") == "Kota Malang\nKota di Jawa Timur."


def test_aliases_come_before_the_description():
    text = EntityResolver.vdb_content("Indonesia Creative Cities Network", "Jejaring.", f"ICCN{SEP}Jejaring Kota Kreatif")
    assert text == "Indonesia Creative Cities Network\nAlias: ICCN, Jejaring Kota Kreatif\nJejaring."


def test_only_aliases_with_different_words_are_kept():
    records = [{"raw_name": "INDONESIA CREATIVE CITIES NETWORK"}, {"raw_name": "iccn"}, {"raw_name": "ICCN"}, {}]
    aliases = EntityResolver.merged_aliases("Indonesia Creative Cities Network", "Jejaring ICCN", records)
    assert aliases.split(SEP) == ["ICCN", "Jejaring ICCN"]
    many = [{"raw_name": f"Alias {i:02d}"} for i in range(30)]
    assert len(EntityResolver.merged_aliases("X", None, many).split(SEP)) == 10
    assert EntityResolver.merged_aliases("X", None, [{"raw_name": f"A{SEP}B"}]) == f"A{SEP}B"


def test_rewrite_remembers_the_raw_name():
    nodes, _ = rewrite({"ICCN": [{"entity_name": "ICCN"}], "Kota Malang": [{"entity_name": "Kota Malang"}]}, {},
                       {"ICCN": "Indonesia Creative Cities Network"})
    assert nodes["Indonesia Creative Cities Network"][0]["raw_name"] == "ICCN"
    assert "raw_name" not in nodes["Kota Malang"][0]


def test_an_ontology_property_named_aliases_cannot_clash():
    assert _ontology_properties([{"properties": {"aliases": "x"}}], None) == {"ontology_aliases": "x"}


# --- verdict cache ---------------------------------------------------------------------


class Store:
    """PairVerdictStore in memory."""

    def __init__(self):
        self.rows = {}

    pair_id = staticmethod(PairVerdictStore.pair_id)

    async def get(self, ids):
        return {i: self.rows[i] for i in set(ids) if i in self.rows}

    async def put(self, rows):
        for pid, a, b, same, confidence, preferred, reason in rows:
            self.rows[pid] = {"id": pid, "same": same, "confidence": confidence, "preferred": preferred, "reason": reason}


class Graph:
    def __init__(self, nodes=None, degrees=None):
        self.nodes, self.degrees, self.writes = nodes or {}, degrees or {}, []

    async def get_nodes_batch(self, names):
        return {n: self.nodes[n] for n in names if n in self.nodes}

    async def get_all_nodes(self):
        return [{**props, "id": name} for name, props in self.nodes.items()]

    async def node_degrees_batch(self, names):
        return {n: self.degrees.get(n, 0) for n in names}

    async def upsert_node(self, *args, **kwargs):
        self.writes.append(args)


def test_a_judged_pair_is_never_sent_to_the_llm_again():
    store, judge = Store(), Judge(survivor_rule(("Dewan Pakar", "Dewan Pakar ICCN", "Dewan Pakar ICCN", 0.97)))
    profiles = {n: Profile(n, "organization", f"Deskripsi {n}.") for n in ("Dewan Pakar", "Dewan Pakar ICCN")}
    adjudicator = EntityAdjudicator(graph=Graph(), chain=judge, verdicts=store)
    first = asyncio.run(adjudicator.adjudicate(dict(profiles), known=[]))
    asked = len(judge.seen)
    second = asyncio.run(adjudicator.adjudicate(dict(profiles), known=[]))
    assert asked == 1 and len(judge.seen) == asked  # second run answered from the store
    assert adjudicator.cached == 1 and first == second
    assert first["Dewan Pakar"].target == "Dewan Pakar ICCN"


def test_a_stored_threshold_confidence_still_merges():
    # Postgres REAL hands 0.95 back as 0.9499999880790710.
    store, judge = Store(), Judge(survivor_rule(("Dewan Pakar", "Dewan Pakar ICCN", "Dewan Pakar ICCN", 0.95)))
    profiles = {n: Profile(n, "organization", f"Deskripsi {n}.") for n in ("Dewan Pakar", "Dewan Pakar ICCN")}
    adjudicator = EntityAdjudicator(graph=Graph(), chain=judge, verdicts=store)
    asyncio.run(adjudicator.adjudicate(dict(profiles), known=[]))
    for row in store.rows.values():
        row["confidence"] = 0.949999988079071
    assert "Dewan Pakar" in asyncio.run(adjudicator.adjudicate(dict(profiles), known=[]))


# --- consolidation planning ----------------------------------------------------------------


class KV:
    def __init__(self, rows=None):
        self.rows = rows or {}

    async def get_by_ids(self, ids):
        return [self.rows.get(i) for i in ids]

    async def get_by_id(self, key):
        return self.rows.get(key)

    async def upsert(self, data):
        self.rows.update(data)


class VDB:
    async def get_vectors_by_ids(self, ids):
        return {i: np.array([1.0, float(n)]) for n, i in enumerate(ids)}


class DocStatus:
    def __init__(self, busy=None):
        self.busy = busy or {}

    async def get_docs_by_statuses(self, statuses):
        return self.busy


class Rag:
    workspace = "ws_test"

    def __init__(self, nodes, chunks, degrees=None, busy=None):
        self.chunk_entity_relation_graph = Graph(nodes, degrees)
        self.entity_chunks = KV(chunks)
        self.entities_vdb = VDB()
        self.doc_status = DocStatus(busy)
        self.entity_aliases = type("Registry", (), {"db": None})()


def consolidator(rag, judge):
    c = EntityConsolidator(rag, chain=judge)
    c.adjudicator.verdicts = Store()
    return c


def node(etype="organization", description="Dewan pakar ICCN."):
    return {"entity_type": etype, "description": description, "source_id": "c1"}


def test_dry_run_plans_merges_and_writes_nothing():
    rag = Rag({"Dewan Pakar": node(), "Dewan Pakar ICCN": node()},
              {"Dewan Pakar": {"chunk_ids": ["c1"]}, "Dewan Pakar ICCN": {"chunk_ids": ["c1", "c2", "c3"]}})
    judge = Judge(survivor_rule(("Dewan Pakar", "Dewan Pakar ICCN", "Dewan Pakar", 0.97)))
    report = asyncio.run(consolidator(rag, judge).run(dry_run=True))
    # The node with more sources survives even though the LLM preferred the other name.
    assert [(m.survivor, m.absorbed) for m in report.merges] == [("Dewan Pakar ICCN", "Dewan Pakar")]
    assert report.entities == 2 and report.pairs_judged == 1 and not report.applied
    assert rag.chunk_entity_relation_graph.writes == []


def test_equal_sources_fall_back_to_the_llm_preferred_name():
    rag = Rag({"Dewan Pakar": node(), "Dewan Pakar ICCN": node()},
              {"Dewan Pakar": {"chunk_ids": ["c1"]}, "Dewan Pakar ICCN": {"chunk_ids": ["c2"]}})
    judge = Judge(survivor_rule(("Dewan Pakar", "Dewan Pakar ICCN", "Dewan Pakar", 0.97)))
    report = asyncio.run(consolidator(rag, judge).run(dry_run=True))
    assert [(m.survivor, m.absorbed) for m in report.merges] == [("Dewan Pakar", "Dewan Pakar ICCN")]


def test_never_merge_of_every_registry_applies():
    rag = Rag({"Kemenparekraf": node(), "Kemenparekraf Bekraf": node()},
              {"Kemenparekraf": {"chunk_ids": ["c1"]}, "Kemenparekraf Bekraf": {"chunk_ids": ["c2"]}})
    vetoed = EntityConsolidator._never_merge()
    assert frozenset(("kemenparekraf", "bekraf")) in vetoed.never_merge
    assert EntityAdjudicator.is_vetoed(Profile("Kemenparekraf", "organization"), Profile("Bekraf", "organization"), vetoed)


def test_refuses_to_run_while_documents_are_being_ingested():
    rag = Rag({"A B": node()}, {}, busy={"doc-1": object()})
    with pytest.raises(RuntimeError, match="doc-1"):
        asyncio.run(consolidator(rag, Judge()).run(dry_run=True))


# --- merge pieces ------------------------------------------------------------------------


def test_absorbed_edge_keeps_its_ontology_types_and_weight_once():
    props = {"description": "d", "keywords": "k", "weight": 3.0,
             "relation_triples": ["Dewan Pakar|bagianDari|ICCN", "ICCN|memiliki|Dewan Pakar"]}
    records = EntityConsolidator._edge_records(props, "Dewan Pakar", "Dewan Pakar ICCN", "ICCN", ["c1"])
    assert [(r["src_id"], r["relation_type"], r["tgt_id"]) for r in records] == [
        ("Dewan Pakar ICCN", "bagianDari", "ICCN"), ("ICCN", "memiliki", "Dewan Pakar ICCN")]
    assert sum(r["weight"] for r in records) == 3.0
    plain = EntityConsolidator._edge_records({"weight": 2}, "A", "S", "N", ["c1"])
    assert plain == [{**plain[0], "src_id": "S", "tgt_id": "N", "weight": 2.0}] and "relation_type" not in plain[0]


def test_document_indexes_follow_the_merge():
    class R:
        text_chunks = KV({"c1": {"full_doc_id": "doc-1"}})
        full_entities = KV({"doc-1": {"entity_names": ["Dewan Pakar", "ICCN", "Dewan Pakar ICCN"], "count": 3}})
        full_relations = KV({"doc-1": {"relation_pairs": [["Dewan Pakar", "ICCN"], ["Dewan Pakar ICCN", "ICCN"],
                                                          ["Dewan Pakar", "Dewan Pakar ICCN"]], "count": 3}})

    c = EntityConsolidator.__new__(EntityConsolidator)
    c.rag = R()
    asyncio.run(c._repoint_documents("Dewan Pakar", "Dewan Pakar ICCN", ["c1"]))
    assert R.full_entities.rows["doc-1"]["entity_names"] == ["Dewan Pakar ICCN", "ICCN"]
    assert R.full_relations.rows["doc-1"]["relation_pairs"] == [["Dewan Pakar ICCN", "ICCN"]]


def test_new_queries_are_scoped_to_one_workspace():
    for key in ("repoint_entity_alias", "get_pair_verdicts"):
        assert "workspace=$1" in SQL_TEMPLATES[key], key
    assert "workspace" in SQL_TEMPLATES["upsert_pair_verdict"].split("VALUES")[0]
