"""Entity resolution Fase 2: candidates, vetoes, LLM verdicts and star merges, offline."""

import asyncio
import json

import numpy as np
import pytest
from pydantic import ValidationError

from hybrid_rag.ingestion.doc_status_fields import doc_status_transition_metadata
from hybrid_rag.ingestion.knowledge_graph.entity_adjudication import (
    AdjudicationOutput,
    EntityAdjudicator,
    PairVerdict,
    Profile,
)
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import (
    EntityResolver,
    load_entity_registry,
    match_key,
)
from hybrid_rag.storage.postgres.sql import SQL_TEMPLATES


class Registry:
    """hybrid_rag_entity_alias for one workspace, in memory, same upsert rule."""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})  # id -> (canonical, source, confidence)

    async def get(self, ids):
        return {i: self.rows[i][0] for i in set(ids) if i in self.rows}

    async def add(self, decisions):
        for d in decisions:
            if d.alias_id not in self.rows or d.source == "user":
                self.rows[d.alias_id] = (d.canonical, d.source, d.confidence)

    async def known_entities(self):
        return sorted({(canonical, i.split(":", 1)[0]) for i, (canonical, *_) in self.rows.items()})


class Graph:
    def __init__(self, nodes=None):
        self.nodes = nodes or {}

    async def get_nodes_batch(self, names):
        return {n: self.nodes[n] for n in names if n in self.nodes}


class Judge:
    """Fake LLM: ``rule(a_name, b_name) -> (same, confidence, preferred)``; records pairs seen."""

    def __init__(self, rule=None, fail=None):
        self.rule = rule or (lambda a, b: (False, 0.9, "a"))
        self.fail = list(fail or [])
        self.seen = []

    async def ainvoke(self, inputs):
        if self.fail:
            raise self.fail.pop(0)
        verdicts = []
        for pair in json.loads(inputs["pairs"]):
            self.seen.append((pair["a"]["name"], pair["b"]["name"]))
            same, confidence, preferred = self.rule(pair["a"]["name"], pair["b"]["name"])
            verdicts.append({"pair_id": pair["id"], "same": same, "confidence": confidence,
                             "preferred": preferred, "reason": "uji"})
        return AdjudicationOutput.model_validate({"verdicts": verdicts})


def existing(name, etype="organization", description=""):
    """Registry row + graph node for an entity an earlier document wrote."""
    return {f"{etype}:{match_key(name)}": (name, "rule", None)}, {
        name: {"entity_type": etype, "description": description or f"Deskripsi {name}."}
    }


def same_if(*pairs, confidence=0.95, preferred="a"):
    wanted = {frozenset(p) for p in pairs}
    return lambda a, b: (frozenset((a, b)) in wanted, confidence, preferred)


def record(name, etype="organization"):
    return {"entity_name": name, "entity_type": etype, "description": f"Deskripsi {name}.", "source_id": "c1"}


def run(names, *, known=(), judge=None, use_llm=True, registry=None, edges=(), registry_name=None):
    """Resolve one document; ``known`` = entities earlier documents wrote."""
    rows, graph_nodes = {}, {}
    for args in known:
        r, n = existing(*args) if isinstance(args, tuple) else existing(args)
        rows.update(r)
        graph_nodes.update(n)
    registry = registry or Registry(rows)
    judge = judge or Judge()
    adjudicator = EntityAdjudicator(graph=Graph(graph_nodes), chain=judge)
    nodes = {}
    for name, etype in names:
        nodes.setdefault(name, []).append(record(name, etype))
    edge_map = {(s, t): [{"src_id": s, "tgt_id": t, "description": "x", "source_id": "c1"}] for s, t in edges}
    out_nodes, out_edges = asyncio.run(
        EntityResolver(registry, adjudicator).resolve(nodes, edge_map, registry_name, use_llm=use_llm)
    )
    return set(out_nodes), out_edges, registry, judge


# --- L3: candidates --------------------------------------------------------------


def proposed(name, others, etype="organization"):
    adjudicator = EntityAdjudicator(graph=Graph())
    pairs = adjudicator._propose_lexical({"new": Profile(name, etype)}, {o: etype for o in others})
    return {(p.other, p.signal) for p in pairs}


def test_qualifier_spelling_and_acronyms_become_candidates():
    assert ("Dewan Pakar ICCN", "containment") in proposed("Dewan Pakar", ["Dewan Pakar ICCN"])
    assert ("Dadik Wahyu Chang", "spelling") in proposed("Dadik Wahyu Cang", ["Dadik Wahyu Chang"], "person")
    assert ("Menteri Pemuda Dan Olahraga", "acronym") in proposed("Menpora", ["Menteri Pemuda Dan Olahraga"])
    assert ("BPS", "acronym") in proposed("Badan Pusat Statistik", ["BPS"])


def test_one_word_subsets_are_not_candidates():
    assert not any(o == "Dewan Pakar ICCN" for o, _ in proposed("ICCN", ["Dewan Pakar ICCN"]))


def test_incompatible_types_are_never_compared():
    adjudicator = EntityAdjudicator(graph=Graph())
    pairs = adjudicator._propose_lexical({"new": Profile("Dewan Pakar", "person")}, {"Dewan Pakar ICCN": "organization"})
    assert pairs == []


# --- L4: vetoes ---------------------------------------------------------------------


@pytest.mark.parametrize("a, b", [
    ("Kota Malang", "Kabupaten Malang"),
    ("Dinas Pariwisata Kota Malang", "Dinas Pariwisata Kota Batu"),
    ("Provinsi Sumatera Utara", "Provinsi Sumatera Barat"),
    ("Kota Kreatif", "Kota/Kabupaten Kreatif"),
    ("UU No. 24 Tahun 2019 Pasal 9", "UU No. 24 Tahun 2019 Pasal 10"),
    ("Dinas Pariwisata", "Dinas Pariwisata Kota Malang"),
    ("Badan Ekonomi Kreatif", "Dinas Ekonomi Kreatif"),
])
def test_provably_different_pairs_are_vetoed(a, b):
    assert EntityAdjudicator.is_vetoed(Profile(a, "organization"), Profile(b, "organization"))


@pytest.mark.parametrize("a, b", [
    ("Provinsi Sumatera Utara", "Sumatera Utara"),
    ("WCCE", "World Conference On Creative Economy 2026"),
    ("Disparbud Kota Malang", "Dinas Pariwisata Dan Kebudayaan Kota Malang"),
    ("Dewan Pakar", "Dewan Pakar ICCN"),
])
def test_ambiguous_pairs_reach_the_llm(a, b):
    assert not EntityAdjudicator.is_vetoed(Profile(a, "organization"), Profile(b, "organization"))


def test_type_and_never_merge_vetoes():
    assert EntityAdjudicator.is_vetoed(Profile("Malang", "person"), Profile("Malang Raya", "organization"))
    assert not EntityAdjudicator.is_vetoed(Profile("Asta Ekraf", "concept"), Profile("Asta Ekraf Program", "program"))
    domain = load_entity_registry("iccn")
    assert EntityAdjudicator.is_vetoed(Profile("Kemenparekraf", "organization"), Profile("Bekraf", "organization"), domain)


# --- L5/L6: verdicts and merges ------------------------------------------------------


def test_new_name_merges_into_existing_entity_with_llm_row():
    names, edges, registry, _ = run(
        [("Dewan Pakar", "organization"), ("ICCN", "organization")],
        known=["Dewan Pakar ICCN"],
        judge=Judge(same_if(("Dewan Pakar", "Dewan Pakar ICCN"))),
        edges=[("Dewan Pakar", "ICCN")],
    )
    assert names == {"Dewan Pakar ICCN", "ICCN"}
    assert set(edges) == {("Dewan Pakar ICCN", "ICCN")}
    canonical, source, confidence = registry.rows[f"organization:{match_key('Dewan Pakar')}"]
    assert (canonical, source, confidence) == ("Dewan Pakar ICCN", "llm", 0.95)
    assert registry.rows[f"organization:{match_key('Dewan Pakar ICCN')}"][1] == "rule"  # existing untouched


def test_low_confidence_and_unproposed_verdicts_are_ignored():
    names, *_ = run([("Dewan Pakar", "organization")], known=["Dewan Pakar ICCN"],
                    judge=Judge(same_if(("Dewan Pakar", "Dewan Pakar ICCN"), confidence=0.6)))
    assert names == {"Dewan Pakar"}

    class Inventing(Judge):
        async def ainvoke(self, inputs):
            return AdjudicationOutput.model_validate(
                {"verdicts": [{"pair_id": "p999", "same": True, "confidence": 1.0}]})

    names, *_ = run([("Dewan Pakar", "organization")], known=["Dewan Pakar ICCN"], judge=Inventing())
    assert names == {"Dewan Pakar"}


def survivor_rule(*pairs):
    """Judge rule: each (a, b, survivor, confidence) is the same entity, keeping ``survivor``."""
    table = {frozenset((a, b)): (survivor, conf) for a, b, survivor, conf in pairs}

    def rule(a, b):
        hit = table.get(frozenset((a, b)))
        if hit is None:
            return False, 0.9, "a"
        survivor, conf = hit
        return True, conf, "a" if a == survivor else "b"

    return rule


def test_new_vs_new_keeps_the_llm_preferred_name():
    short, long = "Pusat Kolaborasi Riset", "Pusat Kolaborasi Riset Ekonomi Kreatif"
    for survivor in (short, long):
        names, *_ = run([(short, "organization"), (long, "organization")],
                        judge=Judge(survivor_rule((short, long, survivor, 0.97))))
        assert names == {survivor}


def test_merges_never_chain():
    # A→B accepted first; B→C would make B both a target and a source: rejected.
    a, b, c = "Insentif Fiskal", "Skema Insentif Fiskal", "Skema Insentif Fiskal Nasional"
    names, *_ = run([(n, "concept") for n in (a, b, c)],
                    judge=Judge(survivor_rule((a, b, b, 0.99), (b, c, c, 0.97))))
    assert names == {b, c}


def test_llm_failure_keeps_the_rule_result():
    for error in (TimeoutError("provider down"), RuntimeError("500")):
        names, _, registry, _ = run([("Dewan Pakar", "organization")], known=["Dewan Pakar ICCN"],
                                    judge=Judge(same_if(("Dewan Pakar", "Dewan Pakar ICCN")), fail=[error]))
        assert names == {"Dewan Pakar"}
        assert not any(source == "llm" for _, source, _ in registry.rows.values())


def test_invalid_output_is_retried_once():
    try:
        PairVerdict.model_validate({})
    except ValidationError as e:
        invalid = e
    names, *_ = run([("Dewan Pakar", "organization")], known=["Dewan Pakar ICCN"],
                    judge=Judge(same_if(("Dewan Pakar", "Dewan Pakar ICCN")), fail=[invalid]))
    assert names == {"Dewan Pakar ICCN"}


def test_without_use_llm_the_llm_is_never_called():
    names, _, _, judge = run([("Dewan Pakar", "organization")], known=["Dewan Pakar ICCN"],
                             judge=Judge(same_if(("Dewan Pakar", "Dewan Pakar ICCN"))), use_llm=False)
    assert names == {"Dewan Pakar"} and judge.seen == []


def test_registry_entities_without_a_graph_node_are_not_targets():
    rows, _ = existing("Dewan Pakar ICCN")  # deleted: registry row left, node gone
    names, _, _, judge = run([("Dewan Pakar", "organization")], registry=Registry(rows),
                             judge=Judge(same_if(("Dewan Pakar", "Dewan Pakar ICCN"))))
    assert names == {"Dewan Pakar"} and judge.seen == []


def test_rebuild_maps_raw_names_to_the_llm_target():
    _, _, registry, _ = run([("Dewan Pakar", "organization")], known=["Dewan Pakar ICCN"],
                            judge=Judge(same_if(("Dewan Pakar", "Dewan Pakar ICCN"))))
    mapping = asyncio.run(EntityResolver(registry).canonical_names([{"Dewan Pakar": [record("Dewan Pakar")]}]))
    assert mapping == {"Dewan Pakar": "Dewan Pakar ICCN"}


def test_embedding_candidates_reach_the_llm():
    class VDB:
        _max_batch_size = 8

        async def embedding_func(self, texts, context):
            return np.array([[1.0, 0.0] for _ in texts])

        async def nearest_entities(self, vectors, top_k, max_distance):
            return [[("Kementerian Pemuda Dan Olahraga", 0.1)] for _ in vectors]

    rows, nodes = existing("Kementerian Pemuda Dan Olahraga")
    judge = Judge(same_if(("Kemenpora", "Kementerian Pemuda Dan Olahraga")))
    adjudicator = EntityAdjudicator(graph=Graph(nodes), entity_vdb=VDB(), chain=judge)
    out, _ = asyncio.run(EntityResolver(Registry(rows), adjudicator).resolve(
        {"Kemenpora": [record("Kemenpora")]}, {}, use_llm=True))
    assert set(out) == {"Kementerian Pemuda Dan Olahraga"}


# --- workspace isolation & plumbing ----------------------------------------------------


def test_every_registry_and_candidate_query_is_scoped_to_one_workspace():
    for key in ("get_entity_aliases", "known_entity_aliases", "entities_nearest"):
        assert "workspace" in SQL_TEMPLATES[key] and "$1" in SQL_TEMPLATES[key], key
    assert "workspace" in SQL_TEMPLATES["upsert_entity_alias"].split("VALUES")[0]


def test_another_workspace_is_never_a_candidate():
    rows_a, nodes_a = existing("Dewan Pakar ICCN")
    judge = Judge(same_if(("Dewan Pakar", "Dewan Pakar ICCN")))
    # Workspace B has its own registry and graph, both empty: workspace A's entity is invisible.
    adjudicator_b = EntityAdjudicator(graph=Graph(), chain=judge)
    registry_b = Registry()
    out, _ = asyncio.run(EntityResolver(registry_b, adjudicator_b).resolve(
        {"Dewan Pakar": [record("Dewan Pakar")]}, {}, use_llm=True))
    assert set(out) == {"Dewan Pakar"} and judge.seen == []
    assert all(canonical != "Dewan Pakar ICCN" for canonical, *_ in registry_b.rows.values())


def test_llm_flag_survives_status_transitions():
    class Doc:
        metadata = {"entity_resolution": True, "entity_resolution_llm": False}

    assert doc_status_transition_metadata(Doc())["entity_resolution_llm"] is False
