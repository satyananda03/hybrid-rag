import asyncio

import pytest
from langchain_core.runnables import RunnableLambda

from hybrid_rag import providers
from hybrid_rag.ingestion.knowledge_graph.chain import get_kg_chains
from hybrid_rag.ingestion.knowledge_graph.ontology_schema import (
    load_ontology_schema,
    prune_chunk,
    render_schema_for_prompt,
    resolve_ontology_path,
)


class _FakeLLM:
    def with_structured_output(self, model):
        return RunnableLambda(lambda _: model(entities=[], relationships=[]))


def _node(name, type_, **props):
    return {name: [{"entity_name": name, "entity_type": type_, "properties": props}]}


def _edge(src, tgt, type_):
    return {(src, tgt): [{"src_id": src, "tgt_id": tgt, "relation_type": type_, "properties": {}}]}


def test_ontology_loads_and_renders():
    schema = load_ontology_schema("hukum")
    assert {n.label for n in schema.node_types} >= {"Peraturan", "Pasal", "Lembaga"}
    prompt = render_schema_for_prompt(schema)
    # braces escaped for ChatPromptTemplate: formatting yields the plain dump
    assert prompt.format() == str(schema.model_dump(exclude_none=True))
    for bad in ("../hukum", "nope"):
        with pytest.raises((ValueError, FileNotFoundError)):
            resolve_ontology_path(bad)


def test_prune_keeps_schema_and_repairs_direction():
    nodes = {
        **_node("UU 1/2020", "Peraturan", tahun="2020"),
        **_node("Pasal 1", "Pasal"),
        **_node("Kucing", "Hewan"),  # not an ontology type
    }
    edges = {
        **_edge("Pasal 1", "UU 1/2020", "memilikiPasal"),  # reversed: repaired
        **_edge("UU 1/2020", "Kucing", "mengaturTentang"),  # endpoint pruned
    }
    kept_nodes, kept_edges = asyncio.run(prune_chunk(nodes, edges, load_ontology_schema("hukum"), "c1"))
    assert set(kept_nodes) == {"UU 1/2020", "Pasal 1"}
    assert kept_nodes["UU 1/2020"][0]["properties"] == {"tahun": "2020"}
    assert list(kept_edges) == [("UU 1/2020", "Pasal 1")]


def test_chains_per_ontology():
    providers.configure(get_llm=lambda **_: _FakeLLM())
    default, onto = get_kg_chains(), get_kg_chains("hukum")
    assert default.schema is None and get_kg_chains("") is default
    assert onto.schema is not None
    system = onto.extraction_prompt.messages[0].prompt.template
    assert "Peraturan" in system and "{ontology_schema}" not in system


def test_settings_require_names_missing_env(monkeypatch):
    from hybrid_rag.config import RagSettings

    s = RagSettings(_env_file=None, LLM_BINDING_HOST="h", LLM_BINDING_API_KEY="k")
    with pytest.raises(RuntimeError, match="EMBEDDING_BINDING_HOST, EMBEDDING_BINDING_API_KEY"):
        s.require()
