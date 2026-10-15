"""The extraction contract: entity types outside the Literal are dropped per record, never per response."""

import json
import random
from typing import get_args

from hybrid_rag.ingestion.knowledge_graph.schema import (
    EntityExtractionOutput,
    EntityType,
    OntologyExtractionOutput,
)

ALLOWED = set(get_args(EntityType))


def entity(name, type_):
    return {"name": name, "type": type_, "description": f"Deskripsi {name}."}


def rel(source, target):
    return {"source": source, "target": target, "keywords": "mengatur", "description": "Terkait."}


def parse(payload, model=EntityExtractionOutput):
    # model_validate_json is what the OpenAI SDK calls on the provider's reply.
    return model.model_validate_json(json.dumps(payload))


def test_near_miss_types_map_to_the_official_type():
    out = parse({"entities": [
        entity("Kemenekraf", "organization "),
        entity("Dinas Pariwisata", "Organisasi"),
        entity("UU No. 24 Tahun 2019 - Pasal 9", "pasal"),
        entity("Menteri Ekonomi Kreatif", "Jabatan"),
    ], "relationships": []})
    assert [e.type for e in out.entities] == ["Organization", "Organization", "Article", "Role"]


def test_unknown_type_drops_the_entity_and_its_relations_not_the_response():
    out = parse({
        "entities": [
            entity("Kemenekraf", "Organization"),
            entity("Jakarta", "Lokasi"),
            entity("Subsektor Kuliner", "Subsector"),
        ],
        "relationships": [
            rel("Kemenekraf", "Jakarta"),
            rel("Kemenekraf", "Subsektor Kuliner"),
        ],
    })
    assert [e.name for e in out.entities] == ["Kemenekraf", "Subsektor Kuliner"]
    assert [(r.source, r.target) for r in out.relationships] == [("Kemenekraf", "Subsektor Kuliner")]


def test_every_surviving_entity_has_an_allowed_type():
    rng = random.Random(7)
    pool = [*ALLOWED, "organisasi", " PERSON", "Konsep", "Lokasi", "Tanggal", "Other", "", "UNKNOWN", 42, None]
    for _ in range(200):
        names = [f"E{i}" for i in range(rng.randint(0, 8))]
        types = {n: rng.choice(pool) for n in names}
        payload = {
            "entities": [entity(n, t) for n, t in types.items()],
            "relationships": [rel(rng.choice(names), rng.choice(names)) for _ in range(len(names))] if names else [],
        }
        out = parse(payload)
        kept = {e.name for e in out.entities}
        assert all(e.type in ALLOWED for e in out.entities)
        assert all(r.source in kept and r.target in kept for r in out.relationships if r.source in types and r.target in types)


def test_relation_to_a_name_that_is_also_kept_stays():
    # Same name twice, one valid: the name still exists, so relations to it are kept.
    out = parse({
        "entities": [entity("Kemenekraf", "Lokasi"), entity("Kemenekraf", "Organization"), entity("Rindekraf", "Program")],
        "relationships": [rel("Kemenekraf", "Rindekraf")],
    })
    assert [e.type for e in out.entities] == ["Organization", "Program"]
    assert len(out.relationships) == 1


def test_arrays_wrapped_in_a_json_string_are_still_recovered():
    out = parse({
        "entities": json.dumps([entity("Kemenekraf", "organisasi"), entity("Jakarta", "Lokasi")]),
        "relationships": json.dumps([rel("Kemenekraf", "Jakarta")]),
    })
    assert [(e.name, e.type) for e in out.entities] == [("Kemenekraf", "Organization")]
    assert out.relationships == []


def test_valid_response_is_unchanged():
    payload = {
        "entities": [entity("Teuku Riefky Harsya", "Person"), entity("Menteri Ekonomi Kreatif", "Role")],
        "relationships": [rel("Teuku Riefky Harsya", "Menteri Ekonomi Kreatif")],
    }
    assert parse(payload).model_dump() == payload


def test_ontology_labels_are_kept_verbatim():
    payload = {
        "entities": [
            {**entity("Kemenekraf", "organization"), "properties": []},
            {**entity("Jakarta", "Lokasi"), "properties": []},
        ],
        "relationships": [{**rel("Kemenekraf", "Jakarta"), "type": "BERLOKASI_DI", "properties": []}],
    }
    out = parse(payload, OntologyExtractionOutput)
    assert [(e.name, e.type) for e in out.entities] == [("Kemenekraf", "organization"), ("Jakarta", "Lokasi")]
    assert len(out.relationships) == 1


def test_relations_missing_required_fields_are_dropped_not_the_response():
    # The production error: relationships 13 and 14 came back with only source + description.
    good = [rel("Kemenekraf", "Subsektor Kuliner") for _ in range(13)]
    broken = [
        {"source": "Teknologi Digital", "description": "Menuju Indonesia Emas 2045."},
        {"source": "Platform Integrasi", "description": "Menuju Indonesia Emas 2045."},
    ]
    out = parse({
        "entities": [entity("Kemenekraf", "Organization"), entity("Subsektor Kuliner", "Subsector")],
        "relationships": good + broken,
    })
    assert len(out.entities) == 2
    assert len(out.relationships) == 13


def test_keywords_sent_as_a_list_are_joined():
    out = parse({
        "entities": [entity("Kemenekraf", "Organization"), entity("Rindekraf", "Program")],
        "relationships": [{**rel("Kemenekraf", "Rindekraf"), "keywords": ["menyusun", " menetapkan "]}],
    })
    assert out.relationships[0].keywords == "menyusun, menetapkan"


def test_entity_missing_a_field_is_dropped_with_its_relations():
    out = parse({
        "entities": [{"name": "Jakarta", "type": "Organization"}, entity("Kemenekraf", "Organization")],
        "relationships": [rel("Kemenekraf", "Jakarta")],
    })
    assert [e.name for e in out.entities] == ["Kemenekraf"]
    assert out.relationships == []


def test_unusable_relationships_array_keeps_the_entities():
    out = parse({"entities": [entity("Kemenekraf", "Organization")], "relationships": "not json"})
    assert [e.name for e in out.entities] == ["Kemenekraf"]
    assert out.relationships == []


def test_response_without_entities_array_still_fails():
    # Nothing to salvage: the chunk-level fallback in extraction handles this one.
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        parse({"relationships": []})


def test_ontology_record_missing_a_field_is_dropped_and_labels_stay_verbatim():
    out = parse({
        "entities": [
            {**entity("Kemenekraf", "organization"), "properties": []},
            entity("Jakarta", "Lokasi"),  # no properties: invalid for the ontology model
        ],
        "relationships": [],
    }, OntologyExtractionOutput)
    assert [(e.name, e.type) for e in out.entities] == [("Kemenekraf", "organization")]


# --- _extract_or_none: the chunk-level fallback ----------------------------------

import asyncio  # noqa: E402

import pytest  # noqa: E402

from hybrid_rag.ingestion.knowledge_graph import extraction  # noqa: E402


class ScriptedChain:
    """Returns or raises the scripted outcomes in order; counts calls."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    async def ainvoke(self, inputs, config=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _invalid_output_error():
    try:
        parse({"relationships": []})
    except Exception as e:
        return e


def run(chain):
    return asyncio.run(extraction._extract_or_none(chain, {}, {}, "chunk-1", "extraction"))


def test_invalid_output_is_retried_once_and_the_retry_is_used():
    good = parse({"entities": [entity("Kemenekraf", "Organization")], "relationships": []})
    chain = ScriptedChain(_invalid_output_error(), good)
    assert run(chain) is good
    assert chain.calls == 2


def test_output_still_invalid_after_retry_yields_none_not_an_error():
    chain = ScriptedChain(_invalid_output_error(), _invalid_output_error())
    assert run(chain) is None
    assert chain.calls == 2
    # None becomes "no nodes, no edges" for the chunk — nothing is written to the graph.
    assert extraction._parse_extraction_output(None, "chunk-1", 0) == ({}, {})


def test_infrastructure_errors_still_fail_the_document():
    chain = ScriptedChain(TimeoutError("provider down"))
    with pytest.raises(TimeoutError):
        run(chain)
    assert chain.calls == 1
