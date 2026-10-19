"""Entity resolution Fase 1: the cases of docs/entity-deduplication/00-ringkasan-dan-rekomendasi.md §6.4/§6.7."""

import asyncio

import pytest

from hybrid_rag.ingestion.knowledge_graph.entity_resolution import (
    EntityResolver,
    entity_key,
    load_entity_registry,
    match_key,
    split_long_abbr,
)


class FakeAliasRegistry:
    """Dict stand-in for hybrid_rag_entity_alias with the same upsert rule."""

    def __init__(self):
        self.rows = {}  # id -> (canonical, source)

    async def get(self, ids):
        return {i: self.rows[i][0] for i in set(ids) if i in self.rows}

    async def add(self, decisions):
        for d in decisions:
            if d.alias_id not in self.rows or d.source == "user":
                self.rows[d.alias_id] = (d.canonical, d.source)


def node(name, type_="organization"):
    return {"entity_name": name, "entity_type": type_, "description": f"Deskripsi {name}.", "source_id": "c1"}


def edge(src, tgt):
    return {"src_id": src, "tgt_id": tgt, "description": "Terkait.", "source_id": "c1"}


def resolve(names, edges=(), registry_name=None, store=None):
    """names: [(name, type)] -> (node names after resolution, edges, store)."""
    store = store or FakeAliasRegistry()
    nodes = {}
    for name, type_ in names:
        nodes.setdefault(name, []).append(node(name, type_))
    edge_map = {(s, t): [edge(s, t)] for s, t in edges}
    out_nodes, out_edges = asyncio.run(
        EntityResolver(store).resolve(nodes, edge_map, registry_name)
    )
    return set(out_nodes), out_edges, store


# --- keys ------------------------------------------------------------------------


def test_punctuation_splits_numbers_instead_of_joining_them():
    assert entity_key("Lampiran 1.2") != entity_key("Lampiran 12")


def test_match_key_ignores_word_order_and_case_but_not_digit_order():
    assert match_key("Kota Malang") == match_key("Malang Kota") == match_key("KOTA MALANG")
    assert match_key("Pasal 9 Ayat 1") != match_key("Pasal 1 Ayat 9")
    assert match_key("Pasal 9") != match_key("Pasal 10")


def test_names_empty_after_normalisation_get_no_key():
    assert match_key("—") == match_key("()") == ""


def test_non_latin_names_keep_their_letters():
    assert entity_key("東京") == "東京"
    assert match_key("東京") != match_key("大阪")


def test_abbreviation_must_be_the_initials():
    assert split_long_abbr("Indonesia Creative Cities Networks (ICCN)") == (
        "Indonesia Creative Cities Networks", "ICCN")
    assert split_long_abbr("Usaha Mikro, Kecil, Dan Menengah (UMKM)") == (
        "Usaha Mikro, Kecil, Dan Menengah", "UMKM")
    assert split_long_abbr("Kota Malang (JATIM)") is None
    assert split_long_abbr("Rencana Induk (table)") is None
    assert split_long_abbr("Pasal 9 (1)") is None


def test_unknown_registry_lists_the_available_ones():
    with pytest.raises(FileNotFoundError, match="iccn"):
        load_entity_registry("tidak-ada")


# --- merges ----------------------------------------------------------------------


def test_long_abbr_pattern_merges_abbreviation_and_long_form():
    names, _, _ = resolve([
        ("ICCN", "organization"),
        ("Indonesia Creative Cities Networks (ICCN)", "organization"),
        ("Indonesia Creative Cities Networks", "organization"),
    ])
    assert names == {"Indonesia Creative Cities Networks"}


def test_long_abbr_alone_registers_the_long_form_for_later_documents():
    names, _, store = resolve([("Badan Pusat Statistik (BPS)", "organization")])
    assert names == {"Badan Pusat Statistik"}
    names, _, _ = resolve([("BPS", "organization"), ("badan pusat statistik", "organization")], store=store)
    assert names == {"Badan Pusat Statistik"}


def test_word_order_and_case_variants_merge():
    names, _, _ = resolve([("Kota Malang", "location"), ("Malang Kota", "location"), ("KOTA MALANG", "location")])
    assert names == {"Kota Malang"}


def test_yaml_alias_merges_what_no_rule_can():
    names, _, _ = resolve(
        [("ICCN", "organization"), ("Indonesia Creative Cities Network", "organization"),
         ("Kemenparekraf", "organization")],
        registry_name="iccn",
    )
    assert names == {"Indonesia Creative Cities Network", "Kementerian Pariwisata dan Ekonomi Kreatif"}


def test_applied_yaml_alias_holds_for_later_documents_without_the_yaml():
    _, _, store = resolve([("ICCN", "organization")], registry_name="iccn")
    names, _, _ = resolve([("ICCN", "organization")], store=store)
    assert names == {"Indonesia Creative Cities Network"}


def test_later_document_resolves_to_the_first_documents_name():
    _, _, store = resolve([("Kota Malang", "location")])
    names, _, _ = resolve([("Malang Kota", "location")], store=store)
    assert names == {"Kota Malang"}


def test_rebuild_maps_raw_names_with_registry_lookup_only():
    _, _, store = resolve([("ICCN", "organization"), ("Indonesia Creative Cities Networks (ICCN)", "organization")])
    resolver = EntityResolver(store)
    mapping = asyncio.run(resolver.canonical_names([{"ICCN": [node("ICCN")], "Pasal 9": [node("Pasal 9")]}]))
    assert mapping == {"ICCN": "Indonesia Creative Cities Networks"}
    assert asyncio.run(resolver.canonical_names([{"Kemenparekraf": [node("Kemenparekraf")]}])) == {}


# --- must stay apart -------------------------------------------------------------


@pytest.mark.parametrize("a, b", [
    (("Kota Malang", "location"), ("Kabupaten Malang", "location")),
    (("Kota Malang", "location"), ("Kota Malang Raya", "location")),
    (("Kota Malang", "location"), ("Pemerintah Kota Malang", "organization")),
    (("UU No. 24 Tahun 2019 Pasal 9", "article"), ("UU No. 24 Tahun 2019 Pasal 10", "article")),
    (("Lampiran 1.2", "concept"), ("Lampiran 12", "concept")),
    (("Pasal 9 Ayat 1", "article"), ("Pasal 1 Ayat 9", "article")),
    (("東京", "location"), ("大阪", "location")),
    (("Kementerian Keuangan", "organization"), ("Kementerian Perdagangan", "organization")),
])
def test_similar_but_different_entities_stay_apart(a, b):
    names, _, _ = resolve([a, b])
    assert names == {a[0], b[0]}


def test_same_tokens_of_another_type_stay_apart():
    names, _, _ = resolve([("Kota Malang", "location"), ("Malang Kota", "organization")])
    assert names == {"Kota Malang", "Malang Kota"}


def test_ambiguous_abbreviation_joins_neither_long_form():
    names, _, _ = resolve([
        ("Badan Pusat Statistik (BPS)", "organization"),
        ("Balai Pelestarian Sejarah (BPS)", "organization"),
        ("BPS", "organization"),
    ])
    assert names == {"Badan Pusat Statistik", "Balai Pelestarian Sejarah", "BPS"}


def test_never_merge_beats_a_yaml_alias(tmp_path, monkeypatch):
    from hybrid_rag.ingestion.knowledge_graph import entity_resolution

    (tmp_path / "uji.yaml").write_text(
        "global:\n  - canonical: Kementerian Ekonomi Kreatif\n    aliases: [Kemenparekraf]\n"
        "never_merge:\n  - [Kemenparekraf, Kementerian Ekonomi Kreatif]\n"
    )
    monkeypatch.setattr(entity_resolution, "REGISTRY_DIR", tmp_path)
    names, _, _ = resolve([("Kemenparekraf", "organization")], registry_name="uji")
    assert names == {"Kemenparekraf"}


def test_empty_and_multimodal_names_are_left_alone():
    names, _, store = resolve([("—", "concept"), ("Rencana Induk (table)", "table"), ("drawing-p3-1", "drawing")])
    assert names == {"—", "Rencana Induk (table)", "drawing-p3-1"}
    assert store.rows == {}


# --- edges -----------------------------------------------------------------------


def test_edges_follow_the_merge_and_new_self_loops_drop():
    _, edges, _ = resolve(
        [("Kota Malang", "location"), ("Malang Kota", "location"), ("ICCN", "organization"),
         ("Tabel 1 (table)", "table")],
        edges=[("Malang Kota", "ICCN"), ("Kota Malang", "Malang Kota"), ("Tabel 1 (table)", "Malang Kota")],
    )
    assert set(edges) == {("ICCN", "Kota Malang"), ("Kota Malang", "Tabel 1 (table)")}
    (record,) = edges[("ICCN", "Kota Malang")]
    assert (record["src_id"], record["tgt_id"]) == ("Kota Malang", "ICCN")


# --- institutions and connectors -------------------------------------------------


def test_institution_names_keep_prefix_and_word_order():
    assert match_key("Dinas Pariwisata, Ekonomi Kreatif, Kepemudaan Dan Olahraga") != match_key(
        "Dinas Kepemudaan, Olahraga, Pariwisata, Dan Ekonomi Kreatif")
    assert match_key("Dinas Pariwisata") != match_key("Suku Dinas Pariwisata")
    assert match_key("Badan Ekonomi Kreatif") != match_key("Ekonomi Kreatif")
    assert match_key("DINAS PARIWISATA KOTA MALANG") == match_key("Dinas Pariwisata Kota Malang")


def test_connector_words_do_not_change_identity():
    assert match_key("Badan Riset Dan Inovasi Nasional") == match_key("Badan Riset Inovasi Nasional")
    assert match_key("Perpres No 12 Tahun 2025") == match_key("Perpres Nomor 12 Tahun 2025")
    assert match_key("Strategi Komunikasi & Narasi") == match_key("Strategi Komunikasi Dan Narasi")
    assert match_key("Dan") == ""


def test_regional_offices_with_reordered_names_stay_apart():
    names, _, _ = resolve([
        ("Dinas Pariwisata, Ekonomi Kreatif, Kepemudaan Dan Olahraga", "organization"),
        ("Dinas Kepemudaan, Olahraga, Pariwisata, Dan Ekonomi Kreatif", "organization"),
    ])
    assert len(names) == 2


def test_rebuild_skips_units_of_documents_ingested_without_resolution():
    from hybrid_rag.ingestion.knowledge_graph.extraction import _units_ingested_without_resolution

    class KV:
        def __init__(self, rows):
            self.rows = rows

        async def get_by_ids(self, ids):
            return [self.rows.get(i) for i in ids]

    chunks = KV({"u-on": {"full_doc_id": "doc-on"}, "u-off": {"full_doc_id": "doc-off"},
                 "u-old": {"full_doc_id": "doc-old"}})
    docs = KV({"doc-on": {"metadata": {"entity_resolution": True}},
               "doc-off": {"metadata": {"entity_resolution": False}},
               "doc-old": {"metadata": {}}})  # ingested before the flag existed: default on
    off = asyncio.run(_units_ingested_without_resolution(["u-on", "u-off", "u-old"], chunks, docs))
    assert off == {"u-off"}
