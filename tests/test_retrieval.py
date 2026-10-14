import asyncio
import pytest
from hybrid_rag.config import settings
from hybrid_rag.core.base import collection_suffix
from hybrid_rag.core.namespace import NameSpace
from hybrid_rag.retrieval import context
from hybrid_rag.retrieval.pipeline import _retrieve
from hybrid_rag.retrieval.query import _normalize_keyword_list
from hybrid_rag.retrieval.types import QueryParam
from hybrid_rag.retrieval.utils import generate_reference_list_from_chunks, pick_by_weighted_polling
from hybrid_rag.storage.postgres.sql import SQL_TEMPLATES
from hybrid_rag.storage.postgres.vector import vector_table

class FakeKV:
    def __init__(self, rows):
        self.rows, self.calls = rows, 0
    async def get_by_ids(self, ids):
        self.calls += 1
        return [self.rows.get(i) for i in ids]

ROWS = {
    "c1": {"sidecar": {"page_start": 4, "page_end": 4, "parent_id": "p1"}},
    "c2": {"sidecar": {"page_start": 5, "page_end": 5, "parent_id": "p1"}},
    "c3": {"sidecar": {"page_start": "Rekap", "page_end": "Rekap"}},
    "p1": {"content": "parent", "sidecar": {"page_start": 4, "page_end": 5}},
}
CHUNKS = [{"chunk_id": c, "content": c, "file_path": "f"} for c in ("c1", "c2", "c3")]
IMAGE_CHUNK = {
    "chunk_id": "c3",
    "content": "Image Content Analysis:\nImage Path: https://s3/x/fig.jpg?sig=1\nCaptions: None",
    "file_path": "f",
}

def attach(db, **kwargs):
    return asyncio.run(context._attach_sources(CHUNKS, db, limit=None, **kwargs))

def build_context(include_citations, chunks=CHUNKS):
    return context._build_context_str(
        [], [], chunks, "q", QueryParam(enable_rerank=False, include_citations=include_citations),
        {}, text_chunks_db=FakeKV(ROWS),
    )

class FakeRag:
    def __init__(self, include_citations):
        self.include_citations = include_citations
    async def aretrieve(self, query, param):
        text, data = await build_context(self.include_citations, CHUNKS[:2] + [IMAGE_CHUNK])
        return {**data, "context": text}
def test_references_are_numbered_per_file_and_location():
    chunks = [
        {"file_path": "a.pdf", "page_start": 6, "page_end": 6},
        {"file_path": "b.xlsx", "page_start": "Rekap", "page_end": "Rekap"},
        {"file_path": "b.xlsx", "page_start": "Rekap", "page_end": "Rekap"},
        {"file_path": "a.pdf", "page_start": 2, "page_end": 3},
        {"file_path": "c.md"},
        {"file_path": "unknown_source"},
    ]
    refs, out = generate_reference_list_from_chunks(chunks)
    assert refs == [
        {"reference_id": "1", "file_path": "b.xlsx", "location": 'sheet "Rekap"'},
        {"reference_id": "2", "file_path": "a.pdf", "location": "hal. 7"},
        {"reference_id": "3", "file_path": "a.pdf", "location": "hal. 3–4"},
        {"reference_id": "4", "file_path": "c.md"},
    ]
    assert [c["reference_id"] for c in out] == ["2", "1", "1", "3", "4", ""]

def test_attach_sources_returns_child_or_merged_parent_chunks():
    child = attach(FakeKV(ROWS), return_parent=False, include_citations=True)
    assert [(c["content"], c["page_start"], c["page_end"]) for c in child] == [
        ("c1", 4, 4), ("c2", 5, 5), ("c3", "Rekap", "Rekap"),
    ]
    parent = attach(FakeKV(ROWS), return_parent=True, include_citations=True)
    assert [(c["chunk_id"], c["content"], c["page_start"], c["page_end"]) for c in parent] == [
        ("c1", "parent", 4, 5), ("c3", "c3", "Rekap", "Rekap"),
    ]

def test_attach_sources_does_no_lookup_without_citations():
    db = FakeKV(ROWS)
    assert attach(db, return_parent=False, include_citations=False) == CHUNKS
    assert db.calls == 0

def test_context_includes_citations_only_when_asked():
    text, data = asyncio.run(build_context(True))
    assert "[1] f, hal. 5\n" in text and 'sheet "Rekap"' in text
    assert data["data"]["chunks"][0]["page_start"] == 4

    text, data = asyncio.run(build_context(False))
    assert "Reference Document List" not in text and "reference_id" not in text
    assert data["data"]["references"] == []
    assert set(data["data"]["chunks"][0]) == {"content", "file_path", "chunk_id"}

def test_images_follow_their_citation():
    result = asyncio.run(_retrieve(FakeRag(True), "q", QueryParam(), include_raw_data=True))
    assert result.images == [{"path": "https://s3/x/fig.jpg?sig=1", "reference_id": "3"}]
    assert '[3] f, sheet "Rekap"' in result.context  # the image's page, via its reference
    assert "image_path" not in result.raw_data["data"]["chunks"][0]
    result = asyncio.run(_retrieve(FakeRag(False), "q", QueryParam(), include_raw_data=False))
    assert result.images == [{"path": "https://s3/x/fig.jpg?sig=1"}]

def test_query_param_defaults_come_from_settings():
    param = QueryParam()
    assert (param.top_k, param.chunk_top_k) == (settings.top_k, settings.chunk_top_k)
    assert param.enable_rerank is settings.rerank_by_default
    assert param.include_citations is True and param.return_parent is False

@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, []),
        (5, []),  
        ("a, b;c\nd", ["a", "b", "c", "d"]),  
        (["x, y", " z ", 3, ""], ["x, y", "z"]), 
    ],
)
def test_keyword_payloads_are_normalized(raw, expected):
    assert _normalize_keyword_list(raw, "field") == expected

def test_weighted_polling_gives_more_chunks_to_more_important_items():
    items = [{"sorted_chunks": ["a1", "a2", "a3"]}, {"sorted_chunks": ["b1", "b2", "b3"]}]
    assert pick_by_weighted_polling(items, max_related_chunks=3, min_related_chunks=1) == ["a1", "a2", "a3", "b1"]

def test_weighted_polling_hands_unused_quota_to_the_next_items():
    items = [{"sorted_chunks": ["a1"]}, {"sorted_chunks": ["b1", "b2", "b3"]}]
    assert pick_by_weighted_polling(items, max_related_chunks=3, min_related_chunks=1) == ["a1", "b1", "b2", "b3"]
    assert pick_by_weighted_polling([], max_related_chunks=3) == []

@pytest.mark.integration
def test_hybrid_search_stays_in_its_workspace_and_runs_a_top_k_scan(pg):
    schema = settings.postgres_schema
    suffix = collection_suffix(settings.embedding_model, settings.embedding_dim)
    table = f"{schema}.{vector_table(NameSpace.VECTOR_STORE_CHUNKS, suffix)[0]}"
    sql = SQL_TEMPLATES["chunks_hybrid_bm25"].format(table_name=table, vector_cast="vector")
    try:
        samples = pg(
            f"SELECT DISTINCT ON (workspace) workspace, id, content_vector, content_cleaned FROM {table} "
            "WHERE content_cleaned <> '' ORDER BY workspace, id"
        )
        pg(f"SELECT bm25_key FROM {table} LIMIT 1")
    except Exception as e:
        pytest.skip(f"{table} is missing or outdated, run `make db-init`: {type(e).__name__}")
    if not samples:
        pytest.skip(f"{table} has no chunks")
    for sample in samples:
        workspace = sample["workspace"]
        own_ids = {r["id"] for r in pg(f"SELECT id FROM {table} WHERE workspace = $1", workspace)}
        query = " ".join(sample["content_cleaned"].split()[:2])
        args = (workspace, 1 - settings.cosine_threshold, 30, sample["content_vector"], query, 1.0, 0.8)
        rows = pg(sql, *args)
        assert {r["id"] for r in rows} <= own_ids
        assert sample["id"] in {r["id"] for r in rows}
        plan = "\n".join(r[0] for r in pg("EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF) " + sql, *args))
        assert "TopKScanExecState" in plan 
        assert '"term":{"field":"workspace"' in plan and "heap_filter" not in plan

# --- reranker wiring --------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, host, built",
    [("rr", "http://rr", True), (None, "http://rr", False), ("rr", None, False)],
)
def test_reranker_is_built_from_settings_only_when_fully_configured(model, host, built, monkeypatch):
    from hybrid_rag import providers
    from hybrid_rag.retrieval import factory

    calls = []
    monkeypatch.setitem(providers._hooks, "get_reranker", lambda *a, **kw: calls.append((a, kw)) or "rerank-fn")
    monkeypatch.setattr(settings, "rerank_model", model)
    monkeypatch.setattr(settings, "rerank_binding_host", host)
    monkeypatch.setattr(settings, "rerank_binding_api_key", "rr-key")
    monkeypatch.setattr(settings, "rerank_timeout", 12.0)
    monkeypatch.setattr(settings, "max_retries", 4)
    monkeypatch.setattr(type(settings), "require", lambda self, *roles: None)

    class StubRag:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def initialize_storages(self):
            pass

    monkeypatch.setattr(factory, "RetrievalRAG", StubRag)
    monkeypatch.setattr(factory, "build_embedding_func", lambda: object())
    rag = asyncio.run(factory.build_rag("ws"))

    assert rag.kwargs["rerank_model_func"] == ("rerank-fn" if built else None)
    assert calls == ([(("rr",), {"base_url": "http://rr", "api_key": "rr-key", "timeout": 12.0, "max_retries": 4})] if built else [])
# --- context rendering ------------------------------------------------------------------

from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.ingestion.multimodal.multimodal_template import TEMPLATES as CHUNK_TEMPLATES


def image_content(neighbor):
    return CHUNK_TEMPLATES["image_chunk"].format(
        section_path="Timeline", neighbor_text=neighbor, image_path="https://s3/x/a.jpg",
        captions="None", footnotes="None", enhanced_caption="Pria berpidato.\nTahun 2026.",
    )


def render(chunks, rows, entities=(), originals=None):
    return asyncio.run(context._build_context_str(
        list(entities), [], chunks, "q", QueryParam(enable_rerank=False, include_citations=True),
        {}, entity_id_to_original=originals, text_chunks_db=FakeKV(rows),
    ))[0]


def test_image_neighbor_text_is_dropped_only_when_its_page_is_already_in_context():
    rows = {"t": {"sidecar": {"page_start": 0, "page_end": 0}}, "i": {"sidecar": {"page_start": 0, "page_end": 0}},
            "j": {"sidecar": {"page_start": 1, "page_end": 1}}}
    chunks = [
        {"chunk_id": "t", "content": "teks halaman satu", "file_path": "f"},
        {"chunk_id": "i", "content": image_content("Teks\n\ntetangga satu"), "file_path": "f"},
        {"chunk_id": "j", "content": image_content("Teks tetangga dua"), "file_path": "f"},
    ]
    text = render(chunks, rows)
    assert '<chunk ref_id="1" source="f, hal. 1" type="image" image_url="https://s3/x/a.jpg">' in text
    assert "tetangga satu" not in text  # page 0 is carried by chunk t
    assert "Neighbor Text: Teks tetangga dua" in text  # page 1 has no text chunk
    assert "[Section: Timeline]\n" in text and "Captions" not in text
    assert "Visual Analysis: Pria berpidato.\nTahun 2026.\n</chunk>" in text


def test_kg_lines_are_one_line_and_cite_only_chunks_in_context():
    rows = {"t": {"sidecar": {"page_start": 0, "page_end": 0}}}
    entities = [
        {"entity": "WCCE", "type": "event", "description": f"Konferensi.\n\nKe-5{GRAPH_FIELD_SEP}Di Jakarta"},
        {"entity": "Lain", "type": "concept", "description": "x"},
    ]
    originals = {"WCCE": {"source_id": f"t{GRAPH_FIELD_SEP}gone"}, "Lain": {"source_id": "gone"}}
    text = render([{"chunk_id": "t", "content": "t", "file_path": "f"}], rows, entities, originals)
    assert "- **WCCE** (event): Konferensi. Ke-5 Di Jakarta [Ref: 1]\n" in text
    assert "- **Lain** (concept): x\n" in text
    assert GRAPH_FIELD_SEP not in text
    assert text.rstrip().endswith("<references>\n[1] f, hal. 1\n</references>\n</context>")


# --- slide images -----------------------------------------------------------------------

SLIDE_ROWS = {
    "s1": {"sidecar": {"page_start": 0, "page_end": 0, "images": ["https://s3/slide_0.jpg"]}},
    "s2": {"sidecar": {"page_start": 0, "page_end": 1, "images": ["https://s3/slide_0.jpg", "https://s3/slide_1.jpg"]}},
}
SLIDE_CHUNKS = [{"chunk_id": c, "content": c, "file_path": "deck.pdf"} for c in ("s1", "s2")]


def test_slide_images_follow_the_stored_sidecar():
    chunks = asyncio.run(context._attach_sources(
        SLIDE_CHUNKS, FakeKV(SLIDE_ROWS), return_parent=False, include_citations=True, limit=None,
    ))
    assert [c["images"] for c in chunks] == [SLIDE_ROWS[c]["sidecar"]["images"] for c in ("s1", "s2")]
    assert all("images" not in c for c in attach(FakeKV(ROWS), return_parent=False, include_citations=True))


def test_each_slide_is_shown_once_with_the_citation_of_its_first_chunk():
    class SlideRag:
        async def aretrieve(self, query, param):
            text, data = await context._build_context_str(
                [], [], SLIDE_CHUNKS + [IMAGE_CHUNK], "q", QueryParam(enable_rerank=False), {},
                text_chunks_db=FakeKV({**SLIDE_ROWS, **ROWS}),
            )
            return {**data, "context": text}

    result = asyncio.run(_retrieve(SlideRag(), "q", QueryParam(), include_raw_data=True))
    refs = {c["chunk_id"]: c["reference_id"] for c in result.raw_data["data"]["chunks"]}
    assert result.images == [
        {"path": "https://s3/x/fig.jpg?sig=1", "reference_id": refs["c3"]},
        {"path": "https://s3/slide_0.jpg", "reference_id": refs["s1"]},
        {"path": "https://s3/slide_1.jpg", "reference_id": refs["s2"]},
    ]
    assert f'[{refs["s1"]}] deck.pdf, hal. 1\n' in result.context
