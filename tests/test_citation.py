import asyncio

from hybrid_rag.core.chunk_schema import format_location, normalize_page
from hybrid_rag.ingestion.chunking.parent_child import _page_range
from hybrid_rag.ingestion.multimodal.content import page_offsets_of
from hybrid_rag.retrieval import context
from hybrid_rag.retrieval.types import QueryParam
from hybrid_rag.retrieval.utils import generate_reference_list_from_chunks


def test_location_helpers():
    assert [normalize_page(v) for v in (0, 3, " Rekap ", "", None, True, 1.5)] == [0, 3, "Rekap", None, None, None, None]
    assert format_location(None) is None
    assert format_location(2) == format_location(2, 2) == "hal. 3"
    assert format_location(2, 3) == "hal. 3–4"
    assert format_location("Rekap") == 'sheet "Rekap"'
    assert format_location("Rekap", "Detail") == 'sheet "Rekap" – "Detail"'


def test_sheet_pages_flow_into_chunk_ranges():
    offsets = page_offsets_of(["isi rekap", "isi detail"], ["Rekap", "Detail"])
    starts = [o["start"] for o in offsets]
    assert _page_range(offsets, starts, (0, len("isi rekap\n\nisi"))) == {"page_start": "Rekap", "page_end": "Detail"}
    assert page_offsets_of(["teks .md"], [None]) == []


def test_references_per_file_and_location():
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
IMAGE_CHUNK = {"chunk_id": "c3", "content": "Image Content Analysis:\nImage Path: https://s3/x/fig.jpg?sig=1\nCaptions: None", "file_path": "f"}


def attach(db, **kw):
    return asyncio.run(context._attach_sources(CHUNKS, db, limit=None, **kw))


def test_attach_sources():
    child = attach(FakeKV(ROWS), return_parent=False, include_citations=True)
    assert [(c["content"], c["page_start"], c["page_end"]) for c in child] == [
        ("c1", 4, 4), ("c2", 5, 5), ("c3", "Rekap", "Rekap"),
    ]
    parent = attach(FakeKV(ROWS), return_parent=True, include_citations=True)
    assert [(c["chunk_id"], c["content"], c["page_start"], c["page_end"]) for c in parent] == [
        ("c1", "parent", 4, 5), ("c3", "c3", "Rekap", "Rekap"),
    ]
    db = FakeKV(ROWS)
    assert attach(db, return_parent=False, include_citations=False) == CHUNKS and db.calls == 0


def build_async(include_citations, chunks=CHUNKS):
    return context._build_context_str(
        [], [], chunks, "q", QueryParam(enable_rerank=False, include_citations=include_citations),
        {}, text_chunks_db=FakeKV(ROWS),
    )


def build(include_citations):
    return asyncio.run(build_async(include_citations))


def test_build_context_citations_toggle():
    text, data = build(True)
    assert '[1] f, hal. 5\n' in text and 'sheet "Rekap"' in text
    assert data["data"]["chunks"][0]["page_start"] == 4

    text, data = build(False)
    assert "Reference Document List" not in text and "reference_id" not in text
    assert data["data"]["references"] == []
    assert set(data["data"]["chunks"][0]) == {"content", "file_path", "chunk_id"}


class FakeRag:
    def __init__(self, include_citations):
        self.include_citations = include_citations

    async def aretrieve(self, query, param):
        text, data = await build_async(self.include_citations, CHUNKS[:2] + [IMAGE_CHUNK])
        return {**data, "context": text}


def test_images_follow_their_citation():
    from hybrid_rag.retrieval.pipeline import _retrieve

    result = asyncio.run(_retrieve(FakeRag(True), "q", QueryParam(), include_raw_data=True))
    assert result.images == [{"path": "https://s3/x/fig.jpg?sig=1", "reference_id": "3"}]
    assert '[3] f, sheet "Rekap"' in result.context  # the image's page, via its reference
    assert "image_path" not in result.raw_data["data"]["chunks"][0]

    result = asyncio.run(_retrieve(FakeRag(False), "q", QueryParam(), include_raw_data=False))
    assert result.images == [{"path": "https://s3/x/fig.jpg?sig=1"}]
