import asyncio
import json
from pathlib import Path
import pytest
from docx import Document
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XlImage
from PIL import Image
from hybrid_rag import progress
from hybrid_rag.config import RagSettings, settings
from hybrid_rag.core.chunk_schema import format_location, normalize_page
from hybrid_rag.ingestion import pipeline
from hybrid_rag.ingestion.chunking.parent_child import _page_range
from hybrid_rag.ingestion.knowledge_graph.chain import get_kg_chains
from hybrid_rag.ingestion.knowledge_graph.ontology_schema import (
    load_ontology_schema,
    prune_chunk,
    render_schema_for_prompt,
    resolve_ontology_path,
)
from hybrid_rag.ingestion.multimodal.content import page_offsets_of
from hybrid_rag.ingestion.parser.office.parser import parse_docx
from hybrid_rag.ingestion.parser.spreadsheet.parser import parse_xlsx
from hybrid_rag.storage.s3 import objects as s3_objects
from hybrid_rag.storage.s3 import require_bucket, upload_figures_to_s3

def images(items):
    return [i for i in items if i["type"] == "image"]

@pytest.fixture
def docx_path(tmp_path, make_png):
    doc = Document()
    doc.add_paragraph("Pengantar")
    doc.add_picture(str(make_png("small.png", (100, 100))))
    doc.add_picture(str(make_png("big.png", (900, 900))))
    path = tmp_path / "a.docx"
    doc.save(path)
    return path

@pytest.fixture
def xlsx_path(tmp_path, make_png):
    wb = Workbook()
    ws = wb.active
    ws.title = "Rekap"
    ws["A1"], ws["B1"] = "Nama", "Nilai"
    ws["A2"], ws["B2"] = "x", 1
    ws.add_image(XlImage(str(make_png("small.png", (100, 100)))), "D2")
    ws.add_image(XlImage(str(make_png("big.png", (900, 900)))), "D20")
    path = tmp_path / "a.xlsx"
    wb.save(path)
    return path

def test_docx_keeps_only_large_images(docx_path, tmp_path):
    items = parse_docx(docx_path, tmp_path / "out")
    [image] = images(items)
    assert Image.open(image["img_path"]).size == (900, 900)
    assert {"type": "text", "text": "Pengantar", "page_idx": 0} in items

def test_xlsx_keeps_only_large_images_and_uses_the_sheet_as_page(xlsx_path, tmp_path):
    [image] = images(parse_xlsx(xlsx_path, tmp_path / "out"))
    assert Image.open(image["img_path"]).size == (900, 900)
    assert image["page_idx"] == "Rekap"

def test_figure_thresholds_come_from_settings(docx_path, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "figure_min_width_px", 0)
    monkeypatch.setattr(settings, "figure_min_height_px", 0)
    assert len(images(parse_docx(docx_path, tmp_path / "out"))) == 2

def parse(path):
    async def go():
        async with pipeline.parse_file(path) as content_list:
            return content_list
    return asyncio.run(go())

@pytest.fixture
def hosted(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "host_figures", calls.append)
    return calls

def test_supported_extensions_follow_the_routing_table():
    assert pipeline.SUPPORTED_INGEST_EXTENSIONS == frozenset(pipeline.PARSERS)
    assert {"pdf", "docx", "xlsx", "md", "txt", "json"} <= pipeline.SUPPORTED_INGEST_EXTENSIONS
    assert not {"pptx", "doc", "odt", "rtf"} & pipeline.SUPPORTED_INGEST_EXTENSIONS

@pytest.mark.parametrize("name", ["a.docx", "a.xlsx"])
def test_office_files_are_parsed_then_hosted(name, docx_path, xlsx_path, hosted):
    content_list = parse({"a.docx": docx_path, "a.xlsx": xlsx_path}[name])
    assert hosted == [content_list] and len(images(content_list)) == 1

def test_pdf_goes_through_textract_then_hosting_and_cleans_up(tmp_path, monkeypatch, hosted):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    seen = {}
    async def fake_textract(path, output_file):
        seen["path"], seen["tmp_dir"] = path, Path(output_file).parent
        return {"raw": True}
    monkeypatch.setattr(pipeline.async_amazon_textract, "parse_document_to_json", fake_textract)
    monkeypatch.setattr(pipeline, "reconstruct_content_list", lambda raw, output_file: [{"type": "text", "text": "isi"}])
    async def go():
        async with pipeline.parse_file(pdf) as content_list:
            assert seen["tmp_dir"].exists()  # still there while the caller inserts
            return content_list
    assert asyncio.run(go()) == [{"type": "text", "text": "isi"}]
    assert seen["path"] == str(pdf) and len(hosted) == 1
    assert not seen["tmp_dir"].exists()

def test_text_and_content_list_files_are_not_hosted(tmp_path, hosted):
    md = tmp_path / "a.md"
    md.write_text("halo", encoding="utf-8")
    assert parse(md) == [{"type": "text", "text": "halo"}]
    prebuilt = tmp_path / "a.json"
    prebuilt.write_text(json.dumps([{"type": "image", "img_path": "https://s3/x.jpg"}]), encoding="utf-8")
    assert parse(prebuilt)[0]["img_path"] == "https://s3/x.jpg"
    assert hosted == []


def test_parse_file_rejects_bad_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse(tmp_path / "missing.pdf")

    unsupported = tmp_path / "a.pptx"
    unsupported.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"\.pptx tidak didukung"):
        parse(unsupported)

    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of object"):
        parse(bad)

class FakeS3Client:
    def __init__(self):
        self.puts = []

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs):
        if key.endswith("bad.png"):
            raise OSError("boom")
        self.puts.append((bucket, key, ExtraArgs["ContentType"]))

def test_figures_get_urls_and_a_failed_upload_keeps_its_local_path(tmp_path, make_png, monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(s3_objects, "get_client", lambda: client)
    monkeypatch.setattr(s3_objects, "figure_key", lambda path: f"figs/{Path(path).name}")
    monkeypatch.setattr(settings, "s3_bucket", "bkt")
    monkeypatch.setattr(settings, "s3_region", "ap-southeast-3")
    ok, bad = make_png("ok.png", (10, 10)), make_png("bad.png", (10, 10))

    items = [{"img_path": str(ok)}, {"img_path": str(bad)}]
    upload_figures_to_s3(items)

    assert items[0]["img_path"] == "https://bkt.s3.ap-southeast-3.amazonaws.com/figs/ok.png"
    assert items[1]["img_path"] == str(bad)
    assert client.puts == [("bkt", "figs/ok.png", "image/png")]

def test_require_bucket_names_the_variable(monkeypatch):
    monkeypatch.setattr(settings, "s3_bucket", None)
    with pytest.raises(ValueError, match="S3_BUCKET"):
        require_bucket()

def node(name, type_, **props):
    return {name: [{"entity_name": name, "entity_type": type_, "properties": props}]}

def edge(src, tgt, type_):
    return {(src, tgt): [{"src_id": src, "tgt_id": tgt, "relation_type": type_, "properties": {}}]}

def test_ontology_loads_and_renders_for_a_prompt_template():
    schema = load_ontology_schema("hukum")
    assert {n.label for n in schema.node_types} >= {"Peraturan", "Pasal", "Lembaga"}
    assert render_schema_for_prompt(schema).format() == str(schema.model_dump(exclude_none=True))

@pytest.mark.parametrize("name", ["../hukum", "nope"])
def test_unknown_or_unsafe_ontology_names_fail(name):
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_ontology_path(name)

def test_pruning_keeps_the_schema_and_repairs_direction():
    nodes = {
        **node("UU 1/2020", "Peraturan", tahun="2020"),
        **node("Pasal 1", "Pasal"),
        **node("Kucing", "Hewan"), 
    }
    edges = {
        **edge("Pasal 1", "UU 1/2020", "memilikiPasal"),  # reversed: repaired
        **edge("UU 1/2020", "Kucing", "mengaturTentang"),  # endpoint pruned
    }
    kept_nodes, kept_edges = asyncio.run(prune_chunk(nodes, edges, load_ontology_schema("hukum"), "c1"))
    assert set(kept_nodes) == {"UU 1/2020", "Pasal 1"}
    assert kept_nodes["UU 1/2020"][0]["properties"] == {"tahun": "2020"}
    assert list(kept_edges) == [("UU 1/2020", "Pasal 1")]

def test_chains_are_built_per_ontology(fake_llm):
    default, onto = get_kg_chains(), get_kg_chains("hukum")
    assert default.schema is None and get_kg_chains("") is default
    assert onto.schema is not None
    system = onto.extraction_prompt.messages[0].prompt.template
    assert "Peraturan" in system and "{ontology_schema}" not in system

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

class RecordingSink:
    def __init__(self):
        self.events = []
    def stage(self, name):
        self.events.append(("stage", name, progress.current_doc_id.get(), progress.current_workspace.get()))
    def skip(self, name, reason):
        self.events.append(("skip", name, reason))
    def activity(self, message, level="info"):
        self.events.append(("activity", message, level))

def test_progress_without_a_sink_is_a_noop():
    progress.set_sink(None)
    progress.stage("parse"), progress.skip("x", "y"), progress.activity("z")

def test_progress_sink_receives_events_with_the_library_set_context():
    sink = RecordingSink()
    progress.set_sink(sink)
    try:
        async def run():
            progress.current_doc_id.set("doc-1")
            progress.current_workspace.set("ws")
            progress.stage("chunk")
            progress.skip("multimodal", "tidak ada gambar")
            progress.activity("menunggu", level="warning")
        asyncio.run(run())
    finally:
        progress.set_sink(None)
    assert sink.events == [
        ("stage", "chunk", "doc-1", "ws"),
        ("skip", "multimodal", "tidak ada gambar"),
        ("activity", "menunggu", "warning"),
    ]

def test_settings_require_checks_only_the_roles_asked_for():
    s = RagSettings(_env_file=None, KEYWORD_LLM_MODEL="m", KEYWORD_LLM_BINDING_HOST="h")
    s.require()  # no role, nothing to check
    with pytest.raises(RuntimeError) as e:
        s.require("keyword", "contextualize")
    assert str(e.value) == (
        "Missing required env vars: KEYWORD_LLM_BINDING_API_KEY, CONTEXTUALIZE_LLM_MODEL, "
        "CONTEXTUALIZE_LLM_BINDING_HOST, CONTEXTUALIZE_LLM_BINDING_API_KEY"
    )


def test_every_model_role_is_configured_by_the_same_four_settings():
    from hybrid_rag.config import MODEL_ROLES

    for field, env in MODEL_ROLES.values():
        for suffix in ("model", "binding_host", "binding_api_key"):
            assert hasattr(settings, f"{field}_{suffix}"), f"{field}_{suffix}"
        aliases = {f.validation_alias for f in RagSettings.model_fields.values()}
        assert {f"{env}_MODEL", f"{env}_BINDING_HOST", f"{env}_BINDING_API_KEY"} <= aliases

# --- provider contract ---------------------------------------------------------------


@pytest.fixture
def recorded_hooks(monkeypatch):
    """Installs recording hooks for the three model factories; returns their call logs."""
    from hybrid_rag import providers

    calls = {"llm": [], "embeddings": [], "reranker": []}
    monkeypatch.setitem(providers._hooks, "get_llm", lambda *a, **kw: calls["llm"].append((a, kw)))
    monkeypatch.setitem(providers._hooks, "get_embeddings", lambda *a, **kw: calls["embeddings"].append((a, kw)))
    monkeypatch.setitem(providers._hooks, "get_reranker", lambda *a, **kw: calls["reranker"].append((a, kw)))
    monkeypatch.setattr(settings, "max_retries", 7)
    return calls


def test_every_model_factory_gets_the_same_connection_arguments(recorded_hooks):
    from hybrid_rag import providers

    providers.get_llm("m", base_url="h", api_key="k", timeout=9.0, max_tokens=None)
    providers.get_embeddings("m", base_url="h", api_key="k", timeout=9.0)
    providers.get_reranker("m", base_url="h", api_key="k", timeout=9.0)

    connection = {"base_url": "h", "api_key": "k", "timeout": 9.0, "max_retries": 7}
    assert recorded_hooks["llm"] == [(("m",), {**connection, "max_tokens": None})]
    assert recorded_hooks["embeddings"] == [(("m",), connection)]
    assert recorded_hooks["reranker"] == [(("m",), connection)]


def test_vlm_falls_back_to_the_llm_factory_uncapped(recorded_hooks, monkeypatch):
    from hybrid_rag import providers

    monkeypatch.setitem(providers._hooks, "get_vlm", None)
    providers.get_vlm("m", base_url="h", api_key="k", timeout=9.0)
    assert recorded_hooks["llm"] == [(("m",), {"base_url": "h", "api_key": "k", "timeout": 9.0, "max_retries": 7, "max_tokens": None})]


def test_temperature_is_forwarded_only_when_the_library_sets_it(recorded_hooks):
    from hybrid_rag import providers

    providers.get_llm("m", base_url=None, api_key=None)
    providers.get_llm("m", base_url=None, api_key=None, temperature=0.0)
    assert ["temperature" in kw for _, kw in recorded_hooks["llm"]] == [False, True]


def test_reranker_is_none_without_a_host_hook(monkeypatch):
    from hybrid_rag import providers

    monkeypatch.setitem(providers._hooks, "get_reranker", None)
    assert providers.get_reranker("m", base_url="h", api_key="k") is None


def test_embedding_func_builds_its_client_from_settings(recorded_hooks, monkeypatch):
    from hybrid_rag.core.embedding import build_embedding_func

    monkeypatch.setattr(settings, "embedding_model", "emb-model")
    monkeypatch.setattr(settings, "embedding_binding_host", "http://emb")
    monkeypatch.setattr(settings, "embedding_binding_api_key", "emb-key")
    monkeypatch.setattr(settings, "embedding_timeout", 11.0)
    build_embedding_func()
    [(args, kwargs)] = recorded_hooks["embeddings"]
    assert args == ("emb-model",)
    assert kwargs == {"base_url": "http://emb", "api_key": "emb-key", "timeout": 11.0, "max_retries": 7}


def test_env_example_lists_exactly_the_variables_config_reads():
    import re

    read = {f.validation_alias for f in RagSettings.model_fields.values() if isinstance(f.validation_alias, str)}
    listed = re.findall(r"^([A-Z][A-Z0-9_]*)=", Path(".env.example").read_text(), flags=re.M)
    assert sorted(listed) == sorted(read) and len(listed) == len(set(listed))


def test_env_example_changes_no_default():
    example = RagSettings(_env_file=".env.example")
    defaults = RagSettings(_env_file=None)
    # empty values mean "not set": every field must come out as the code default
    assert example.model_dump() == defaults.model_dump()
