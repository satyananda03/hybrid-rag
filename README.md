# hybrid-rag

RAG knowledge graph + vector di atas PostgreSQL (pgvector + ParadeDB BM25) dan Neo4j:
ingestion dokumen (Textract → chunk parent-child → caption multimodal → ekstraksi
graf), deletion, dan retrieval mode `mix` (graf + vector + BM25, RRF, rerank).

Package ini tidak meng-import aplikasi host. Host cukup:

1. mengisi `.env` (nama variabel di `hybrid_rag/config.py`, mis. `POSTGRES_*`,
   `NEO4J_*`, `EMBEDDING_*`, `EXTRACT_LLM_*`, `CONTEXTUALIZE_LLM_*`, `KEYWORD_LLM_*`, `VLM_*`, `TOP_K`, `CHUNK_TOP_K`, `TEXTRACT_*`,
   `S3_*`: S3 dipakai langsung oleh parser Textract, bukan lewat provider);
2. memasang provider sekali saat startup;
3. memakai API publik `hybrid_rag.ingestion` / `hybrid_rag.retrieval`.

```python
from hybrid_rag import providers

# Tiga factory dengan argumen koneksi yang sama (diisi library dari .env), semuanya keyword
# setelah model_id: base_url, api_key, timeout (detik, None = tanpa batas), max_retries (MAX_RETRIES).
# Retry model dikerjakan klien yang dibangun factory; library tidak menambah retry sendiri.
providers.configure(
    get_llm=my_get_llm,                # (model_id, *, base_url, api_key, timeout, max_retries, max_tokens[, temperature]) -> LangChain chat model
    get_embeddings=my_get_embeddings,  # (model_id, *, base_url, api_key, timeout, max_retries) -> LangChain Embeddings
    get_reranker=my_get_reranker,      # opsional: (model_id, *, base_url, api_key, timeout, max_retries) -> async (query, documents, top_n) -> [{"index", "relevance_score"}]
)

# opsional: progress ingest/delete ke UI. Sink = objek dengan stage(name), skip(name, reason),
# activity(message, level); doc/workspace yang sedang berjalan: progress.current_doc_id.get()
from hybrid_rag import progress
progress.set_sink(my_sink)

from hybrid_rag.ingestion import parse_file, workspace_pool as write_pool, delete_document
from hybrid_rag.retrieval import retrieve_context

async with parse_file("dokumen.pdf") as content_list:
    async with write_pool.lease("workspace_a") as rag:
        await rag.insert_content_list(content_list, file_path="dokumen.pdf", doc_id="doc-1")

result = await retrieve_context("pertanyaan", workspace="workspace_a")
print(result.context)
```

Opsi per-query semuanya lewat `QueryParam`:

```python
from hybrid_rag.retrieval import QueryParam

result = await retrieve_context(
    "pertanyaan",
    workspace="workspace_a",
    param=QueryParam(return_parent=True, include_citations=False),
)
```

### Sitasi

Dengan `include_citations=True` (default), context diakhiri daftar referensi, satu
entri per file + lokasi. Lokasi diambil dari `page_idx` di content list: int berarti
halaman (0-based), str berarti nama sheet Excel, tidak ada berarti tanpa lokasi.

```
Reference Document List (cite a chunk with its [reference_id]; ...):
[1] laporan.pdf, hal. 3–4
[2] data.xlsx, sheet "Rekap 2024"
[3] catatan.md
```

Data yang sama ada di `raw_data["data"]["references"]` (`reference_id`, `file_path`,
`location`). Biarkan LLM hanya menulis `[n]`, lalu host yang merender sitasinya.
Contoh instruksi untuk system prompt:

> Kutip setiap klaim dengan [reference_id] dari Reference Document List. Jangan menulis
> nomor halaman atau nama file sendiri.

Gambar ikut sitasinya. Chunk gambar memuat `Image Path: <url S3>` di isinya, jadi LLM
tahu gambar itu milik `[n]` yang mana. Di sisi host, `result.images` berisi
`[{"path", "reference_id"}]`, dan setiap chunk di `raw_data` yang punya gambar membawa
`image_path`.

Instalasi:

- retrieval saja: `hybrid-rag`
- dengan ingestion: `hybrid-rag[ingestion]` (boto3/Textract, PyMuPDF, Pillow, python-docx/openpyxl lewat unstructured)
- dengan dashboard knowledge graph: `hybrid-rag[dashboard]` (FastAPI)

Dari folder lokal di project lain: `uv add ./hybrid-rag` (atau `uv add --editable ./hybrid-rag`).

## Dashboard knowledge graph

```python
from fastapi import FastAPI
from hybrid_rag.dashboard import mount_dashboard

app = FastAPI()
mount_dashboard(app, prefix="/kg")   # viewer di /kg/, API di /kg/api/*
```

Auth milik host: `mount_dashboard(app, dependencies=[Depends(my_auth)])`. Endpoint:
`GET /api/workspaces`, `/api/graph`, `/api/labels/popular`, `/api/labels/search`
(semua butuh `workspace`).

Sumber React ada di `dashboard/`. Wheel hanya berisi hasil build-nya, jadi build dulu
sebelum `uv build` atau sebelum memasang dari git:

```bash
cd dashboard && bun install && bun run build   # menulis src/hybrid_rag/dashboard/static/
bun run dev                                    # dev server :5174, proxy /api -> VITE_BACKEND_URL
```

## Development

### Test

```bash
make test               # unit test: tanpa jaringan, database, atau model
make test-integration   # test yang membaca PostgreSQL dari .env (read-only); jalankan make db-init dulu
```

`tests/test_ingestion.py` (parser, routing `parse_file`, S3, ontology, progress) dan
`tests/test_retrieval.py` (sitasi, context, keyword, pencarian hybrid BM25).
Fixture bersama ada di `tests/conftest.py`. Satu file/test saja:
`uv run --extra ingestion pytest tests/test_retrieval.py -k citations`.

### Schema database

Runtime tidak menjalankan DDL. Schema PostgreSQL (schema, extension `vector`,
tabel, index) dan constraint Neo4j dibuat oleh:

```bash
make db-init        # = uv run python -m hybrid_rag.db_init
```