# hybrid-rag

RAG knowledge graph + vector di atas PostgreSQL (pgvector + ParadeDB BM25) dan Neo4j:
ingestion dokumen (Textract → chunk parent-child → caption multimodal → ekstraksi
graf), deletion, dan retrieval mode `mix` (graf + vector + BM25, RRF, rerank).

Package ini tidak meng-import aplikasi host. Host cukup:

1. mengisi `.env` (nama variabel di `hybrid_rag/config.py`, mis. `POSTGRES_*`,
   `NEO4J_*`, `EMBEDDING_*`, `LLM_*`, `TOP_K`, `CHUNK_TOP_K`, `TEXTRACT_*`);
2. memasang provider sekali saat startup;
3. memakai API publik `hybrid_rag.ingestion` / `hybrid_rag.retrieval`.

```python
from hybrid_rag import providers

providers.configure(
    get_llm=my_get_llm,                # (model_id, *, max_tokens, base_url, api_key, timeout, temperature) -> LangChain chat model
    get_embeddings=my_get_embeddings,  # (model_id, base_url, api_key) -> LangChain Embeddings
    rerank=my_rerank,                  # opsional: (query, documents, top_n=...) -> [{"index", "relevance_score"}]
    s3=my_s3,                          # ingestion: upload_figures_to_s3, upload_fileobj, delete_object, require_bucket
    progress=my_progress,              # opsional: stage, skip, activity, current_doc_id, current_workspace
)

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
- dengan ingestion: `hybrid-rag[ingestion]` (boto3/Textract, PyMuPDF, Pillow, unoserver)
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

```bash
uv sync --extra dashboard && uv run pytest
```

### Schema database

Runtime tidak menjalankan DDL. Schema PostgreSQL (schema, extension `vector`,
tabel, index) dan constraint Neo4j dibuat oleh:

```bash
make db-init        # = uv run python -m hybrid_rag.db_init
```