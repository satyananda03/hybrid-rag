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

Instalasi:

- retrieval saja: `hybrid-rag`
- dengan ingestion: `hybrid-rag[ingestion]` (boto3/Textract, PyMuPDF, Pillow, unoserver)

DDL tabel dan index dibuat saat instance dibangun dengan
`hybrid_rag.config.settings.rag_schema_bootstrap = True` (idempoten).

Isolasi data per `workspace`: kolom `workspace` di setiap tabel PostgreSQL dan
properti `workspace` pada node `HybridRagEntity` di Neo4j (constraint unik
`(workspace, entity_id)`). Jumlah workspace tidak menambah schema. Ingestion dan retrieval memakai pool PostgreSQL
terpisah.

Contoh host: `app/rag.py` di repo majaAI-be.
