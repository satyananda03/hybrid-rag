"""Penampil knowledge graph: halaman React + JSON API, dipasang (mount) ke aplikasi FastAPI.

Jalankan dari root repo, supaya `.env` terbaca:

    uv run --extra ingestion --extra dashboard --with uvicorn \
        uvicorn kg_viewer:app --app-dir examples

Lalu buka http://127.0.0.1:8000/kg/ dan pilih workspace-nya di halaman itu.

API-nya membuka graph SEMUA workspace tanpa autentikasi. Uvicorn mendengarkan 127.0.0.1 secara
bawaan; kalau dibuka ke jaringan, pasang autentikasi lewat `mount_dashboard(app, dependencies=[...])`.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_openai import OpenAIEmbeddings

from hybrid_rag import providers
from hybrid_rag.dashboard import mount_dashboard
from hybrid_rag.retrieval.workspace import workspace_pool


def get_embeddings(model_id, *, base_url, api_key, timeout, max_retries):
    # Pool retrieval membangun fungsi embedding saat dibuat, walau penampil hanya membaca graph dan
    # tidak pernah memanggilnya. Untuk ingest dan retrieval pakai `RetryingEmbeddings` dari
    # rag.ipynb, yang memasangkan vektor berdasarkan `index`.
    return OpenAIEmbeddings(
        model=model_id, base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries,
        check_embedding_ctx_length=False,
    )


providers.configure(get_embeddings=get_embeddings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await workspace_pool.shutdown()


app = FastAPI(title="Knowledge Graph Viewer", lifespan=lifespan)
mount_dashboard(app, prefix="/kg")  # penampil di /kg/, API di /kg/api/*


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/kg/")
