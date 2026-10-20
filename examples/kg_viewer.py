from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_openai import OpenAIEmbeddings
from hybrid_rag import providers
from hybrid_rag.dashboard import mount_dashboard
from hybrid_rag.retrieval.workspace import workspace_pool

def get_embeddings(model_id, *, base_url, api_key, timeout, max_retries):
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
mount_dashboard(app, prefix="/kg")

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/kg/")