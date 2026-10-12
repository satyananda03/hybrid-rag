"""The viewer's JSON API and its static mount."""

from collections.abc import Sequence
from pathlib import Path

from fastapi import APIRouter, FastAPI, Query
from fastapi.params import Depends
from fastapi.staticfiles import StaticFiles

from hybrid_rag.core.types import KnowledgeGraph
from hybrid_rag.retrieval.workspace import workspace_pool
from hybrid_rag.storage.neo4j import list_workspaces

STATIC_DIR = Path(__file__).parent / "static"


def create_router(dependencies: Sequence[Depends] | None = None) -> APIRouter:
    """The graph endpoints. ``dependencies`` is where the host puts its auth."""
    router = APIRouter(prefix="/api", dependencies=list(dependencies or []))

    @router.get("/workspaces")
    async def workspaces() -> list[str]:
        return await list_workspaces()

    @router.get("/graph")
    async def graph(
        workspace: str,
        label: str = "*",
        max_depth: int = Query(3, ge=1, le=10),
        max_nodes: int = Query(1000, ge=1),
    ) -> KnowledgeGraph:
        async with workspace_pool.lease(workspace) as rag:
            return await rag.get_knowledge_graph(label, max_depth, max_nodes)

    @router.get("/labels/popular")
    async def popular_labels(workspace: str, limit: int = Query(50, ge=1, le=500)) -> list[str]:
        async with workspace_pool.lease(workspace) as rag:
            return await rag.chunk_entity_relation_graph.get_popular_labels(limit)

    @router.get("/labels/search")
    async def search_labels(workspace: str, q: str, limit: int = Query(50, ge=1, le=500)) -> list[str]:
        async with workspace_pool.lease(workspace) as rag:
            return await rag.chunk_entity_relation_graph.search_labels(q, limit)

    return router


def mount_dashboard(app: FastAPI, prefix: str = "/kg", dependencies: Sequence[Depends] | None = None) -> None:
    """Serve the viewer at ``{prefix}/`` and its API at ``{prefix}/api``.

    The API router goes first so the static mount, which matches everything under
    ``prefix``, never shadows it.
    """
    if not STATIC_DIR.is_dir():
        raise RuntimeError(f"{STATIC_DIR} is missing: run `bun run build` in dashboard/ first.")
    app.include_router(create_router(dependencies), prefix=prefix)
    app.mount(prefix, StaticFiles(directory=STATIC_DIR, html=True), name="hybrid-rag-dashboard")
