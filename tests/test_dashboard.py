from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hybrid_rag.core.types import KnowledgeGraph
from hybrid_rag.dashboard import server


class FakeGraph:
    async def get_popular_labels(self, limit):
        return ["a", "b"][:limit]

    async def search_labels(self, query, limit):
        return [query]


class FakeRag:
    chunk_entity_relation_graph = FakeGraph()

    async def get_knowledge_graph(self, label, max_depth, max_nodes):
        return KnowledgeGraph(is_truncated=(label, max_depth, max_nodes) == ("x", 2, 5))


class FakePool:
    @asynccontextmanager
    async def lease(self, workspace):
        yield FakeRag()


async def fake_list_workspaces():
    return ["ws1"]


def test_router(monkeypatch):
    monkeypatch.setattr(server, "workspace_pool", FakePool())
    monkeypatch.setattr(server, "list_workspaces", fake_list_workspaces)
    app = FastAPI()
    app.include_router(server.create_router())
    client = TestClient(app)

    assert client.get("/api/workspaces").json() == ["ws1"]
    assert client.get("/api/graph", params={"workspace": "ws1", "label": "x", "max_depth": 2, "max_nodes": 5}).json() == {
        "nodes": [], "edges": [], "is_truncated": True,
    }
    assert client.get("/api/labels/popular", params={"workspace": "ws1", "limit": 1}).json() == ["a"]
    assert client.get("/api/labels/search", params={"workspace": "ws1", "q": "foo"}).json() == ["foo"]
    assert client.get("/api/graph").status_code == 422  # workspace is required
