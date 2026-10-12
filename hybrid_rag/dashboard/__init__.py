"""Knowledge-graph viewer: a FastAPI router plus the built React app.

Needs the ``dashboard`` extra (``pip install "hybrid-rag[dashboard]"``). The app
sources live in the repository's ``dashboard/`` folder; ``bun run build`` there
writes ``static/`` next to this file.

    from hybrid_rag.dashboard import mount_dashboard

    mount_dashboard(app, prefix="/kg")   # viewer at /kg/, JSON API at /kg/api/*
"""

from hybrid_rag.dashboard.server import create_router, mount_dashboard

__all__ = ["create_router", "mount_dashboard"]
