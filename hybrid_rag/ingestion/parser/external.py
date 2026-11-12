"""The ``external`` parser backend: the file is parsed by the parser server, not in this process.

Upload the file (``POST /parse``) → the server queues a job → poll its status
(``GET /parse/{job_id}``) until it ends → return its content list. That list is finished: text
items are stamped with ``layout`` and every ``img_path`` is already an S3 URL, the same as the
builtin backend (``router.py``) yields after ``host_figures``.

Debug: ``python -m hybrid_rag.ingestion.parser.external file.pdf``
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from hybrid_rag import progress
from hybrid_rag.config import settings
from hybrid_rag.ingestion.parser.parse import SUPPORTED_INGEST_EXTENSIONS

logger = logging.getLogger(__name__)

POLL_FIRST_S = 1.0
POLL_MAX_S = 10.0
#: One request to the server (upload or status); parsing itself is bounded by ``PARSER_TIMEOUT``.
REQUEST_TIMEOUT_S = 120.0

#: Test seam: an ``httpx`` transport to use instead of the network.
_transport: httpx.AsyncBaseTransport | None = None

# A dropped connection is retried; an answer from the server, even an error, is not.
_retry_on_network_error = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.TransportError),
    reraise=True,
)


class ParserServerError(RuntimeError):
    """The parser server rejected the file, failed to parse it, or lost the job."""


@asynccontextmanager
async def parse_file(file_path: str | Path, entity_registry: str | None = None) -> AsyncIterator[list[dict]]:
    """Parse one file on the parser server into a content list, ready for ``rag.insert_content_list``.

    ``entity_registry`` is accepted for the shared signature and checked like the builtin backend
    does, but not sent: the server transcribes slides without a domain glossary.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lstrip(".").lower()
    if ext not in SUPPORTED_INGEST_EXTENSIONS:
        raise ValueError(
            f"Tipe file .{ext} tidak didukung. "
            f"Didukung: {', '.join('.' + e for e in sorted(SUPPORTED_INGEST_EXTENSIONS))}"
        )
    if entity_registry:
        from hybrid_rag.ingestion.knowledge_graph.entity_resolution import load_entity_registry

        load_entity_registry(entity_registry)  # a typo fails here, as with the builtin backend
    settings.require_parser_binding()

    logger.info(f"[parse] starting file={path.name} (parser server)")
    progress.stage("parse")
    async with _client() as client:
        job = await _submit(client, path)
        result = await _wait_for_result(client, job, path.name)
    content_list = result["content_list"]
    logger.info(
        f"[parse] prepared {len(content_list)} items from file={path.name} "
        f"(layout={result['layout']}, server {result['elapsed_seconds']}s)"
    )
    yield content_list


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.parser_binding_host,
        headers={"X-API-Key": settings.parser_binding_api_key},
        timeout=REQUEST_TIMEOUT_S,
        transport=_transport,
    )


@_retry_on_network_error
async def _submit(client: httpx.AsyncClient, path: Path) -> dict:
    """Upload ``path``; the job as the server first reports it (a light file is already done).

    ponytail: a retry after the server received the upload but the answer was lost starts a second
    job; its result is simply never fetched. Add an idempotency key if that waste shows up.
    """
    with open(path, "rb") as file:
        response = await client.post("/parse", files={"file": (path.name, file)})
    return _job_of(response, path.name)


@_retry_on_network_error
async def _status(client: httpx.AsyncClient, job_id: str, name: str) -> dict:
    return _job_of(await client.get(f"/parse/{job_id}"), name)


def _job_of(response: httpx.Response, name: str) -> dict:
    """The job in ``response``, or the server's own explanation as a ``ParserServerError``."""
    if response.is_success:
        return response.json()
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text[:300]
    raise ParserServerError(f"{name}: parser server menjawab HTTP {response.status_code}: {detail}")


async def _wait_for_result(client: httpx.AsyncClient, job: dict, name: str) -> dict:
    """Poll until the job ends; its result, or ``ParserServerError`` when it failed.

    ``PARSER_TIMEOUT`` counts from when the job starts running: time spent queued behind other
    documents is the server being busy, not this document being stuck.
    """
    delay, deadline, announced = POLL_FIRST_S, None, None
    while job["status"] not in ("succeeded", "failed"):
        if job["status"] != announced:
            announced = job["status"]
            progress.activity("Parser server: menunggu antrean" if announced == "queued" else "Parser server: memproses dokumen")
        if job["status"] == "running":
            deadline = deadline or time.monotonic() + settings.parser_timeout
            if time.monotonic() > deadline:
                raise TimeoutError(f"{name}: parser server belum selesai setelah {settings.parser_timeout:.0f} detik")
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, POLL_MAX_S)
        job = await _status(client, job["job_id"], name)
    if job["status"] == "failed":
        raise ParserServerError(f"{name}: parser server gagal memparse: {job['error']}")
    return job["result"]


if __name__ == "__main__":
    import sys
    from collections import Counter

    async def _main(file_path: str) -> None:
        async with parse_file(file_path) as content_list:
            print(f"{len(content_list)} item:", dict(Counter(item.get("type", "text") for item in content_list)))

    asyncio.run(_main(sys.argv[1]))
