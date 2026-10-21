import asyncio
import bisect
from collections.abc import Callable
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from functools import cache

from hybrid_rag import providers
from hybrid_rag.config import NOVA_MODELS, settings
from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.core.tokenizer import Tokenizer
from hybrid_rag.ingestion.chunking.structure import chunker_for
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import acronym_glossary
from hybrid_rag.ingestion.token_limit import _normalized_child_offsets
from hybrid_rag.ingestion.utils import raise_first_failure
import logging

logger = logging.getLogger(__name__)


CHILD_CHUNK_SIZE = settings.child_chunk_size
CHILD_CHUNK_OVERLAP = settings.child_chunk_overlap
CONTEXTUALIZE_MAX_ASYNC = settings.contextualize_max_async

def _locate(haystack: str, needle: str, search_from: int) -> tuple[int, int] | None:
    """Span ``needle`` di ``haystack`` sejak ``search_from``; verbatim dulu, lalu abaikan spasi."""
    pos = haystack.find(needle, search_from)
    if pos >= 0:
        return pos, pos + len(needle)
    return _normalized_child_offsets(haystack, needle, search_from)


def _page_range(page_offsets: list[dict[str, Any]], starts: list[int], span: tuple[int, int]) -> dict[str, Any]:
    """Halaman awal/akhir untuk satu span karakter; ``starts`` = ``[o["start"]]`` terurut.

    Plus ``images`` — gambar slide di rentang itu — bila offset-nya membawa ``image``.
    """
    first = max(0, bisect.bisect_right(starts, span[0]) - 1)
    last = max(0, bisect.bisect_right(starts, max(span[0], span[1] - 1)) - 1)
    pages = {"page_start": page_offsets[first]["page"], "page_end": page_offsets[last]["page"]}
    images = list(dict.fromkeys(o["image"] for o in page_offsets[first:last + 1] if o.get("image")))
    return {**pages, "images": images} if images else pages


def chunking_parent_child(
    tokenizer: Tokenizer,
    content: str,
    *,
    chunk_options: dict[str, Any] | None = None,
    parent_max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split content into parent chunks along its structure, then sub-split each parent into children.

    Parents come from the document's chunker (``structure.chunker_for``): one section
    per parent, cut between paragraphs only when the section is over
    ``parent_max_tokens``. Children are LangChain's RecursiveCharacterTextSplitter
    over each parent's text.

    ``chunk_options`` is what ``separate_content`` stored for ``content``:
    ``page_offsets`` (``[{"page", "start"}]``) only add ``sidecar={"page_start",
    "page_end"}`` to each chunk without touching its ``content``, so embedding and
    retrieval see no difference; ``heading_offsets`` and ``layout`` pick the parent
    boundaries. A child whose position is not found (it should never be: the splitter
    only strips) is left without a ``sidecar``.

    Returns:
        parent_chunks: list of ``{tokens, content, chunk_order_index[, heading][, sidecar]}``
        child_chunks:  list of ``{tokens, content, chunk_order_index, parent_index[, sidecar]}``

    ``heading`` (``{"level", "heading", "parent_headings"}``) is the parent's section
    path, read by entity extraction; children carry none, so what retrieval shows is
    unchanged. A child points at its parent through ``parent_index``; that is the
    only link between the two levels. Do not return a list-position map —
    ``enforce_chunk_token_limit_before_embedding`` splits children after this step,
    so any map built here is stale before it is used.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunker = chunker_for(tokenizer, chunk_options, parent_max_tokens)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_chunks: list[dict[str, Any]] = []
    child_chunks: list[dict[str, Any]] = []

    child_global_index = 0
    page_offsets = (chunk_options or {}).get("page_offsets") or []
    page_starts = [o["start"] for o in page_offsets]
    unlocated = 0

    for parent_idx, parent_span in enumerate(chunker.split(content)):
        parent_text_stripped = content[parent_span.start:parent_span.end]
        parent: dict[str, Any] = {
            "tokens": len(tokenizer.encode(parent_text_stripped)),
            "content": parent_text_stripped,
            "chunk_order_index": parent_idx,
        }
        if parent_span.headings:
            *outer, own = parent_span.headings
            parent["heading"] = {
                "level": own.level,
                "heading": own.text,
                "parent_headings": [h.text for h in outer],
            }
        if page_starts:
            # Rentang halaman parent melebar melewati tiap child-nya; inilah
            # yang dipakai laporan untuk merender halaman asli saat bukti
            # yang dikembalikan adalah teks parent.
            parent["sidecar"] = _page_range(page_offsets, page_starts, (parent_span.start, parent_span.end))
        parent_chunks.append(parent)
        child_cursor = 0

        # Split parent into children
        child_texts = child_splitter.split_text(parent_text_stripped)

        for child_text in child_texts:
            child_text_stripped = child_text.strip()
            if not child_text_stripped:
                continue

            child_tokens = len(tokenizer.encode(child_text_stripped))
            child: dict[str, Any] = {
                "tokens": child_tokens,
                "content": child_text_stripped,
                "chunk_order_index": child_global_index,
                "parent_index": parent_idx,
            }
            local = _locate(parent_text_stripped, child_text_stripped, child_cursor)
            if local is not None:
                child_cursor = local[0] + 1
                span = (parent_span.start + local[0], parent_span.start + local[1])
                if page_starts:
                    child["sidecar"] = _page_range(page_offsets, page_starts, span)
            else:
                unlocated += 1
            child_chunks.append(child)
            child_global_index += 1

    if unlocated:
        logger.warning(f"[parent-child] {unlocated} child tanpa posisi sumber; halaman tidak tercatat untuk mereka")

    logger.info(
        f"[parent-child] Split into {len(parent_chunks)} parents, "
        f"{len(child_chunks)} children "
        f"({type(chunker).__name__}, parent_max_tokens={chunker.max_tokens}, child_size={CHILD_CHUNK_SIZE})"
    )

    return parent_chunks, child_chunks

CONTEXTUALIZE_SYSTEM_PROMPT = """
    ROLE : You are an expert AI assistant specializing in Document Analysis and Contextual Retrieval. Your primary task is to generate succinct context that situates a specific chunk of text within its parent document for search retrieval.
    LANGUAGE : ALWAYS RESPONSE in Bahasa Indonesia
    """

CONTEXTUALIZE_USER_TEMPLATE = (
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{child_content}\n</chunk>\n\n"
    "{glossary}\n\n"
    "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
)

@cache
def contextualize_chain() -> Runnable:
    """Prompt piped into the contextualize model, built on first use.

    The user message keeps two separate text parts on purpose — the parent
    document comes first and is identical for every child of that parent, which
    is what the proxy's prompt cache keys on.
    """
    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
                (
                    "human",
                    [
                        {
                            "type": "text",
                            "text": "<document>\n{parent_content}\n</document>\n\n",
                            # Nova tidak cache otomatis: LiteLLM menjadikan ini Bedrock cachePoint
                            # setelah blok dokumen (prefix system + parent). Model lain tidak dikirimi.
                            **({"cache_control": {"type": "ephemeral"}} if settings.contextualize_llm_model in NOVA_MODELS else {}),
                        },
                        {"type": "text", "text": CONTEXTUALIZE_USER_TEMPLATE},
                    ],
                ),
            ]
        )
        | providers.get_llm(
            model_id=settings.contextualize_llm_model,
            temperature=0.0,
            # The answer is one short paragraph, but a hard cap would truncate it
            # mid-sentence and poison the chunk it is prepended to. On Nova the cap
            # also counts reasoning tokens, so a small one yields an empty context.
            max_tokens=None,
            base_url=settings.contextualize_llm_binding_host,
            api_key=settings.contextualize_llm_binding_api_key,
            timeout=settings.contextualize_llm_timeout,
        )
        | StrOutputParser()
    )

class ContextualizationError(RuntimeError):
    """A child chunk could not be given its parent's context."""

    def __init__(self, chunk_order_index: int, cause: object):
        super().__init__(
            f"Contextualization failed for child chunk {chunk_order_index}: {cause}"
        )


async def contextualize_children(
    parent_chunks: list[dict[str, Any]],
    child_chunks: list[dict[str, Any]],
    *,
    max_concurrent: int | None = None,
    on_contextualized: Callable[[dict[str, Any]], None] | None = None,
    entity_registry: str | None = None,
) -> list[dict[str, Any]]:
    """Contextualize each child chunk using its parent as context via LLM.

    ``on_contextualized`` is called with each child as soon as its context is
    written, so the caller can start work on it (embedding) without waiting
    for the slowest child.

    Runs on :data:`contextualize_chain` against ``CONTEXTUALIZE_LLM_*`` — a
    model chosen for prompt caching, which is why this step has its own chain
    and its own model, endpoint, key and timeout rather than sharing the
    extraction ones.

    Always runs — there is no off switch. A child chunk without its parent's
    context is what retrieval was failing on, so skipping this step silently
    degrades every downstream answer. For the same reason a chunk whose context
    still cannot be produced after the client's retries fails the whole call
    (and with it the document) instead of falling back to its bare content.

    ``entity_registry`` adds its acronyms that the parent mentions (see
    ``acronym_glossary``); placed after the document block, so the cached
    prefix stays the same for every child of a parent.

    Raises:
        ContextualizationError: a child could not be contextualized; carries
            the child's index. The remaining calls are cancelled.
    """
    concurrency = max_concurrent or CONTEXTUALIZE_MAX_ASYNC
    semaphore = asyncio.Semaphore(concurrency)

    async def _contextualize_one(child: dict[str, Any]) -> dict[str, Any]:
        parent_idx = child.get("parent_index", 0)
        parent_content = (
            parent_chunks[parent_idx]["content"]
            if parent_idx < len(parent_chunks)
            else ""
        )

        async with semaphore:
            try:
                context = await contextualize_chain().ainvoke(
                    {
                        "parent_content": parent_content,
                        "child_content": child["content"],
                        "glossary": acronym_glossary(entity_registry, parent_content),
                    }
                )
            except Exception as e:
                raise ContextualizationError(child["chunk_order_index"], e) from e
        context = (context or "").strip().rstrip(".")
        if not context:
            raise ContextualizationError(child["chunk_order_index"], "empty context")

        new_child = dict(child)
        new_child["_original_content"] = child["content"]
        new_child["content"] = f"{context}. {child['content']}"
        if on_contextualized is not None:
            on_contextualized(new_child)
        return new_child

    logger.info(
        f"[parent-child] Contextualizing {len(child_chunks)} children "
        f"(concurrency={concurrency})..."
    )

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(_contextualize_one(child)) for child in child_chunks]
    except* ContextualizationError as failures:
        raise_first_failure(failures)

    logger.info("[parent-child] Contextualization complete.")
    return [task.result() for task in tasks]

def redirect_source_ids(
    chunk_results: list,
    parent_to_child_map: dict[str, list[str]],
) -> None:
    """In-place replace source_id in extraction results from parent → child keys.

    After entity/relation extraction runs on parent chunks, each entity's
    source_id points to a parent chunk_key. This function redirects those
    to the child chunk_keys so that graph retrieval returns child chunks.

    Args:
        chunk_results: List of (maybe_nodes, maybe_edges) from _process_extract_entities.
        parent_to_child_map: ``{parent_chunk_key: [child_chunk_keys]}``, dibangun
            di worker dari ``parent_index`` tiap child (lihat pipeline.py).
    """
    redirected_count = 0

    for maybe_nodes, maybe_edges in chunk_results:
        for _entity_name, entities in maybe_nodes.items():
            for entity in entities:
                source_id = entity.get("source_id", "")
                if source_id in parent_to_child_map:
                    entity["source_id"] = GRAPH_FIELD_SEP.join(
                        parent_to_child_map[source_id]
                    )
                    redirected_count += 1

        for _edge_key, edges in maybe_edges.items():
            for edge in edges:
                source_id = edge.get("source_id", "")
                if source_id in parent_to_child_map:
                    edge["source_id"] = GRAPH_FIELD_SEP.join(
                        parent_to_child_map[source_id]
                    )
                    redirected_count += 1

    logger.info(f"[parent-child] Redirected {redirected_count} source_ids")