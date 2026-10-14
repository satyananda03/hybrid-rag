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
from hybrid_rag.ingestion.token_limit import _normalized_child_offsets
from hybrid_rag.ingestion.utils import raise_first_failure
import logging

logger = logging.getLogger(__name__)


PARENT_CHUNK_SIZE = settings.parent_chunk_size
PARENT_CHUNK_OVERLAP = settings.parent_chunk_overlap
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
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
    *,
    parent_overlap: int | None = None,
    child_overlap: int | None = None,
    page_offsets: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split content into parent chunks, then sub-split each parent into children.

    Uses LangChain's RecursiveCharacterTextSplitter for both levels.

    ``page_offsets`` (``[{"page", "start"}]``, awal tiap halaman di ``content``) hanya
    menambah satu kunci pada tiap child — ``sidecar={"page_start","page_end"}`` — tanpa
    menyentuh ``content`` child sedikit pun; embedding dan retrieval tidak melihat
    perbedaan. Child yang posisinya tidak ketemu (semestinya tidak pernah: splitter
    hanya ``strip``) dibiarkan tanpa ``sidecar``.

    Returns:
        parent_chunks: list of ``{tokens, content, chunk_order_index[, sidecar]}``
        child_chunks:  list of ``{tokens, content, chunk_order_index, parent_index[, sidecar]}``

    Child menunjuk induknya lewat ``parent_index``; itulah satu-satunya tali antara
    kedua level. Jangan kembalikan peta posisi list — ``enforce_chunk_token_limit_
    before_embedding`` memecah child setelah tahap ini, jadi peta apa pun yang
    dibangun di sini sudah basi sebelum dipakai.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_size = parent_chunk_size or PARENT_CHUNK_SIZE
    child_size = child_chunk_size or CHILD_CHUNK_SIZE
    p_overlap = parent_overlap if parent_overlap is not None else PARENT_CHUNK_OVERLAP
    c_overlap = child_overlap if child_overlap is not None else CHILD_CHUNK_OVERLAP

    # Estimate chars per token (rough: 4 chars/token for mixed Indonesian/English)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=p_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split into parents
    parent_texts = parent_splitter.split_text(content)

    parent_chunks: list[dict[str, Any]] = []
    child_chunks: list[dict[str, Any]] = []

    child_global_index = 0
    page_starts = [o["start"] for o in page_offsets] if page_offsets else []
    # Kursor maju: parent (dengan overlap) dan child berurutan di dalam `content`,
    # jadi pencarian sekuensial O(n) dan potongan berulang terpetakan ke urutannya.
    parent_cursor = 0
    unlocated = 0

    for parent_idx, parent_text in enumerate(parent_texts):
        parent_text_stripped = parent_text.strip()
        if not parent_text_stripped:
            continue

        parent_tokens = len(tokenizer.encode(parent_text_stripped))
        parent: dict[str, Any] = {
            "tokens": parent_tokens,
            "content": parent_text_stripped,
            "chunk_order_index": parent_idx,
        }
        parent_chunks.append(parent)

        parent_span = _locate(content, parent_text_stripped, parent_cursor)
        if parent_span is not None:
            # Overlap: parent berikutnya mulai SEBELUM akhir yang ini, jadi kursor
            # hanya maju ke awal parent ini, bukan ke akhirnya.
            parent_cursor = parent_span[0] + 1
            if page_starts:
                # Rentang halaman parent melebar melewati tiap child-nya; inilah
                # yang dipakai laporan untuk merender halaman asli saat bukti
                # yang dikembalikan adalah teks parent.
                parent["sidecar"] = _page_range(page_offsets, page_starts, parent_span)
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
            local = _locate(parent_text_stripped, child_text_stripped, child_cursor) if parent_span else None
            if local is not None:
                child_cursor = local[0] + 1
                span = (parent_span[0] + local[0], parent_span[0] + local[1])
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
        f"(parent_size={parent_size}, child_size={child_size})"
    )

    return parent_chunks, child_chunks

CONTEXTUALIZE_SYSTEM_PROMPT = """
    ROLE : You are an expert AI assistant specializing in Document Analysis and Contextual Retrieval. Your primary task is to generate succinct context that situates a specific chunk of text within its parent document for search retrieval.
    LANGUAGE : ALWAYS RESPONSE in Bahasa Indonesia
    """

CONTEXTUALIZE_USER_TEMPLATE = (
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{child_content}\n</chunk>\n\n"
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