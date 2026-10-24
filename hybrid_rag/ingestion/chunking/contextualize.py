"""Contextualize each child chunk with its parent through an LLM (Contextual Retrieval)."""
import asyncio
import logging
from collections.abc import Callable
from functools import cache
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from hybrid_rag import providers
from hybrid_rag.config import NOVA_MODELS, settings
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import acronym_glossary
from hybrid_rag.ingestion.utils import raise_first_failure

logger = logging.getLogger(__name__)

CONTEXTUALIZE_MAX_ASYNC = settings.contextualize_max_async

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
