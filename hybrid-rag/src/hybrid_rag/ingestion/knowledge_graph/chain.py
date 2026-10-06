"""LangChain chains for the text side of the graph.

Extraction, gleaning, and the entity/relation description summary.

A prompt template from YAML piped into a model bound to a Pydantic output
schema, plus a cache lookup inside the chain
(see ``core/llm_cache.py``). ``ainvoke`` returns a validated model, so callers
read fields off it instead of parsing a response body.

Extraction and gleaning are built per *domain* by ``get_kg_chains``: a document
selects which contract (entity types, relation vocabulary, domain rules) applies
to it, and one workspace may hold several. ``lru_cache`` keeps the "built once"
property per domain. The chain *structure* is the same for every domain — a YAML
file can only supply prompt text, never the number or order of LLM calls.

Each chain talks to its own role binding (``EXTRACT_*``), falling back to the
base LLM settings. Models come from ``hybrid_rag.providers`` and are built on
first use. The VLM chains live in ``multimodal/vlm.py``.
"""


import asyncio
from functools import cache, lru_cache
from pathlib import Path
from typing import NamedTuple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from hybrid_rag import providers
from hybrid_rag.config import NOVA_MODELS, settings
from hybrid_rag.ingestion.knowledge_graph.schema import EntityExtractionOutput
from hybrid_rag.core.llm_cache import cached_chain
# load_prompt_file is the cached YAML parse; used here to test key presence
# without a second read and without swallowing parse errors.
from hybrid_rag.core.prompts import load_prompt, load_prompt_file

KG_PROMPTS_PATH = Path(__file__).resolve().parent / "knowledge_graph.yaml"
DOMAIN_PROMPTS_DIR = KG_PROMPTS_PATH.parent / "domains"

#: Values that mean "the default contract" rather than naming a file of their own.
#: Note the default contract is *not* domain-neutral — it is the Indonesian
#: legal/regulatory one (see the entity_types block in knowledge_graph.yaml).
#:
#: ``"default"`` lets a host that lists a domain for every document spell the
#: default contract explicitly. Consequence: a file named
#: ``domains/default.yaml`` would never be read — the name resolves here first.
_DEFAULT_DOMAINS = (None, "", "default")

@cache
def _extract_llm() -> Runnable:
    """The extraction model bound to the output schema, built on first use."""
    return providers.get_llm(
        model_id=settings.resolved_extract_model,
        max_tokens=None,  # a capped structured output is truncated JSON: 40% of extractions exceed 900 tokens
        temperature=0.0,
        base_url=settings.resolved_extract_host,
        api_key=settings.resolved_extract_api_key,
        timeout=settings.extract_llm_timeout,
    ).with_structured_output(EntityExtractionOutput)

class KGChains(NamedTuple):
    """One domain's extraction contract, built and cached as a unit."""

    extraction_prompt: ChatPromptTemplate
    extraction_chain: Runnable
    gleaning_prompt: ChatPromptTemplate
    gleaning_chain: Runnable
    section_context: str


def _domain_path(domain: str | None) -> Path:
    """Locate a domain's YAML. Unknown names fail loudly rather than defaulting.

    A typo'd domain that quietly ingested under the wrong contract could only be
    undone by reingesting the document, so a missing file is an error here.
    """
    if domain in _DEFAULT_DOMAINS:
        return KG_PROMPTS_PATH
    path = DOMAIN_PROMPTS_DIR / f"{domain}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in DOMAIN_PROMPTS_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"unknown extraction domain {domain!r}: {path} not found. "
            f"available: {available or '(none)'} plus {list(_DEFAULT_DOMAINS)}"
        )
    return path


def _load_domain_prompt(key: str, path: Path) -> dict[str, str]:
    """Read one prompt key, falling back to the default contract when absent.

    Only ``entity_extraction`` is mandatory in a domain file. Everything else —
    gleaning mechanics, section context, summaries — is machinery a domain has no
    reason to restate, so it comes from knowledge_graph.yaml unless overridden.

    Membership is tested against the parsed mapping rather than by catching
    ``load_prompt``'s ValueError, which also fires on a malformed file: a broken
    domain YAML must fail loudly, not quietly extract under the default contract.
    """
    if path != KG_PROMPTS_PATH and key in load_prompt_file(path):
        return load_prompt(key, path)
    return load_prompt(key, KG_PROMPTS_PATH)


def get_kg_chains(domain: str | None = None) -> KGChains:
    """Return the extraction + gleaning chains for one domain, built once each.

    Caching is keyed on the resolved path, not on ``domain``: ``None``, ``""`` and
    an omitted argument all mean the default contract and must share one entry
    rather than build a copy each.
    """
    return _build_chains(_domain_path(domain))


@lru_cache(maxsize=8)
def _build_chains(path: Path) -> KGChains:
    """Build one contract's chains.

    ``load_prompt`` caches the *parsed YAML*, not the built objects, so without this
    cache every chunk would rebuild two ChatPromptTemplates and two cached_chain
    closures. That is what makes this the registry — there is no registry class.
    """
    extraction = _load_domain_prompt("entity_extraction", path)
    # Schema tool + system prompt identik untuk setiap chunk satu domain (~1.9K token, di atas minimum
    # cache Bedrock). Nova tidak cache otomatis, jadi cachePoint dipasang setelah system; LiteLLM
    # menerjemahkan cache_control ke cachePoint. Teks prompt tidak berubah.
    system = (
        [{"type": "text", "text": extraction["system_prompt"], "cache_control": {"type": "ephemeral"}}]
        if settings.resolved_extract_model in NOVA_MODELS
        else extraction["system_prompt"]
    )

    extraction_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", extraction["user_prompt"])]
    )

    # The gleaning pass runs under the extraction rules, so it reuses that system
    # prompt rather than keeping a second copy in the YAML. The placeholder replays
    # the first pass (its user prompt and the entities it returned) so the model can
    # tell what it has already covered. Because the system prompt comes from the
    # domain, gleaning follows the domain automatically — a domain file only needs
    # its own entity_continue_extraction when the gleaning *mechanics* differ.
    gleaning_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("messages"),
            ("human", _load_domain_prompt("entity_continue_extraction", path)["user_prompt"]),
        ]
    )

    return KGChains(
        extraction_prompt=extraction_prompt,
        extraction_chain=cached_chain(
            extraction_prompt,
            _extract_llm(),
            cache_type="extract",
            output_model=EntityExtractionOutput,
            model_id=settings.resolved_extract_model,
        ),
        gleaning_prompt=gleaning_prompt,
        gleaning_chain=cached_chain(
            gleaning_prompt,
            _extract_llm(),
            cache_type="extract",
            output_model=EntityExtractionOutput,
            model_id=settings.resolved_extract_model,
        ),
        # Injected into the extraction user prompt when the chunk carries a heading
        # breadcrumb; str.format()ed by extract_entities, empty string when it does not.
        section_context=_load_domain_prompt("section_context", path)["user_prompt"],
    )


@cache
def _summary_chain() -> Runnable:
    """Merge many descriptions of one entity/relation into one, built on first use.

    Free-form prose, so no schema to bind — StrOutputParser gives the plain
    string the graph stores.
    """
    summary_llm = providers.get_llm(
        model_id=settings.resolved_extract_model,
        max_tokens=None,
        base_url=settings.resolved_extract_host,
        api_key=settings.resolved_extract_api_key,
        timeout=settings.extract_llm_timeout,
    )
    summary_prompt = ChatPromptTemplate.from_messages(
        [("human", load_prompt("summarize_entity_descriptions", KG_PROMPTS_PATH)["user_prompt"])]
    )
    return cached_chain(
        summary_prompt,
        summary_llm | StrOutputParser(),
        cache_type="summary",
        model_id=settings.resolved_extract_model,
    )

# Summaries fan out from merge_nodes_and_edges, whose own semaphore is
# ``llm_model_max_async * 2``. That is a looser bound than the role queue this
# replaced, so cap the concurrency here instead of leaning on the caller.
_summary_gate = asyncio.Semaphore(
    max(1, settings.summary_max_async)
)


async def summarize(inputs: dict, config: dict | None = None) -> str:
    """Run the summary chain under its concurrency cap."""
    async with _summary_gate:
        return await _summary_chain().ainvoke(inputs, config)
