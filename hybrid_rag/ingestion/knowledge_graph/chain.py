"""LangChain chains for the text side of the graph.

Extraction, gleaning, and the entity/relation description summary.

A prompt template from YAML piped into a model bound to a Pydantic output
schema, plus a cache lookup inside the chain
(see ``core/llm_cache.py``). ``ainvoke`` returns a validated model, so callers
read fields off it instead of parsing a response body.

Extraction and gleaning are built per *ontology* by ``get_kg_chains``: a document
selects which ontology (node types, relationship types, patterns) it is extracted
under, or none for the default contract, and one workspace may hold several.
``lru_cache`` keeps the "built once" property per ontology. The chain *structure*
is the same for every ontology — only the system prompt and output model differ.

Each chain talks to its own role binding (``EXTRACT_*``), falling back to the
base LLM settings. Models come from ``hybrid_rag.providers`` and are built on
first use. The VLM chains live in ``multimodal/chain.py``.
"""


import asyncio
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from hybrid_rag import providers
from hybrid_rag.config import NOVA_MODELS, settings
from hybrid_rag.ingestion.knowledge_graph.entity_repairment import (
    OntologyOrphanRepairOutput,
    OrphanRepairOutput,
)
from hybrid_rag.ingestion.knowledge_graph.ontology_schema import (
    load_ontology_schema,
    render_schema_for_prompt,
)
from hybrid_rag.ingestion.knowledge_graph.schema import (
    EntityExtractionOutput,
    OntologyExtractionOutput,
)
from hybrid_rag.core.llm_cache import cached_chain
from hybrid_rag.core.prompts import load_prompt

if TYPE_CHECKING:
    from neo4j_graphrag.components.schema import GraphSchema

KG_PROMPTS_PATH = Path(__file__).resolve().parent / "knowledge_graph.yaml"

@cache
def _extract_llm(output_model: type = EntityExtractionOutput) -> Runnable:
    """The extraction model bound to one output schema, built on first use."""
    return providers.get_llm(
        model_id=settings.extract_llm_model,
        max_tokens=None,  # a capped structured output is truncated JSON: 40% of extractions exceed 900 tokens
        temperature=0.0,
        base_url=settings.extract_llm_binding_host,
        api_key=settings.extract_llm_binding_api_key,
        timeout=settings.extract_llm_timeout,
    ).with_structured_output(output_model)

class KGChains(NamedTuple):
    """One ontology's extraction contract, built and cached as a unit."""

    extraction_prompt: ChatPromptTemplate
    extraction_chain: Runnable
    gleaning_prompt: ChatPromptTemplate
    gleaning_chain: Runnable
    #: Asks only for relationships of entities the earlier passes left unlinked.
    repair_chain: Runnable
    section_context: str
    #: The ontology's GraphSchema, which extraction prunes against; None when the
    #: document has no ontology (the default contract, no validation).
    schema: "GraphSchema | None" = None


def _prompt(key: str) -> dict[str, str]:
    return load_prompt(key, KG_PROMPTS_PATH)


def get_kg_chains(ontology: str | None = None) -> KGChains:
    """Return the extraction + gleaning chains for one ontology, built once each.

    ``ontology`` names a file in ``ontology/``. It swaps the extraction system
    prompt for ``ontology_entity_extraction`` filled with that ontology's schema,
    and the output contract for ``OntologyExtractionOutput``. ``None`` and ``""``
    both mean the default contract and share one cache entry.
    """
    return _build_chains(ontology or None)


@lru_cache(maxsize=8)
def _build_chains(ontology: str | None) -> KGChains:
    """Build one contract's chains.

    ``load_prompt`` caches the *parsed YAML*, not the built objects, so without this
    cache every chunk would rebuild two ChatPromptTemplates and two cached_chain
    closures. That is what makes this the registry — there is no registry class.
    """
    extraction = _prompt("entity_extraction")
    schema = None
    output_model = EntityExtractionOutput
    repair_output = OrphanRepairOutput
    if ontology:
        schema = load_ontology_schema(ontology)
        output_model = OntologyExtractionOutput
        repair_output = OntologyOrphanRepairOutput
        extraction = {
            **extraction,
            "system_prompt": _prompt("ontology_entity_extraction")["system_prompt"].replace(
                "{ontology_schema}", render_schema_for_prompt(schema)
            ),
        }
    # Schema tool + system prompt identik untuk setiap chunk satu ontology (~1.9K token, di atas minimum
    # cache Bedrock). Nova tidak cache otomatis, jadi cachePoint dipasang setelah system; LiteLLM
    # menerjemahkan cache_control ke cachePoint. Teks prompt tidak berubah.
    system = (
        [{"type": "text", "text": extraction["system_prompt"], "cache_control": {"type": "ephemeral"}}]
        if settings.extract_llm_model in NOVA_MODELS
        else extraction["system_prompt"]
    )

    extraction_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", extraction["user_prompt"])]
    )

    # The gleaning pass runs under the extraction rules, so it reuses that system
    # prompt rather than keeping a second copy in the YAML. The placeholder replays
    # the first pass (its user prompt and the entities it returned) so the model can
    # tell what it has already covered. Because the system prompt comes from the
    # ontology, gleaning follows the ontology automatically.
    gleaning_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("messages"),
            ("human", _prompt("entity_continue_extraction")["user_prompt"]),
        ]
    )

    repair_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", _prompt("orphan_relation_repair")["user_prompt"])]
    )

    return KGChains(
        extraction_prompt=extraction_prompt,
        extraction_chain=cached_chain(
            extraction_prompt,
            _extract_llm(output_model),
            cache_type="extract",
            output_model=output_model,
            model_id=settings.extract_llm_model,
        ),
        gleaning_prompt=gleaning_prompt,
        gleaning_chain=cached_chain(
            gleaning_prompt,
            _extract_llm(output_model),
            cache_type="extract",
            output_model=output_model,
            model_id=settings.extract_llm_model,
        ),
        repair_chain=cached_chain(
            repair_prompt,
            _extract_llm(repair_output),
            cache_type="extract",
            output_model=repair_output,
            model_id=settings.extract_llm_model,
        ),
        # Injected into the extraction user prompt when the chunk carries a heading
        # breadcrumb; str.format()ed by extract_entities, empty string when it does not.
        section_context=_prompt("section_context")["user_prompt"],
        schema=schema,
    )


@cache
def _summary_chain() -> Runnable:
    """Merge many descriptions of one entity/relation into one, built on first use.

    Free-form prose, so no schema to bind — StrOutputParser gives the plain
    string the graph stores.
    """
    summary_llm = providers.get_llm(
        model_id=settings.extract_llm_model,
        max_tokens=None,
        base_url=settings.extract_llm_binding_host,
        api_key=settings.extract_llm_binding_api_key,
        timeout=settings.extract_llm_timeout,
    )
    summary_prompt = ChatPromptTemplate.from_messages(
        [("human", _prompt("summarize_entity_descriptions")["user_prompt"])]
    )
    return cached_chain(
        summary_prompt,
        summary_llm | StrOutputParser(),
        cache_type="summary",
        model_id=settings.extract_llm_model,
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
