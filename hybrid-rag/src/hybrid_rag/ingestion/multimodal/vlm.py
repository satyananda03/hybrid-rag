"""VLM chains for the multimodal items of a document: image, table, equation, generic.

Each is a prompt from ``multimodal.yaml`` piped into the VLM bound to
:class:`VisionAnalysis`, behind the LLM cache. The output models' names and field
descriptions are part of the cache key (their JSON schema is hashed into it), so
renaming them retires every cached analysis.
"""

from functools import cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from hybrid_rag import providers
from hybrid_rag.config import settings
from hybrid_rag.core.llm_cache import cached_chain
from hybrid_rag.core.prompts import load_prompt
from hybrid_rag.core.utils import Prose

MULTIMODAL_PROMPTS_PATH = Path(__file__).resolve().parent / "multimodal.yaml"


class AnalysisEntity(BaseModel):
    entity_name: str = Field(description="Semantic title for this item.")
    entity_type: str = Field(description="image | table | equation | the content type.")
    summary: Prose = Field(description="Concise summary of content and significance.")


class VisionAnalysis(BaseModel):
    detailed_description: Prose = Field(
        description="Comprehensive prose analysis of the item."
    )
    entity_info: AnalysisEntity


def _chat_prompt(target: str, *, with_image: bool) -> ChatPromptTemplate:
    cfg = load_prompt(target, MULTIMODAL_PROMPTS_PATH)
    human: list[dict] = [{"type": "text", "text": cfg["user_prompt"]}]
    if with_image:
        human.append({"type": "image_url", "image_url": {"url": "{image_url}"}})
    return ChatPromptTemplate.from_messages(
        [("system", cfg["system_prompt"]), ("human", human)]
    )


@cache
def _vlm() -> Runnable:
    return providers.get_llm(
        settings.vlm_model,
        max_tokens=None,  # table analyses reach ~2.5K tokens; a cap truncates the structured output
        base_url=settings.resolved_vlm_binding_host,
        api_key=settings.resolved_vlm_binding_api_key,
        timeout=settings.vlm_timeout,
    ).with_structured_output(VisionAnalysis)


@cache
def analysis_chain(target: str) -> Runnable:
    """The cached chain for one ``multimodal.yaml`` entry (``image_analysis``, ...)."""
    return cached_chain(
        _chat_prompt(target, with_image=target == "image_analysis"),
        _vlm(),
        cache_type="vlm",
        output_model=VisionAnalysis,
        model_id=settings.vlm_model,
    )


def surrounding_context_block(leading: str, trailing: str) -> str:
    """Render the neighbours block, or "" when the item has no neighbours."""
    if not (leading or trailing):
        return ""
    template = load_prompt("surrounding_context", MULTIMODAL_PROMPTS_PATH)["user_prompt"]
    return template.format(leading=leading or "None", trailing=trailing or "None")
