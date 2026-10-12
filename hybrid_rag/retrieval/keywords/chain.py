"""LangChain chain for the query side of the graph: keyword extraction.

A prompt template from YAML piped into a model bound to a Pydantic output schema,
plus a cache lookup inside the chain (see ``core/llm_cache.py``). ``ainvoke``
returns a validated model, so callers read fields off it instead of parsing a
response body.

The model uses its own role binding (``KEYWORD_*``), falling back to the base
LLM settings. Built on first use, after the host has configured the providers.
"""

from functools import cache
from pathlib import Path

from langchain_core.runnables import Runnable

from hybrid_rag import providers
from hybrid_rag.config import settings
from hybrid_rag.core.llm_cache import cached_chain
from hybrid_rag.core.prompts import load_chat_prompt_template
from hybrid_rag.retrieval.keywords.schema import KeywordsOutput

KEYWORDS_PROMPT_PATH = Path(__file__).resolve().parent / "keywords.yaml"


@cache
def keywords_chain() -> Runnable:
    """The keyword chain; ``extract_keywords_only`` keys its cache entries by ``CACHE_MODE``."""
    llm = providers.get_llm(
        model_id=settings.keyword_llm_model,
        max_tokens=None,
        base_url=settings.keyword_llm_binding_host,
        api_key=settings.keyword_llm_binding_api_key,
        timeout=settings.keyword_llm_timeout,
    ).with_structured_output(KeywordsOutput)
    return cached_chain(
        load_chat_prompt_template(KEYWORDS_PROMPT_PATH, "keywords_extraction"),
        llm,
        cache_type="keywords",
        output_model=KeywordsOutput,
        model_id=settings.keyword_llm_model,
    )
