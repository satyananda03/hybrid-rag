"""Chains for the multimodal items of a document.

- :func:`image_analysis_chain` — sends image + prompt to the VLM (vision-capable model).
- :func:`table_analysis_chain` — sends text-only table prompt to the extractor LLM.
- :func:`equation_analysis_chain` — sends text-only equation prompt to the extractor LLM.
- :func:`slide_transcription_chain` — sends a whole slide + its text layer to the VLM, returns Markdown.
"""

from functools import cache
from pathlib import Path
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, computed_field

from hybrid_rag import providers
from hybrid_rag.config import settings
from hybrid_rag.core.llm_cache import cached_chain
from hybrid_rag.core.prompts import load_prompt
from hybrid_rag.ingestion.text_cleaning import Prose

MULTIMODAL_PROMPTS_PATH = Path(__file__).resolve().parent / "multimodal.yaml"

# IMAGE MODALITY
class ImageEntity(BaseModel):
    entity_name: str = Field(
        description=(
            "Canonical Knowledge Graph node identifier formatted as a Title Case noun phrase (strictly 3 to 7 words). "
            "Combine the visual type and core subject/system (e.g., 'Flowchart Verifikasi Dokumen SIPD', 'Arsitektur Jaringan Server Pusdatin'). "
            "RULES: 1. NO clauses, verbs, or conjunctions ('yang', 'untuk', 'menunjukkan', 'pada'). "
            "2. NO raw filenames, figure numbers ('Gambar 1'), or vague labels ('Diagram Proses')."
        )
    )
    summary: Prose = Field(
        description=(
            "A 1-2 sentence overview of what the image depicts and its functional purpose in the document. "
            "Start directly with the core subject; NEVER use filler openings like 'Gambar ini menunjukkan'."
        )
    )

    @computed_field
    @property
    def entity_type(self) -> Literal["image"]:
        return "image"


class ImageAnalysis(BaseModel):
    entity_info: ImageEntity
    detailed_description: Prose = Field(
        description=(
            "A dense, factual explanation of the image content in 2 to 4 paragraphs (no visual styling). "
            "Include key visible text, system metrics, workflow sequences, and component relationships. "
            "Avoid repetitive sentence structures."
        )
    )


# TABLE MODALITY
class TableEntity(BaseModel):
    entity_name: str = Field(
        description=(
            "Canonical Knowledge Graph node identifier formatted as a Title Case noun phrase (strictly 3 to 7 words). "
            "Represent the exact dataset subject and scope (e.g., 'Tabel Spesifikasi Hardware Server', 'Daftar Jejaring Kabupaten Kota Sumbar'). "
            "RULES: 1. NO clauses, verbs, or conjunctions ('yang', 'untuk', 'berisi', 'pada'). "
            "2. NO generic labels ('Table 1', 'Tabel Data', 'Sheet 03')."
        )
    )
    summary: Prose = Field(
        description=(
            "A 1-2 sentence factual overview stating the dataset's subject, purpose, and structural dimensions. "
            "Start directly with the core subject; NEVER use filler openings like 'Tabel ini berisi' or subjective insights."
        )
    )

    @computed_field
    @property
    def entity_type(self) -> Literal["table"]:
        return "table"


class TableAnalysis(BaseModel):
    entity_info: TableEntity
    detailed_description: Prose = Field(
        description=(
            "A compact synthesis of the table's schema (column headers/units) and key data patterns in 2 to 4 paragraphs of natural prose. "
            "CRITICAL: NEVER transcribe row-by-row, NEVER enumerate every entity/region, NEVER list empty cells individually, and NEVER calculate unprinted totals. "
            "Group rows into high-level categories, cite 2-3 examples, and report only explicitly printed summary numbers."
        )
    )

# EQUATION MODALITY
class EquationEntity(BaseModel):
    entity_name: str = Field(
        description=(
            "A clean, semantic, distinctive title identifying the mathematical formula, law, or calculation (e.g., 'Rumus Indeks Kepuasan Masyarakat'). "
            "NEVER use generic labels like 'Equation 1' or 'Rumus 1'."
        )
    )
    summary: Prose = Field(
        description=(
            "An overall summary stating exactly what this equation calculates or models, its domain context, and its practical significance. "
            "NEVER use introductory filler phrases like 'Persamaan ini digunakan untuk', 'Rumus ini adalah', or 'This equation'. Start directly with the core subject."
        )
    )

    @computed_field
    @property
    def entity_type(self) -> Literal["equation"]:
        return "equation"


class EquationAnalysis(BaseModel):
    detailed_description: Prose = Field(
        description=(
            "A comprehensive breakdown of the mathematical expression in natural prose (no bullet points). "
            "Include the exact transcription of the formula (in standard notation or LaTeX), explicitly define every variable, symbol, subscript, and constant, and explain their mathematical relationships. "
            "Always use specific entity names and variable names so every sentence is fully self-contained. "
            "NEVER invent variable definitions not supported by the context. "
            "NEVER use meta-references like 'Persamaan ini menunjukkan', 'Berdasarkan rumus di atas', or 'Variabel X adalah'. Start directly with the factual mathematical breakdown."
        )
    )
    entity_info: EquationEntity


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
    settings.require("vlm")
    return providers.get_vlm(
        model_id=settings.vlm_model,
        base_url=settings.vlm_binding_host,
        api_key=settings.vlm_binding_api_key,
        timeout=settings.vlm_timeout,
    )


@cache
def _vlm_image() -> Runnable:
    return _vlm().with_structured_output(ImageAnalysis)


@cache
def _extract_llm_table() -> Runnable:
    """Text-only LLM for table modality."""
    return providers.get_llm(
        model_id=settings.extract_llm_model,
        base_url=settings.extract_llm_binding_host,
        api_key=settings.extract_llm_binding_api_key,
        timeout=settings.extract_llm_timeout,
    ).with_structured_output(TableAnalysis)


@cache
def _extract_llm_equation() -> Runnable:
    """Text-only LLM for equation modality."""
    return providers.get_llm(
        model_id=settings.extract_llm_model,
        base_url=settings.extract_llm_binding_host,
        api_key=settings.extract_llm_binding_api_key,
        timeout=settings.extract_llm_timeout,
    ).with_structured_output(EquationAnalysis)

@cache
def slide_transcription_chain() -> Runnable:
    """Vision chain for one whole slide: Markdown text that is ingested as ordinary text."""
    return cached_chain(
        _chat_prompt("slide_transcription", with_image=True),
        _vlm() | StrOutputParser(),
        cache_type="vlm",
        model_id=settings.vlm_model,
    )


@cache
def image_analysis_chain() -> Runnable:
    """Vision chain: prompt + image sent to the VLM."""
    return cached_chain(
        _chat_prompt("image_analysis", with_image=True),
        _vlm_image(),
        cache_type="vlm",
        output_model=ImageAnalysis,
        model_id=settings.vlm_model,
    )


@cache
def table_analysis_chain() -> Runnable:
    """Text-only chain for tables — uses extractor LLM, no VLM."""
    return cached_chain(
        _chat_prompt("table_analysis", with_image=False),
        _extract_llm_table(),
        cache_type="vlm",
        output_model=TableAnalysis,
        model_id=settings.extract_llm_model,
    )

@cache
def equation_analysis_chain() -> Runnable:
    """Text-only chain for equations — uses extractor LLM, no VLM."""
    return cached_chain(
        _chat_prompt("equation_analysis", with_image=False),
        _extract_llm_equation(),
        cache_type="vlm",
        output_model=EquationAnalysis,
        model_id=settings.extract_llm_model,
    )


def surrounding_context_block(leading: str, trailing: str) -> str:
    """Render the neighbours block, or "" when the item has no neighbours."""
    if not (leading or trailing):
        return ""
    template = load_prompt("surrounding_context", MULTIMODAL_PROMPTS_PATH)["user_prompt"]
    return template.format(leading=leading or "None", trailing=trailing or "None")
