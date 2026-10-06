"""The knowledge-graph extraction contract, as a schema the provider enforces.

``EntityExtractionOutput`` is both halves of the contract at once: it is sent to
the provider as a JSON schema (so the model cannot return a different shape) and
used to parse the reply. That is why the extraction prompt in
``prompts/knowledge_graph.yaml`` carries no output template — the shape lives
here, in one place, and cannot drift from what the parser expects.
"""


import json
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field, field_validator

from hybrid_rag.core.utils import repair_vlm_json_escape_damage


Prose = Annotated[str, AfterValidator(repair_vlm_json_escape_damage)]

class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name in title case.")
    type: str = Field(description="Entity type from the provided type guidance.")
    description: Prose = Field(
        description="Concise description grounded solely in the input text."
    )

class ExtractedRelationship(BaseModel):
    source: str = Field(description="Source entity name, matching an entity above.")
    target: str = Field(description="Target entity name, matching an entity above.")
    keywords: str = Field(
        description="STRICTLY Comma-separated high-level keywords for the relationship."
    )
    description: Prose = Field(description="Why the two entities are connected.")

def _as_list(v: Any) -> Any:
    """Terima array yang ter-serialisasi ganda sebagai string JSON.

    ``strict: true`` memang dikirim (lihat docstring modul), tapi itu cuma
    berlaku kalau providernya benar-benar membatasi sampling. Model di balik
    proxy OpenAI-compatible hanya *berusaha* menuruti skema, dan sesekali
    mengembalikan ``"relationships": "[{...}]"`` — array yang dibungkus string.
    Peluangnya naik seiring panjang daftar, jadi justru chunk terkaya yang
    paling sering kena.

    Kenapa dipulihkan, bukan dibiarkan galat: satu chunk yang gagal membatalkan
    SELURUH dokumen (``extract_entities`` memakai ``asyncio.FIRST_EXCEPTION``,
    lalu membuang hasil chunk yang sudah berhasil). Menukar satu ``json.loads``
    dengan 86 chunk yang harus diulang jelas tidak sebanding.

    Yang bukan JSON valid tetap diteruskan apa adanya, supaya pesan galat
    pydantic yang asli yang muncul — bukan galat parse dari sini.
    """
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return v
    return v


class EntityExtractionOutput(BaseModel):
    entities: list[ExtractedEntity] = Field(
        description=(
            "A real JSON array of entity objects — NEVER a string containing JSON. "
            "List of significant entities explicitly mentioned in the source text. "
            "Merge duplicates and resolve pronouns to their exact full names. "
            "NEVER hallucinate or extract trivial/irrelevant nouns."
        )
    )
    relationships: list[ExtractedRelationship] = Field(
        description=(
            "A real JSON array of relationship objects — NEVER a string containing JSON. "
            "List of factual, explicit connections between the extracted entities. "
            "STRICT CONSTRAINT: The 'source' and 'target' fields MUST perfectly match a 'name' already defined in the 'entities' list. Do not infer implicit connections."
        )
    )

    # ``mode="before"`` supaya jalan SEBELUM pydantic memeriksa tipenya — di
    # titik itulah SDK OpenAI melempar sekarang (``model_validate_json``).
    _coerce = field_validator("entities", "relationships", mode="before")(_as_list)
