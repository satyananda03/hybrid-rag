"""The knowledge-graph extraction contract, as a schema the provider enforces."""

import json
import logging
from typing import Any, Literal, get_args, get_origin
from pydantic import BaseModel, Field, ValidationError, model_validator
from hybrid_rag.ingestion.text_cleaning import Prose

logger = logging.getLogger(__name__)

EntityType = Literal[
    "Regulation",
    "Article",
    "Subsector",
    "Organization",
    "Role",
    "Person",
    "Program",
    "Concept",
]

class ExtractedEntity(BaseModel):
    name: str = Field(
        description=(
            "Canonical name in Title Case (stored title-cased; keep acronyms in ALL CAPS only in description). "
            "Prefix 'Article' with parent regulation (e.g., 'UU No. 24 Tahun 2019 - Pasal 9'), "
            "prefix 'Subsector' with 'Subsektor', and keep 'Person' free of academic/job titles."
        )
    )
    type: EntityType = Field(
        description="Strict classification from the 8 allowed types. Keep Person, Role, and Organization separate."
    )
    description: Prose = Field(
        description=(
            "Concise third-person summary in Bahasa Indonesia grounded solely in the text, without pronouns. "
            "Include academic titles/NIP for Person, Ayat/Huruf details for Article, and table metrics here."
        )
    )


class ExtractedRelationship(BaseModel):
    source: str = Field(
        description="One endpoint entity name, matching an extracted entity 'name' exactly."
    )
    target: str = Field(
        description="The other endpoint entity name, matching an extracted entity 'name' exactly."
    )
    keywords: str = Field(
        description="STRICTLY comma-separated relational verbs/phrases in Bahasa Indonesia (e.g., 'menjabat sebagai', 'mengatur', 'bagian dari')."
    )
    description: Prose = Field(
        description="Concise third-person explanation in Bahasa Indonesia of how source and target are connected."
    )


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


# Nama Indonesia dari tipe yang sama. Prompt-nya berbahasa Indonesia, jadi model
# kadang menulis tipenya juga dalam bahasa Indonesia. Hanya padanan persis; nilai
# hasilnya selalu salah satu anggota ``EntityType``.
_TYPE_ALIASES = {
    "peraturan": "Regulation",
    "regulasi": "Regulation",
    "pasal": "Article",
    "subsektor": "Subsector",
    "organisasi": "Organization",
    "lembaga": "Organization",
    "instansi": "Organization",
    "jabatan": "Role",
    "peran": "Role",
    "orang": "Person",
    "tokoh": "Person",
    "konsep": "Concept",
}


def _canonical_entity_type(raw: Any, allowed: tuple[str, ...]) -> str | None:
    """Tipe resmi yang dimaksud ``raw`` (beda huruf besar/kecil, spasi, atau nama Indonesia), atau None."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    official = next((t for t in allowed if t.lower() == key), None) or _TYPE_ALIASES.get(key)
    return official if official in allowed else None


def _with_canonical_type(record: Any, allowed: tuple[str, ...]) -> Any:
    """``record`` dengan ``type`` diganti tipe resminya; tipe yang tak dikenali dibiarkan agar validasi menolaknya."""
    if not isinstance(record, dict):
        return record
    official = _canonical_entity_type(record.get("type"), allowed)
    return {**record, "type": official} if official else record


def _join_keywords(record: Any) -> Any:
    """``keywords`` yang dikirim sebagai list → string dipisah koma, bentuk yang diminta schema."""
    keywords = record.get("keywords") if isinstance(record, dict) else None
    if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
        return {**record, "keywords": ", ".join(k.strip() for k in keywords if k.strip())}
    return record


def _refers_to(relationship: Any, names: set[str]) -> bool:
    """Apakah ``source`` atau ``target`` relasi ini salah satu dari ``names``."""
    return isinstance(relationship, dict) and any(
        isinstance(relationship.get(end), str) and relationship[end] in names
        for end in ("source", "target")
    )


def _valid_records(records: list, model: type[BaseModel], kind: str) -> tuple[list[BaseModel], list[Any]]:
    """Validasi tiap record sendiri-sendiri: ``(instance yang valid, record mentah yang tidak valid)``.

    Record yang tidak valid dicatat bersama key yang ada di dalamnya — dari situ
    terlihat kalau model menulis sebuah field dengan nama lain.
    """
    valid, invalid = [], []
    for index, record in enumerate(records):
        try:
            valid.append(model.model_validate(record))
        except ValidationError as e:
            invalid.append(record)
            keys = sorted(record) if isinstance(record, dict) else type(record).__name__
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or '<record>'}: {err['msg']}" for err in e.errors()
            )
            logger.warning("Dropping extracted %s #%d (keys: %s): %s", kind, index, keys, problems)
    return valid, invalid


class EntityExtractionOutput(BaseModel):
    entities: list[ExtractedEntity] = Field(
        description=(
            "Real JSON array of canonical entity objects. Merge duplicates, split Person/Role/Organization, "
            "and never extract generic nouns or unlisted types."
        )
    )
    relationships: list[ExtractedRelationship] = Field(
        description=(
            "Real JSON array of relationship objects without duplicates or reverse duplicates. "
            "Both 'source' and 'target' MUST exactly match a 'name' in 'entities'."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_invalid_records(cls, data: Any) -> Any:
        """Buang record yang tidak valid, bukan seluruh balasannya.

        ``strict: true`` dikirim, tapi proxy OpenAI-compatible tidak menjamin
        sampling dibatasi, jadi sesekali satu record keluar dari schema (field
        wajib hilang, tipe di luar ``Literal``). Tanpa ini satu record itu
        menggagalkan balasan chunk, dan lewat ``FIRST_EXCEPTION`` seluruh dokumen.

        Jalan sebelum pydantic memeriksa tipe (``mode="before"``) — di titik itulah
        SDK OpenAI memanggil ``model_validate_json``. Urutannya:

        1. Array yang terbungkus string JSON dipulihkan (``_as_list``).
        2. Perbaikan yang tidak ambigu saja: tipe entity ke nilai resmi ``Literal``
           (jalur default), ``keywords`` berupa list jadi string.
        3. Tiap entity divalidasi sendiri; yang tidak valid dibuang.
        4. Relasi ke entity yang dibuang ikut dibuang — kalau tidak, merge
           membuatkan node placeholder bertipe ``UNKNOWN``.
        5. Tiap relasi divalidasi sendiri; yang tidak valid dibuang.

        Record yang rusak tidak pernah ditebak atau dilengkapi. Jalur ontology
        (``type: str``) melewati langkah tipe, jadi labelnya tetap persis; model
        record-nya diambil dari anotasi field, jadi validator ini berlaku sama
        untuk ``OntologyExtractionOutput``. Kalau ``entities`` sendiri bukan
        array, balasannya memang tidak bisa diselamatkan dan dibiarkan gagal.
        """
        if not isinstance(data, dict):
            return data
        entities = _as_list(data.get("entities"))
        if not isinstance(entities, list):
            return data
        relationships = _as_list(data.get("relationships"))
        if not isinstance(relationships, list):
            logger.warning("Dropping extracted relationships: expected an array, got %r", type(relationships).__name__)
            relationships = []

        entity_model = get_args(cls.model_fields["entities"].annotation)[0]
        relationship_model = get_args(cls.model_fields["relationships"].annotation)[0]
        type_annotation = entity_model.model_fields["type"].annotation
        if get_origin(type_annotation) is Literal:
            entities = [_with_canonical_type(e, get_args(type_annotation)) for e in entities]

        kept_entities, dropped_entities = _valid_records(entities, entity_model, "entity")
        # A name that another, valid entity also carries still exists in this response.
        dropped_names = {
            e["name"] for e in dropped_entities if isinstance(e, dict) and isinstance(e.get("name"), str)
        } - {e.name for e in kept_entities}
        relationships = [_join_keywords(r) for r in relationships if not _refers_to(r, dropped_names)]
        kept_relationships, _ = _valid_records(relationships, relationship_model, "relationship")
        return {**data, "entities": kept_entities, "relationships": kept_relationships}

class ExtractedProperty(BaseModel):
    name: str = Field(description="Property name, exactly as defined for the type in the schema.")
    value: str = Field(description="Property value, as stated in the input text.")


class OntologyEntity(ExtractedEntity):
    type: str = Field(description="Node type label from the schema, spelled exactly.")
    properties: list[ExtractedProperty] = Field(
        description="Values of the schema properties of this node type that the input text states."
    )


class OntologyRelationship(ExtractedRelationship):
    type: str = Field(description="Relationship type label from the schema, spelled exactly.")
    properties: list[ExtractedProperty] = Field(
        description="Properties of the relationship that the input text states."
    )


class OntologyExtractionOutput(EntityExtractionOutput):
    entities: list[OntologyEntity] = Field(
        description=EntityExtractionOutput.model_fields["entities"].description
    )
    relationships: list[OntologyRelationship] = Field(
        description=(
            EntityExtractionOutput.model_fields["relationships"].description
            + " Each relationship goes from `source` to `target` as one of the schema's patterns."
        )
    )


# STRUCTURED OUTPUT V1
# import json
# from typing import Any
# from pydantic import BaseModel, Field, field_validator
# from hybrid_rag.ingestion.text_cleaning import Prose

# class ExtractedEntity(BaseModel):
#     name: str = Field(description="Entity name in title case.")
#     type: str = Field(description="Entity type from the provided type guidance.")
#     description: Prose = Field(
#         description="Concise description grounded solely in the input text."
#     )

# class ExtractedRelationship(BaseModel):
#     source: str = Field(description="Source entity name, matching an entity above.")
#     target: str = Field(description="Target entity name, matching an entity above.")
#     keywords: str = Field(
#         description="STRICTLY Comma-separated high-level keywords for the relationship."
#     )
#     description: Prose = Field(description="Why the two entities are connected.")

# def _as_list(v: Any) -> Any:
#     """Terima array yang ter-serialisasi ganda sebagai string JSON.

#     ``strict: true`` memang dikirim (lihat docstring modul), tapi itu cuma
#     berlaku kalau providernya benar-benar membatasi sampling. Model di balik
#     proxy OpenAI-compatible hanya *berusaha* menuruti skema, dan sesekali
#     mengembalikan ``"relationships": "[{...}]"`` — array yang dibungkus string.
#     Peluangnya naik seiring panjang daftar, jadi justru chunk terkaya yang
#     paling sering kena.

#     Kenapa dipulihkan, bukan dibiarkan galat: satu chunk yang gagal membatalkan
#     SELURUH dokumen (``extract_entities`` memakai ``asyncio.FIRST_EXCEPTION``,
#     lalu membuang hasil chunk yang sudah berhasil). Menukar satu ``json.loads``
#     dengan 86 chunk yang harus diulang jelas tidak sebanding.

#     Yang bukan JSON valid tetap diteruskan apa adanya, supaya pesan galat
#     pydantic yang asli yang muncul — bukan galat parse dari sini.
#     """
#     if isinstance(v, str):
#         try:
#             return json.loads(v)
#         except (ValueError, TypeError):
#             return v
#     return v

# class EntityExtractionOutput(BaseModel):
#     entities: list[ExtractedEntity] = Field(
#         description=(
#             "A real JSON array of entity objects — NEVER a string containing JSON. "
#             "List of significant entities explicitly mentioned in the source text. "
#             "Merge duplicates and resolve pronouns to their exact full names. "
#             "NEVER hallucinate or extract trivial/irrelevant nouns."
#         )
#     )
#     relationships: list[ExtractedRelationship] = Field(
#         description=(
#             "A real JSON array of relationship objects — NEVER a string containing JSON. "
#             "List of factual, explicit connections between the extracted entities. "
#             "STRICT CONSTRAINT: The 'source' and 'target' fields MUST perfectly match a 'name' already defined in the 'entities' list. Do not infer implicit connections."
#         )
#     )

#     # ``mode="before"`` supaya jalan SEBELUM pydantic memeriksa tipenya — di
#     # titik itulah SDK OpenAI melempar sekarang (``model_validate_json``).
#     _coerce = field_validator("entities", "relationships", mode="before")(_as_list)