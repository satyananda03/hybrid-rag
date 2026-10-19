"""Orphan repair: one relationship-only call for the entities a chunk's extraction left unlinked.

For documents ingested with ``orphan_repair`` (the default), ``extract_entities`` runs
it after the first pass (and gleaning) of a chunk that has unlinked entities:
:func:`unlinked_names` finds them, :func:`repair_inputs` fills the
``orphan_relation_repair`` prompt, and :func:`accept_repair` keeps only the edges that
are safe to merge. With ``prune_isolated``, whatever stays unlinked is deleted at the
end of the batch (``_prune_isolated_entities``).

The repair call shares the extraction system prompt, whose contract asks for every
relationship endpoint to be an entity of the same response. Given an ``entities`` field,
the model answers that by echoing the whole ``<unlinked>`` list back. So the repair
contract is the extraction one with ``entities`` hidden from the model: it can only
answer with relationships, whose endpoints are the names the prompt lists. The field
stays on the class (empty), so the parser, the validator and the cache read a repair
reply like any other extraction reply.
"""

from collections.abc import Collection
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hybrid_rag.ingestion.knowledge_graph.schema import (
    EntityExtractionOutput,
    ExtractedEntity,
    ExtractedRelationship,
    OntologyEntity,
    OntologyExtractionOutput,
    OntologyRelationship,
)

_RELATIONSHIPS = (
    "Relationships with at least one endpoint in `<unlinked>`; `source` and `target` copied exactly "
    "from `<entities>` or `<unlinked>`."
)


def _hide_entities(schema: dict[str, Any]) -> None:
    """``json_schema_extra`` hook: the schema sent to the model has no ``entities`` field."""
    schema["properties"].pop("entities", None)
    schema["required"] = [name for name in schema.get("required", []) if name != "entities"]


class _RelationshipsOnly(BaseModel):
    """What the two repair contracts share: ``entities`` is not in the schema, and is empty on parse."""

    model_config = ConfigDict(json_schema_extra=_hide_entities)

    @model_validator(mode="before")
    @classmethod
    def _no_entities(cls, data: Any) -> Any:
        # Runs before the extraction validator (subclass validators come first), which
        # reads ``entities`` to decide whether the reply is a usable object at all.
        return {"entities": [], **data} if isinstance(data, dict) else data


class OrphanRepairOutput(_RelationshipsOnly, EntityExtractionOutput):
    """Relationships for entities the extraction left unlinked."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(description=_RELATIONSHIPS)


class OntologyOrphanRepairOutput(_RelationshipsOnly, OntologyExtractionOutput):
    """Relationships for entities the extraction left unlinked."""

    entities: list[OntologyEntity] = Field(default_factory=list)
    relationships: list[OntologyRelationship] = Field(
        description=_RELATIONSHIPS + " Each relationship goes from `source` to `target` as one of the schema's patterns."
    )


def unlinked_names(maybe_nodes: dict, maybe_edges: dict) -> list[str]:
    """The chunk's entities that no edge touches, in extraction order."""
    linked = {name for edge_key in maybe_edges for name in edge_key}
    return [name for name in maybe_nodes if name not in linked]


def repair_inputs(chain_inputs: dict, maybe_nodes: dict, orphans: Collection[str]) -> dict:
    """The extraction inputs plus the two name lists the repair prompt reads."""
    return {
        **chain_inputs,
        "entity_names": "\n".join(name for name in maybe_nodes if name not in orphans),
        "orphan_names": "\n".join(orphans),
    }


def accept_repair(maybe_nodes: dict, maybe_edges: dict, orphans: Collection[str], repair_edges: dict) -> dict:
    """The repair edges worth keeping: between nodes of the chunk, touching an orphan, not already present.

    An endpoint the chunk never extracted would become an ``UNKNOWN`` placeholder node at merge.
    """
    return {
        (src, tgt): records
        for (src, tgt), records in repair_edges.items()
        if src in maybe_nodes
        and tgt in maybe_nodes
        and (src in orphans or tgt in orphans)
        and (src, tgt) not in maybe_edges
        and (tgt, src) not in maybe_edges
    }
