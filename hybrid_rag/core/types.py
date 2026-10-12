
from pydantic import BaseModel, Field
from typing import Any, Optional


class ExtractedEntity(BaseModel):
    """A single entity extracted from text by the LLM."""

    entity_name: str = Field(
        description="Name of the entity. Use title case for case-insensitive names."
    )
    entity_type: str = Field(description="Type/category of the entity.")
    entity_description: str = Field(
        description="Concise yet comprehensive description of the entity based on the input text."
    )


class ExtractedRelationship(BaseModel):
    """A single relationship between two entities extracted from text."""

    source_entity: str = Field(
        description="Name of the source entity in the relationship."
    )
    target_entity: str = Field(
        description="Name of the target entity in the relationship."
    )
    relationship_keywords: str = Field(
        description="Comma-separated high-level keywords summarizing the relationship."
    )
    relationship_description: str = Field(
        description="Concise explanation of the relationship between source and target entities."
    )


class KnowledgeGraphNode(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraphEdge(BaseModel):
    id: str
    type: Optional[str]
    source: str  # id of source node
    target: str  # id of target node
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraph(BaseModel):
    nodes: list[KnowledgeGraphNode] = []
    edges: list[KnowledgeGraphEdge] = []
    is_truncated: bool = False
