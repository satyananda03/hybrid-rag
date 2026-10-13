"""The query-keyword contract, as a schema the provider enforces."""

from pydantic import BaseModel, Field

class KeywordsOutput(BaseModel):
    """Query keywords, split by the retrieval index each side searches."""
    high_level_keywords: list[str] = Field(
        description="Overarching concepts or themes; searched against relations."
    )
    low_level_keywords: list[str] = Field(
        description="Specific entities, names or details; searched against entities."
    )