"""The query-keyword contract, as a schema the provider enforces.

``KeywordsOutput`` is both halves of the contract at once: it is sent to the
provider as a JSON schema (so the model cannot return a different shape) and used
to parse the reply. That is why the prompt in ``keywords.yaml`` carries no output
template — the shape lives here, in one place, and cannot drift from the parser.
"""

from pydantic import BaseModel, Field


class KeywordsOutput(BaseModel):
    """Query keywords, split by the retrieval index each side searches."""

    high_level_keywords: list[str] = Field(
        description="Overarching concepts or themes; searched against relations."
    )
    low_level_keywords: list[str] = Field(
        description="Specific entities, names or details; searched against entities."
    )
