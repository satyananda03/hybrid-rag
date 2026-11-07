"""The position of a heading and of a parent in a document's text."""
from typing import NamedTuple


class Heading(NamedTuple):
    start: int
    end: int
    level: int
    text: str

class ParentSpan(NamedTuple):
    """``content[start:end]`` is the parent's text; ``headings`` its section path, outermost first."""

    start: int
    end: int
    headings: tuple[Heading, ...]
