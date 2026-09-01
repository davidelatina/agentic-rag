
from pydantic import BaseModel


class SparseVector(BaseModel):
    """Lexical sparse embedding components for hybrid search."""

    indices: list[int]
    values: list[float]


class RetrievedChunk(BaseModel):
    """A single knowledge-base passage selected for agent context."""

    text: str
    source: str