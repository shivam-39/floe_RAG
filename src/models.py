"""Shared data models used across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A loaded source document or page with provenance metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A chunk of document text ready for embedding and citation."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with vector similarity score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RagResult:
    """The final RAG response returned to callers."""

    answer: str
    sources: list[RetrievedChunk]
    prompt: str
