"""Text chunking utilities for RAG indexing."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from config import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_SIZE_TOKENS
from models import Chunk, Document


TOKEN_PATTERN = re.compile(r"\S+")


def tokenize(text: str) -> list[str]:
    """Tokenize text with a deterministic whitespace-oriented tokenizer."""

    return TOKEN_PATTERN.findall(text)


def chunk_documents(
    documents: Iterable[Document],
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk documents into fixed token windows while preserving metadata."""

    _validate_chunk_settings(chunk_size_tokens, chunk_overlap_tokens)

    chunks: list[Chunk] = []
    for document_index, document in enumerate(documents):
        chunks.extend(
            chunk_document(
                document,
                document_index=document_index,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        )
    return chunks


def chunk_document(
    document: Document,
    document_index: int,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk a single document into overlapping token windows."""

    _validate_chunk_settings(chunk_size_tokens, chunk_overlap_tokens)

    tokens = tokenize(document.text)
    if not tokens:
        return []

    step = chunk_size_tokens - chunk_overlap_tokens
    chunks: list[Chunk] = []

    for chunk_index, start in enumerate(range(0, len(tokens), step)):
        end = min(start + chunk_size_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break

        text = " ".join(chunk_tokens)
        metadata = {
            **document.metadata,
            "document_index": document_index,
            "chunk_index": chunk_index,
            "start_token": start,
            "end_token": end,
        }
        chunks.append(Chunk(id=_chunk_id(text, metadata), text=text, metadata=metadata))

        if end >= len(tokens):
            break

    return chunks


def _validate_chunk_settings(chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive.")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative.")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")


def _chunk_id(text: str, metadata: dict[str, object]) -> str:
    source = str(metadata.get("source", "unknown"))
    page = str(metadata.get("page", "none"))
    cell = str(metadata.get("cell_index", "none"))
    start = str(metadata.get("start_token", 0))
    digest = hashlib.sha1(f"{source}:{page}:{cell}:{start}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]
