"""Default paths and runtime settings for the RAG project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index_store"

DEFAULT_CHUNK_SIZE_TOKENS = 400
DEFAULT_CHUNK_OVERLAP_TOKENS = 80
DEFAULT_TOP_K = 5

DEFAULT_EMBEDDING_PROVIDER = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PROMPT_TEMPLATE = "citation"
DEFAULT_GENERATION_PROVIDER = "api"
DEFAULT_GENERATION_MODEL: str | None = None
