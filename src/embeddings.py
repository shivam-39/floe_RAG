"""Embedding model wrappers for document and query vectors."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PROVIDER


class EmbeddingModel(ABC):
    """Interface implemented by all embedding providers."""

    model_name: str

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into a normalized float32 matrix."""

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query into a normalized one-dimensional vector."""

        return self.embed_texts([query])[0]


class SentenceTransformerEmbeddings(EmbeddingModel):
    """Local Hugging Face sentence-transformer embeddings."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use local embeddings.") from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return normalize_vectors(embeddings)


class APIEmbeddingModel(EmbeddingModel):
    """Embedding provider for API-compatible embedding endpoints."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "MODEL_API_KEY",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use API-compatible embeddings.") from exc

        if not model_name:
            raise ValueError("model_name is required for API-compatible embeddings.")

        resolved_api_key = _resolve_api_key(api_key=api_key, api_key_env=api_key_env)
        self.model_name = model_name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._client = OpenAI(api_key=resolved_api_key, base_url=base_url)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = self._client.embeddings.create(model=self.model_name, input=texts)
        vectors = [item.embedding for item in response.data]
        return normalize_vectors(vectors)


def build_embedding_model(
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model_name: str | None = None,
    device: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str = "MODEL_API_KEY",
) -> EmbeddingModel:
    """Create an embedding provider from CLI/config values."""

    provider_key = provider.strip().lower()
    if provider_key in {"sentence-transformers", "sentence_transformers", "huggingface", "hf", "local"}:
        return SentenceTransformerEmbeddings(model_name or DEFAULT_EMBEDDING_MODEL, device=device)
    if provider_key in {"api", "api-compatible", "api_compatible", "openai-compatible", "openai_compatible", "openai"}:
        if not model_name:
            raise ValueError("An --embedding-model is required for API-compatible embeddings.")
        return APIEmbeddingModel(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


def normalize_vectors(vectors: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity via inner product search."""

    if isinstance(vectors, np.ndarray):
        matrix = vectors.astype(np.float32, copy=False)
    else:
        matrix = np.asarray(list(vectors), dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 1D or 2D numeric array.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _resolve_api_key(api_key: str | None, api_key_env: str) -> str:
    return api_key or os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY") or "unused"
