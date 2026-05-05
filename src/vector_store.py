"""FAISS vector store with persistent chunk metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from embeddings import EmbeddingModel, normalize_vectors
from models import Chunk, RetrievedChunk


INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"


class FaissVectorStore:
    """A cosine-similarity FAISS store backed by normalized vectors."""

    def __init__(self, index: Any, chunks: list[Chunk], embedding_model_name: str) -> None:
        self.index = index
        self.chunks = chunks
        self.embedding_model_name = embedding_model_name

    @classmethod
    def build(cls, chunks: list[Chunk], embedding_model: EmbeddingModel) -> "FaissVectorStore":
        """Embed chunks and build an in-memory FAISS index."""

        if not chunks:
            raise ValueError("Cannot build a vector store without chunks.")

        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_model.embed_texts(texts)
        return cls.from_embeddings(chunks, embeddings, embedding_model.model_name)

    @classmethod
    def from_embeddings(
        cls,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedding_model_name: str,
    ) -> "FaissVectorStore":
        """Build an index from precomputed embeddings."""

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install faiss-cpu to build or load a FAISS index.") from exc

        if len(chunks) == 0:
            raise ValueError("Cannot build a vector store without chunks.")

        matrix = normalize_vectors(embeddings)
        if matrix.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index=index, chunks=chunks, embedding_model_name=embedding_model_name)

    @classmethod
    def load(cls, index_dir: str | Path) -> "FaissVectorStore":
        """Load a persisted FAISS index and chunk metadata."""

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install faiss-cpu to load a FAISS index.") from exc

        root = Path(index_dir).expanduser().resolve()
        index_path = root / INDEX_FILENAME
        chunks_path = root / CHUNKS_FILENAME
        manifest_path = root / MANIFEST_FILENAME

        if not index_path.exists() or not chunks_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Vector store files are missing from {root}")

        index = faiss.read_index(str(index_path))
        chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        chunks = [
            Chunk(id=item["id"], text=item["text"], metadata=item.get("metadata", {}))
            for item in chunks_payload
        ]
        return cls(
            index=index,
            chunks=chunks,
            embedding_model_name=manifest["embedding_model_name"],
        )

    def save(self, index_dir: str | Path) -> None:
        """Persist FAISS index, chunks, and metadata manifest."""

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install faiss-cpu to save a FAISS index.") from exc

        root = Path(index_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(root / INDEX_FILENAME))
        (root / CHUNKS_FILENAME).write_text(
            json.dumps([chunk_to_dict(chunk) for chunk in self.chunks], indent=2),
            encoding="utf-8",
        )
        (root / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "embedding_model_name": self.embedding_model_name,
                    "chunk_count": len(self.chunks),
                    "dimension": self.index.d,
                    "index_type": "IndexFlatIP",
                    "similarity": "cosine",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def search_by_vector(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        """Search the index with an already embedded query vector."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query = normalize_vectors(query_embedding)
        if query.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension {query.shape[1]} does not match index dimension {self.index.d}."
            )
        scores, indices = self.index.search(query, min(top_k, len(self.chunks)))

        results: list[RetrievedChunk] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[int(index)], score=float(score)))
        return results

    def search(self, query: str, embedding_model: EmbeddingModel, top_k: int = 5) -> list[RetrievedChunk]:
        """Embed and retrieve the most relevant chunks for a query."""

        query_embedding = embedding_model.embed_query(query)
        return self.search_by_vector(query_embedding, top_k=top_k)


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Serialize a chunk for JSON persistence."""

    return {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
