"""Command-line interface for indexing documents and querying the RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chunking import chunk_documents
from config import (
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_INDEX_DIR,
    DEFAULT_GENERATION_PROVIDER,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TOP_K,
)
from embeddings import build_embedding_model
from ingestion import load_documents
from prompts import PROMPT_TEMPLATES, format_source_list
from rag_pipeline import RagPipeline, build_generation_model
from vector_store import FaissVectorStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query a technical-document RAG index.")
    parser.add_argument("--index", action="store_true", help="Build and persist a FAISS vector index.")
    parser.add_argument("--query", type=str, help="Question to answer using the vector index.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Document file or directory to index.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="Directory for FAISS index files.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively load supported files under --data-dir.",
    )

    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE_TOKENS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding provider: sentence-transformers, api, or api-compatible.",
    )
    parser.add_argument("--embedding-model", default=None, help="Embedding model name override.")
    parser.add_argument("--embedding-device", default=None, help="Optional device for sentence-transformers.")
    parser.add_argument("--embedding-base-url", default=None, help="Base URL for API-compatible embeddings.")
    parser.add_argument("--embedding-api-key", default=None, help="API key for API-compatible embeddings.")
    parser.add_argument(
        "--embedding-api-key-env",
        default="MODEL_API_KEY",
        help="Environment variable that stores the embedding API key.",
    )
    parser.add_argument(
        "--allow-embedding-mismatch",
        action="store_true",
        help="Allow querying an index with a different embedding model than the one used to build it.",
    )

    parser.add_argument(
        "--generation-provider",
        dest="generation_provider",
        default=DEFAULT_GENERATION_PROVIDER,
        help="Generation provider: api/api-compatible or local.",
    )
    parser.add_argument(
        "--generation-model",
        dest="generation_model",
        default=None,
        help="Generation model name.",
    )
    parser.add_argument("--generation-base-url", default=None, help="Base URL for API-compatible generation.")
    parser.add_argument("--generation-api-key", default=None, help="API key for API-compatible generation.")
    parser.add_argument(
        "--generation-api-key-env",
        default="MODEL_API_KEY",
        help="Environment variable that stores the generation API key.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        choices=sorted(PROMPT_TEMPLATES),
        help="Prompt template to use for generation.",
    )

    args = parser.parse_args(argv)
    if not args.index and not args.query:
        parser.error("Specify --index, --query, or both.")
    return args


def build_index(args: argparse.Namespace) -> FaissVectorStore:
    documents = load_documents(args.data_dir, recursive=args.recursive)
    chunks = chunk_documents(
        documents,
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"No chunks were created from {args.data_dir}")

    embedding_model = build_embedding_model(
        provider=args.embedding_provider,
        model_name=args.embedding_model,
        device=args.embedding_device,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        api_key_env=args.embedding_api_key_env,
    )
    vector_store = FaissVectorStore.build(chunks, embedding_model)
    vector_store.save(args.index_dir)
    print(f"Indexed {len(documents)} document records into {len(chunks)} chunks.")
    print(f"Saved vector store to {args.index_dir}")
    return vector_store


def run_query(args: argparse.Namespace, vector_store: FaissVectorStore | None = None) -> None:
    embedding_model = build_embedding_model(
        provider=args.embedding_provider,
        model_name=args.embedding_model,
        device=args.embedding_device,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        api_key_env=args.embedding_api_key_env,
    )
    store = vector_store or FaissVectorStore.load(args.index_dir)
    _validate_embedding_model(store, embedding_model.model_name, args.allow_embedding_mismatch)

    generation_model = build_generation_model(
        provider=args.generation_provider,
        model_name=args.generation_model,
        temperature=args.temperature,
        base_url=args.generation_base_url,
        api_key=args.generation_api_key,
        api_key_env=args.generation_api_key_env,
    )
    pipeline = RagPipeline(
        vector_store=store,
        embedding_model=embedding_model,
        generation_model=generation_model,
        prompt_template=args.prompt_template,
        top_k=args.top_k,
    )
    result = pipeline.answer(args.query)

    print("\nAnswer")
    print("------")
    print(result.answer)

    print("\nSources")
    print("-------")
    print(format_source_list(result.sources) or "No sources retrieved.")


def _validate_embedding_model(
    vector_store: FaissVectorStore,
    query_model_name: str,
    allow_mismatch: bool,
) -> None:
    if allow_mismatch or vector_store.embedding_model_name == query_model_name:
        return

    raise ValueError(
        "Embedding model mismatch: index was built with "
        f"'{vector_store.embedding_model_name}', but query uses '{query_model_name}'. "
        "Rebuild the index, pass the matching --embedding-model, or use --allow-embedding-mismatch."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        vector_store = build_index(args) if args.index else None
        if args.query:
            run_query(args, vector_store=vector_store)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
