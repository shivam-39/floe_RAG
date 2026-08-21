"""Run a BEIR SciFact retrieval benchmark against the RAG vector store."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from chunking import chunk_documents
from config import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    DEFAULT_EMBEDDING_MODEL,
)
from embeddings import build_embedding_model
from models import Document, RetrievedChunk
from vector_store import FaissVectorStore

DATASET_NAME = "scifact"
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DEFAULT_DATASET_DIR = Path("benchmark_data") / DATASET_NAME
DEFAULT_INDEX_DIR = Path("benchmark_index") / DATASET_NAME
DEFAULT_RESULTS_PATH = Path("benchmark_results") / f"{DATASET_NAME}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dense retrieval with BEIR SciFact.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE_TOKENS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--k-values", default="1,5,10", help="Comma-separated cutoff values, for example 1,5,10.")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args(argv)


def ensure_dataset(dataset_dir: Path) -> Path:
    """Download and extract SciFact when the expected files are unavailable."""

    root = dataset_dir.expanduser().resolve()
    if _find_dataset_root(root) is not None:
        return _find_dataset_root(root)  # type: ignore[return-value]

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "scifact.zip"
    print(f"Downloading {DATASET_NAME} to {archive_path}")
    urllib.request.urlretrieve(DATASET_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)
    archive_path.unlink()

    dataset_root = _find_dataset_root(root)
    if dataset_root is None:
        raise FileNotFoundError(f"SciFact files were not found after extracting {root}")
    return dataset_root


def load_beir_dataset(dataset_root: Path, split: str) -> tuple[list[Document], dict[str, str], dict[str, set[str]]]:
    """Load BEIR corpus, queries, and qrels into repository-compatible structures."""

    corpus_path = dataset_root / "corpus.jsonl"
    queries_path = dataset_root / "queries.jsonl"
    qrels_path = dataset_root / "qrels" / f"{split}.tsv"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing BEIR file: {path}")

    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            record = json.loads(line)
            document_id = str(record["_id"])
            title = str(record.get("title", "")).strip()
            text = str(record.get("text", "")).strip()
            combined_text = f"{title}\n{text}" if title else text
            if combined_text.strip():
                documents.append(
                    Document(
                        text=combined_text,
                        metadata={"source": f"beir:{DATASET_NAME}", "beir_document_id": document_id},
                    )
                )

    queries: dict[str, str] = {}
    with queries_path.open(encoding="utf-8") as queries_file:
        for line in queries_file:
            record = json.loads(line)
            queries[str(record["_id"])] = str(record.get("text", "")).strip()

    relevant_documents: dict[str, set[str]] = defaultdict(set)
    with qrels_path.open(encoding="utf-8") as qrels_file:
        next(qrels_file, None)
        for line in qrels_file:
            query_id, document_id, relevance = line.rstrip("\n").split("\t")
            if int(relevance) > 0 and query_id in queries:
                relevant_documents[query_id].add(document_id)

    return documents, queries, dict(relevant_documents)


def evaluate_retrieval(
    vector_store: FaissVectorStore,
    embedding_model: Any,
    queries: dict[str, str],
    relevant_documents: dict[str, set[str]],
    k_values: list[int],
) -> dict[str, Any]:
    """Evaluate ranked source documents for each query at multiple cutoffs."""

    max_k = max(k_values)
    per_query: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {}

    for k in k_values:
        aggregate[str(k)] = {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    evaluated_queries = 0
    for query_id, query in queries.items():
        relevant = relevant_documents.get(query_id, set())
        if not relevant or not query:
            continue
        retrieved = vector_store.search(query, embedding_model, top_k=max_k)
        ranked_documents = _unique_document_ids(retrieved)
        query_metrics: dict[str, Any] = {"query_id": query_id, "query": query, "relevant_document_ids": sorted(relevant)}
        query_metrics["retrieved_document_ids"] = ranked_documents

        for k in k_values:
            metrics = _ranking_metrics(ranked_documents, relevant, k)
            query_metrics[str(k)] = metrics
            for name, value in metrics.items():
                aggregate[str(k)][name] += value
        per_query.append(query_metrics)
        evaluated_queries += 1

    for values in aggregate.values():
        for name in values:
            values[name] = values[name] / evaluated_queries if evaluated_queries else 0.0

    return {"query_count": evaluated_queries, "metrics": aggregate, "queries": per_query}


def build_or_load_index(
    dataset_root: Path,
    index_dir: Path,
    embedding_model: Any,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool,
) -> FaissVectorStore:
    """Build a benchmark index once, or load the persisted equivalent."""

    index_path = index_dir / "index.faiss"
    if index_path.exists() and not rebuild:
        return FaissVectorStore.load(index_dir)

    documents, _, _ = load_beir_dataset(dataset_root, split="test")
    chunks = chunk_documents(documents, chunk_size_tokens=chunk_size, chunk_overlap_tokens=chunk_overlap)
    store = FaissVectorStore.build(chunks, embedding_model)
    store.save(index_dir)
    return store


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    k_values = _parse_k_values(args.k_values)
    dataset_root = ensure_dataset(args.dataset_dir)
    embedding_model = build_embedding_model(model_name=args.embedding_model, device=args.embedding_device)
    vector_store = build_or_load_index(
        dataset_root,
        args.index_dir,
        embedding_model,
        args.chunk_size,
        args.chunk_overlap,
        args.rebuild_index,
    )
    _, queries, relevant_documents = load_beir_dataset(dataset_root, args.split)
    evaluation = evaluate_retrieval(vector_store, embedding_model, queries, relevant_documents, k_values)
    payload = {
        "dataset": DATASET_NAME,
        "split": args.split,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "k_values": k_values,
        **evaluation,
    }
    results_path = args.results_path.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Saved benchmark results to {results_path}")
    return 0


def _find_dataset_root(root: Path) -> Path | None:
    candidates = [root, root / DATASET_NAME]
    return next((candidate for candidate in candidates if (candidate / "corpus.jsonl").exists()), None)


def _unique_document_ids(retrieved: list[RetrievedChunk]) -> list[str]:
    document_ids: list[str] = []
    seen: set[str] = set()
    for item in retrieved:
        document_id = str(item.chunk.metadata.get("beir_document_id", ""))
        if document_id and document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)
    return document_ids


def _ranking_metrics(ranked_documents: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top_documents = ranked_documents[:k]
    hits = [document_id in relevant for document_id in top_documents]
    hit_count = sum(hits)
    precision = hit_count / k
    recall = hit_count / len(relevant)
    reciprocal_rank = next((1.0 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0)
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    dcg = sum(1.0 / math.log2(index + 2) for index, hit in enumerate(hits) if hit)
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {"precision_at_k": precision, "recall_at_k": recall, "mrr": reciprocal_rank, "ndcg_at_k": ndcg}


def _parse_k_values(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item <= 0 for item in values):
        raise ValueError("--k-values must contain positive integers.")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
