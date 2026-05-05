"""Evaluation helpers for retrieval quality and qualitative RAG review."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from models import RetrievedChunk
from prompts import format_source_label


@dataclass(frozen=True)
class RetrievalExample:
    """A query with known relevant chunk IDs."""

    query: str
    relevant_chunk_ids: set[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    """Retrieval metrics for one evaluation query."""

    query: str
    precision_at_k: float
    recall_at_k: float
    retrieved_ids: list[str]
    relevant_ids: list[str]


@dataclass(frozen=True)
class QualitativeEvaluation:
    """Lightweight qualitative review record for a generated answer."""

    query: str
    answer: str
    source_count: int
    contains_citation_marker: bool
    source_labels: list[str]
    notes: list[str]


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute precision@k as relevant retrieved results divided by k."""

    _validate_k(k)
    if not retrieved_ids:
        return 0.0

    top_ids = list(retrieved_ids[:k])
    hits = sum(1 for chunk_id in top_ids if chunk_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute recall@k as relevant retrieved results divided by known relevant results."""

    _validate_k(k)
    if not relevant_ids:
        return 0.0

    top_ids = set(retrieved_ids[:k])
    hits = len(top_ids.intersection(relevant_ids))
    return hits / len(relevant_ids)


def evaluate_retrieval(
    examples: Iterable[RetrievalExample],
    retrieve: Callable[[str, int], Sequence[RetrievedChunk]],
    k: int,
) -> list[RetrievalMetrics]:
    """Evaluate a retrieval callable over labeled examples."""

    _validate_k(k)
    results: list[RetrievalMetrics] = []

    for example in examples:
        retrieved = retrieve(example.query, k)
        retrieved_ids = [item.chunk.id for item in retrieved]
        results.append(
            RetrievalMetrics(
                query=example.query,
                precision_at_k=precision_at_k(retrieved_ids, example.relevant_chunk_ids, k),
                recall_at_k=recall_at_k(retrieved_ids, example.relevant_chunk_ids, k),
                retrieved_ids=retrieved_ids[:k],
                relevant_ids=sorted(example.relevant_chunk_ids),
            )
        )

    return results


def mean_metrics(results: Sequence[RetrievalMetrics]) -> dict[str, float]:
    """Average precision@k and recall@k over retrieval results."""

    if not results:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}

    return {
        "precision_at_k": sum(result.precision_at_k for result in results) / len(results),
        "recall_at_k": sum(result.recall_at_k for result in results) / len(results),
    }


def qualitative_evaluation(
    query: str,
    answer: str,
    sources: Sequence[RetrievedChunk],
    notes: Sequence[str] | None = None,
) -> QualitativeEvaluation:
    """Create a basic qualitative evaluation record for a RAG answer."""

    return QualitativeEvaluation(
        query=query,
        answer=answer,
        source_count=len(sources),
        contains_citation_marker="[Source:" in answer,
        source_labels=[format_source_label(source) for source in sources],
        notes=list(notes or []),
    )


def write_evaluation_results(
    retrieval_results: Sequence[RetrievalMetrics],
    output_path: str | Path,
    qualitative_results: Sequence[QualitativeEvaluation] | None = None,
) -> None:
    """Write retrieval and optional qualitative evaluation records to JSON."""

    payload = {
        "summary": mean_metrics(retrieval_results),
        "retrieval_results": [asdict(result) for result in retrieval_results],
        "qualitative_results": [asdict(result) for result in qualitative_results or []],
    }

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive.")
