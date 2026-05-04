"""Prompt templates and citation formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from config import DEFAULT_PROMPT_TEMPLATE
from models import RetrievedChunk


@dataclass(frozen=True)
class PromptTemplate:
    """A reusable prompt template."""

    name: str
    template: str

    def render(self, question: str, context: str) -> str:
        return self.template.format(question=question, context=context).strip()


BASIC_QA_TEMPLATE = PromptTemplate(
    name="basic",
    template="""
You are a technical assistant answering questions from retrieved documentation.
Use only the context below. If the context is insufficient, say what is missing.

Context:
{context}

Question:
{question}

Answer:
""",
)


CITATION_QA_TEMPLATE = PromptTemplate(
    name="citation",
    template="""
You are a technical assistant answering questions from retrieved documentation.
Use only the context below. Cite every factual claim with the source label that
appears beside the relevant context. If the context is insufficient, say what is
missing and cite the closest available source.

Context:
{context}

Question:
{question}

Answer with citations:
""",
)


PROMPT_TEMPLATES = {
    BASIC_QA_TEMPLATE.name: BASIC_QA_TEMPLATE,
    CITATION_QA_TEMPLATE.name: CITATION_QA_TEMPLATE,
}


def get_prompt_template(name: str = DEFAULT_PROMPT_TEMPLATE) -> PromptTemplate:
    """Return a configured prompt template by name."""

    try:
        return PROMPT_TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown prompt template '{name}'. Available templates: {available}") from exc


def render_prompt(
    question: str,
    sources: Sequence[RetrievedChunk],
    template_name: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    """Render a prompt from a query and retrieved chunks."""

    template = get_prompt_template(template_name)
    return template.render(question=question, context=format_context(sources))


def format_context(sources: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks as model-ready context with source labels."""

    if not sources:
        return "No relevant context was retrieved."

    blocks: list[str] = []
    for index, retrieved in enumerate(sources, start=1):
        label = format_source_label(retrieved)
        blocks.append(f"{index}. {label}\n{retrieved.chunk.text}")
    return "\n\n".join(blocks)


def format_source_label(retrieved: RetrievedChunk) -> str:
    """Format a retrieved chunk citation label."""

    metadata = retrieved.chunk.metadata
    filename = str(metadata.get("filename") or metadata.get("source") or "unknown")
    page = metadata.get("page")
    cell_index = metadata.get("cell_index")

    if page not in {None, "", "None"}:
        return f"[Source: {filename}, page {page}]"
    if cell_index not in {None, "", "None"}:
        return f"[Source: {filename}, cell {cell_index}]"
    return f"[Source: {filename}]"


def format_source_list(sources: Sequence[RetrievedChunk]) -> str:
    """Format retrieved sources for user-facing display."""

    lines = []
    for retrieved in sources:
        label = format_source_label(retrieved)
        lines.append(f"{label} score={retrieved.score:.4f}")
    return "\n".join(lines)
