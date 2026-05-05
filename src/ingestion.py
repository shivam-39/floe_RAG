"""Document ingestion for PDF, Markdown, and Jupyter notebook sources."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from models import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".ipynb"}


class UnsupportedDocumentTypeError(ValueError):
    """Raised when a file extension is not supported by the ingestion layer."""


def iter_document_paths(path: str | Path, recursive: bool = True) -> list[Path]:
    """Return supported document paths from a file or directory."""

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Document path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedDocumentTypeError(f"Unsupported document type: {root}")
        return [root]

    pattern = "**/*" if recursive else "*"
    return sorted(
        candidate
        for candidate in root.glob(pattern)
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_documents(paths: str | Path | Iterable[str | Path], recursive: bool = True) -> list[Document]:
    """Load all supported documents from one or many files/directories."""

    if isinstance(paths, (str, Path)):
        document_paths = iter_document_paths(paths, recursive=recursive)
    else:
        document_paths = []
        for path in paths:
            document_paths.extend(iter_document_paths(path, recursive=recursive))

    documents: list[Document] = []
    for path in document_paths:
        documents.extend(load_document(path))
    return documents


def load_document(path: str | Path) -> list[Document]:
    """Load a single supported document into one or more text-bearing records."""

    source_path = Path(path).expanduser().resolve()
    suffix = source_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(source_path)
    if suffix in {".md", ".markdown"}:
        return load_markdown(source_path)
    if suffix == ".ipynb":
        return load_notebook(source_path)

    raise UnsupportedDocumentTypeError(f"Unsupported document type: {source_path}")


def load_pdf(path: Path) -> list[Document]:
    """Extract one document record per PDF page."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf to ingest PDF files.") from exc

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        documents.append(
            Document(
                text=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "page": page_index,
                },
            )
        )
    return documents


def load_markdown(path: Path) -> list[Document]:
    """Load a Markdown document as a single record."""

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [
        Document(
            text=text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "page": None,
            },
        )
    ]


def load_notebook(path: Path) -> list[Document]:
    """Load Markdown and code cells from a Jupyter notebook."""

    try:
        import nbformat
    except ImportError as exc:
        raise ImportError("Install nbformat to ingest Jupyter notebooks.") from exc

    notebook = nbformat.read(path, as_version=4)
    documents: list[Document] = []

    for cell_index, cell in enumerate(notebook.cells):
        cell_type = cell.get("cell_type", "unknown")
        if cell_type not in {"markdown", "code"}:
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            text = "".join(source)
        else:
            text = str(source)

        if not text.strip():
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "page": None,
                    "cell_index": cell_index,
                    "cell_type": cell_type,
                },
            )
        )

    return documents
