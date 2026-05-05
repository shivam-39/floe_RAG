# RAGflow

Retrieval-Augmented Generation (RAG) for technical documents. It ingests PDFs, Markdown files, and Jupyter notebooks; chunks text with citation metadata; embeds chunks; stores vectors in FAISS; retrieves relevant context; and generates cited answers with a configurable LLM.

## Architecture

```mermaid
flowchart LR
    A[PDF / Markdown / Notebook] --> B[Ingestion]
    B --> C[Token Chunking]
    C --> D[Embedding Model]
    D --> E[FAISS Vector Store]
    Q[User Query] --> F[Query Embedding]
    F --> E
    E --> G[Top-k Chunks]
    G --> H[Prompt Template]
    H --> I[LLM]
    I --> J[Answer + Citations]
```

## Project Structure

```text
data/                 Raw technical documents
notebooks/            Experiments and evaluation notebooks
src/
  ingestion.py        PDF, Markdown, and notebook loading
  chunking.py         Fixed token chunking with overlap
  embeddings.py       Local and API-compatible embedding wrappers
  vector_store.py     FAISS indexing, persistence, and retrieval
  prompts.py          Basic QA and citation-focused prompt templates
  rag_pipeline.py     End-to-end retrieval and generation pipeline
  evaluation.py       precision@k, recall@k, and qualitative review helpers
  main.py             CLI for indexing and querying
requirements.txt      Python dependencies
```

## Setup

Use the existing conda environment named `rag_env`.

```bash
conda create --name rag_env
conda activate rag_env
pip install -r requirements.txt
```

For API-backed embeddings or generation, set the relevant API key if your endpoint requires one:

```bash
export MODEL_API_KEY="your-api-key"
```

The default embedding provider is local `sentence-transformers/all-MiniLM-L6-v2`. Generation is provider-neutral: use `--generation-provider api` for chat-completions-compatible APIs, or `--generation-provider local` for Hugging Face text-generation models.

## Usage

Put source documents under `data/`, then build a FAISS index:

```bash
python src/main.py --index --data-dir data --index-dir index_store
```

Query the index:

```bash
python src/main.py --query "How does the system normalize embeddings?" --index-dir index_store
```

Build and query in one run:

```bash
python src/main.py \
  --index \
  --data-dir data \
  --query "What are the main components of the pipeline?"
```

Use API-compatible embeddings:

```bash
python src/main.py \
  --index \
  --embedding-provider api \
  --embedding-model text-embedding-3-small
```

Use an API-compatible hosted chat endpoint:

```bash
python src/main.py \
  --query "Where is FAISS used?" \
  --generation-provider api \
  --generation-model gpt-4o-mini \
  --generation-api-key-env MODEL_API_KEY
```

Use a local API-compatible endpoint such as Ollama, LM Studio, or vLLM:

```bash
python src/main.py \
  --query "Where is FAISS used?" \
  --generation-provider api \
  --generation-model llama3.1 \
  --generation-base-url http://localhost:11434/v1
```

Use a local Hugging Face generation model:

```bash
python src/main.py \
  --query "Where is FAISS used?" \
  --generation-provider local \
  --generation-model google/flan-t5-base
```

Use the citation-focused prompt, which is the default:

```bash
python src/main.py \
  --query "Where is FAISS used?" \
  --prompt-template citation
```

## Pipeline Notes

Ingestion returns `Document` records with provenance such as filename, extension, page number for PDFs, and notebook cell metadata. Chunking creates overlapping fixed-size token windows and carries that metadata into each `Chunk`.

Embeddings are L2-normalized before storage. The FAISS store uses `IndexFlatIP`, so normalized inner product search behaves as cosine similarity. The persisted vector store contains:

```text
index_store/
  index.faiss
  chunks.json
  manifest.json
```

At query time, the pipeline embeds the query with the same embedding model, retrieves top-k chunks, renders a prompt, calls the selected LLM, and returns a `RagResult` with:

```text
answer
sources
prompt
```

Source labels are formatted as:

```text
[Source: filename, page X]
[Source: notebook.ipynb, cell Y]
```

## Evaluation

`src/evaluation.py` provides retrieval metrics and qualitative logging:

```python
from evaluation import RetrievalExample, evaluate_retrieval, mean_metrics

examples = [
    RetrievalExample(
        query="How are embeddings normalized?",
        relevant_chunk_ids={"chunk-id-1"},
    )
]

results = evaluate_retrieval(
    examples=examples,
    retrieve=lambda query, k: vector_store.search(query, embedding_model, top_k=k),
    k=5,
)

print(mean_metrics(results))
```

Available metrics:

- `precision_at_k`
- `recall_at_k`
- mean precision and recall across an evaluation set
- basic qualitative records that track answer text, retrieved source labels, citation markers, and reviewer notes

## Prompt Tuning

Two templates are available in `src/prompts.py`:

- `basic`: concise QA over retrieved context
- `citation`: citation-focused QA that asks the model to cite factual claims

Swap templates with `--prompt-template basic` or by passing `prompt_template="basic"` to `RagPipeline`.
