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