# PDF → Vector Database → LLM Answer Pipeline

## Overview

This pipeline reads PDF files from the `data/` folder, extracts structured text, splits it into overlapping chunks, converts each chunk into a vector embedding, and stores everything in a local persistent vector database (ChromaDB). When you query the database, the most relevant chunks are retrieved and passed to a local **Ollama** LLM which generates a strictly-grounded answer — no external API keys required at any stage.

```
data/*.pdf
    ↓  Step 1: PyMuPDF
Structured text (per page)
    ↓  Step 2: Chunking (sentence or section strategy)
Overlapping text chunks (~500 chars)
    ↓  Step 3: sentence-transformers
384-dimensional vector embeddings
    ↓  Step 4: ChromaDB
vector_db/ (persistent local storage)
    ↓  Step 5: Ollama (strict RAG prompt)
Grounded natural-language answer

Support layer (runs alongside the pipeline):
  progress.json  ←  milestone checkpoints after each PDF batch
  →  resume / error-skip / final report
```

---

## Dependencies

| Library | Version Used | Purpose |
|---|---|---|
| `pymupdf` | 1.27.x | PDF text extraction |
| `sentence-transformers` | 2.7.0 | Local embedding model |
| `transformers` | 4.40.0 | Required by sentence-transformers |
| `chromadb` | 1.5.x | Local vector database |
| `torch` | 2.2.2 | ML backend for embeddings |
| `ollama` | 0.2.x | Python client for local Ollama LLM |

### Install Python packages

```bash
python3 -m pip install -r requirements.txt
# or individually:
python3 -m pip install pymupdf chromadb "sentence-transformers==2.7.0" "transformers==4.40.0" ollama
```

### Install Ollama and pull a model

```bash
# Install Ollama (macOS)
brew install ollama        # or download from https://ollama.com

# Pull the default model used by the pipeline
ollama pull llama3.2

# Optional alternatives
ollama pull mistral
ollama pull phi3
```

> **Note:** `torch>=2.4` is not available for Python 3.12 on macOS (as of April 2026). The pipeline is pinned to `sentence-transformers==2.7.0` + `transformers==4.40.0` which are compatible with `torch 2.2.2`.

---

## Project Structure

```
basic-AI/
├── data/                          # Drop PDF files here
│   └── Soorato Yaaseen October 31-2025.pdf
├── vector_db/                     # Auto-created — ChromaDB persistent storage
├── progress.json                  # Auto-created — batch milestone / resume tracker
├── config.ini                     # All tunable settings (model, chunking, batching)
├── pdf_pipeline.py                # Main pipeline script
├── requirements.txt               # Pinned dependencies
└── PIPELINE.md                    # This file
```

---

## Step-by-Step Explanation

### Step 1 — PDF → Structured Text (PyMuPDF)

**Function:** `extract_text_from_pdf(pdf_path)`

PyMuPDF (`fitz`) opens each PDF and reads it page by page. For each page, it extracts text blocks in layout order (top-to-bottom, left-to-right) using `page.get_text("blocks")`. This preserves document structure better than a plain full-page text dump.

- Only text blocks (`block_type == 0`) are kept; images and drawings are ignored
- Consecutive blank lines are collapsed to reduce noise
- Empty pages are skipped
- Each page produces a dict: `{page, text, source}`

**Output example:**
```python
{
    "page": 42,
    "text": "In the name of Allah, the Most Gracious...",
    "source": "Soorato Yaaseen October 31-2025.pdf"
}
```

---

### Step 2 — Text → Chunks

**Function:** `chunk_text(page_data)`

Raw page text is too long to embed as a single unit (embedding models have token limits and lose context with very long inputs). Each page is split into smaller, overlapping chunks.

**Algorithm:**
1. Split page text into sentences using regex (`(?<=[.!?])\s+`)
2. Accumulate sentences until the chunk reaches ~500 characters
3. When the limit is hit, save the current chunk and start a new one — but carry the last 100 characters forward (the **overlap**) so context isn't lost at boundaries
4. If a single sentence exceeds 500 characters, it is hard-split by character count

**Settings (configurable at top of script):**

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Target characters per chunk |
| `CHUNK_OVERLAP` | 100 | Characters carried over to next chunk |

**Output example:**
```python
{
    "text": "In the name of Allah...",
    "source": "Soorato Yaaseen October 31-2025.pdf",
    "page": 42,
    "chunk_idx": 3
}
```

---

### Step 3 — Text → Embeddings (sentence-transformers)

**Model:** `all-MiniLM-L6-v2`

Each chunk's text is converted into a 384-dimensional float vector using a pre-trained transformer model. The model runs **entirely locally** — no API key or internet connection required after the first download (~90 MB, cached in `~/.cache/huggingface/`).

- The model is downloaded once from HuggingFace on first run
- Encoding is done in batches of 32 chunks at a time for memory efficiency
- Semantically similar texts produce vectors that are close together in 384-dimensional space (measured by cosine similarity)

**Why `all-MiniLM-L6-v2`?**
- Fast: encodes ~14,000 sentences/second on CPU
- Small: ~90 MB model size
- Good quality for semantic search tasks

---

### Step 4 — Store in Vector DB (ChromaDB)

**Function:** `embed_and_store(chunks, collection, model)`

ChromaDB is a local, persistent vector database. It stores each chunk's:
- **Embedding** (the 384-dim vector) — used for similarity search
- **Document text** — returned in query results
- **Metadata** — source filename, page number, chunk index

Because ChromaDB has a maximum batch upsert size of ~5,461 records, chunks are uploaded in batches of 500 to stay well within the limit.

**Storage:**
- Persistent to `vector_db/` directory alongside the script
- Survives between runs — reprocessing is only needed when PDFs change
- Collection name: `pdf_documents`
- Distance metric: cosine similarity

**Chunk ID format:** `{filename}__p{page}__c{chunk_idx}`
Example: `Soorato Yaaseen October 31-2025.pdf__p42__c3`

---

### Query — Semantic Search

**Function:** `query_db(question, collection, model)`

When you query the database:
1. Your question is embedded into a 384-dim vector using the same model
2. ChromaDB finds the top-K stored chunks whose embeddings are closest (by cosine similarity) to the query vector
3. Results are returned with source file, page number, similarity score, and the matched text

Similarity score ranges from 0 (no relation) to 1.0 (identical).

When using the `section` chunking strategy, results also show the **section heading** the chunk belongs to:
```
[1] Source: doc.pdf  Page: 5  Similarity: 0.921  Section: INTRODUCTION
```

---

### Step 5 — Strict RAG Answer Generation (Ollama)

**Function:** `generate_answer(question, query_results, model)`

The top retrieved chunks are assembled into a numbered context block and forwarded to a local Ollama LLM with a strict system prompt. The LLM is instructed to:

1. Answer **only** from the provided context — no external knowledge
2. Return a fixed fallback phrase if the answer is not present in the documents
3. Quote or paraphrase the source material directly
4. Stay concise and factual
5. Use `temperature=0.0` for fully deterministic, reproducible output

**Context block format sent to the LLM:**
```
[CONTEXT 1] Source: document.pdf | Page: 42 | Similarity: 0.912
<chunk text>

[CONTEXT 2] Source: document.pdf | Page: 43 | Similarity: 0.887
<chunk text>
...

QUESTION: <your question>

Answer strictly based on the context above:
```

**Settings (configurable at top of script):**

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Ollama model used for answering |
| `OLLAMA_TOP_K` | `5` | Number of retrieved chunks fed to the LLM |

---

## Usage

```bash
# Process all PDFs in data/ and store in vector_db/
python3 pdf_pipeline.py

# Process PDFs in batches of 5 (saves checkpoint after every batch)
python3 pdf_pipeline.py --pdf-batch-size 5

# Resume an interrupted run (skips already-done files automatically)
python3 pdf_pipeline.py --pdf-batch-size 5

# Process PDFs then immediately run a query (retrieves chunks + Ollama answer)
python3 pdf_pipeline.py --query "how to do wudhu"

# Query the existing vector DB without reprocessing
python3 pdf_pipeline.py --query-only "what is the meaning of Yaaseen"

# Query using a different Ollama model
python3 pdf_pipeline.py --query-only "what is the meaning of Yaaseen" --model mistral

# Skip Ollama — show only raw retrieved chunks (no LLM answer)
python3 pdf_pipeline.py --query-only "what is the meaning of Yaaseen" --no-llm

# Wipe vector DB + progress.json and reprocess all PDFs from scratch
python3 pdf_pipeline.py --reset
```

### CLI flags

| Flag | Description |
|---|---|
| `--query "..."` | Process PDFs then query |
| `--query-only "..."` | Query existing DB only (skip processing) |
| `--reset` | Delete vector DB + progress file, reprocess all PDFs |
| `--model <name>` | Ollama model to use (default: `llama3.2`) |
| `--no-llm` | Skip Ollama; display raw retrieved chunks only |
| `--pdf-batch-size <n>` | PDFs per batch before saving checkpoint (0 = all at once) |

---

## Batching, Error Handling & Resume

### PDF Batching

When processing many PDFs, you can split the work into batches. After each batch completes, a **milestone checkpoint** is written to `progress.json`. If the run is interrupted, the next run automatically picks up from the last checkpoint.

```
20 PDFs ÷ batch_size=5  →  Batch 1 (5 files) → save  →  Batch 2 (5 files) → save  → ...
```

Set `pdf_batch_size` in `config.ini` or pass `--pdf-batch-size` on the command line.

### Error Handling

Each PDF is processed inside a `try/except` block:
- If extraction, chunking, or storing raises **any exception**, the error is logged to `progress.json` with the message
- The pipeline **skips** that file and continues with the next one
- Failed files are **retried** automatically on the next run (their status is `"failed"`, not `"done"`)
- Only `"done"` files are skipped on resume

### Resume

`progress.json` records each file’s outcome:

```json
{
  "quran.pdf": {
    "status":    "done",
    "pages":     2714,
    "chunks":    23421,
    "timestamp": "2026-04-19T10:00:00+00:00"
  },
  "broken.pdf": {
    "status":    "failed",
    "error":     "cannot open broken PDF: no objects found",
    "timestamp": "2026-04-19T10:01:00+00:00"
  }
}
```

On the next run:
- `quran.pdf` → **skipped** (already done)
- `broken.pdf` → **retried**
- Any new PDF added to `data/` → **processed**

Use `--reset` to wipe both `progress.json` and the ChromaDB collection and reprocess everything from scratch.

### Final Report

Printed at the end of every run:

```
============================================================
                    PROCESSING REPORT
============================================================

[OK] Successfully processed: 19/20
     quran.pdf
       pages=2714  chunks=23421  @ 2026-04-19T10:00:00+00:00

[FAIL] Failed: 1/20
     broken.pdf
       error: cannot open broken PDF: no objects found

[PENDING] Not yet processed: 0/20
============================================================
```

**Three categories:**

| Category | Meaning |
|---|---|
| `[OK]` | File fully processed and stored in ChromaDB |
| `[FAIL]` | Exception occurred — file skipped, will retry next run |
| `[PENDING]` | File exists in `data/` but has not been attempted yet (run interrupted mid-batch) |

---

## Results from Current PDF

| Metric | Value |
|---|---|
| PDF processed | `Soorato Yaaseen October 31-2025.pdf` |
| Pages extracted | 2,714 |
| Chunks created | 23,421 |
| Chunks stored in vector DB | 23,421 |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector dimensions | 384 |
| Distance metric | Cosine similarity |

---

## Troubleshooting

### `Import "torch" could not be resolved` (Pylance)
VS Code is using a different Python interpreter than the one where torch is installed.
Fix: **Cmd+Shift+P → Python: Select Interpreter** → choose `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`

### `sentence_transformers` import error with `nn` not defined
The installed version of `sentence-transformers` requires `torch>=2.4` which is not available for this setup.
Fix: pin to compatible versions:
```bash
python3 -m pip install "sentence-transformers==2.7.0" "transformers==4.40.0"
```

### `Batch size of X is greater than max batch size of 5461` (ChromaDB)
The pipeline now automatically batches upserts in groups of 500. If you see this error, ensure you are running the latest version of `pdf_pipeline.py`.

### No results or low similarity scores
- The PDF may contain scanned images rather than selectable text. PyMuPDF cannot extract text from image-based PDFs without OCR.
- Try rephrasing the query to match the language/terminology used in the document.

### A PDF keeps failing on every run
- Check the error message printed during the run or in `progress.json`.
- The file may be password-protected, corrupted, or image-only (no extractable text).
- Remove the file from `data/` or fix it, then run again — the rest of the files will not be re-processed.

### Run was interrupted mid-batch
- Simply run the same command again. `progress.json` records which files completed.
- Files in incomplete batches (status `"pending"` / absent from the file) will be retried.
- Use `--reset` only if you want to start completely from scratch.

### `[Ollama error] model not found`
The model has not been pulled yet. Run:
```bash
ollama pull llama3.2
```
Or specify a model you already have with `--model <name>`.

### `[Ollama error] connection refused` / Ollama server not running
Start the Ollama server before running the pipeline:
```bash
ollama serve   # or open the Ollama.app from Applications
```

### Ollama gives answers outside the documents
- Ensure you are using the latest `pdf_pipeline.py` — the strict system prompt and `temperature=0.0` are required.
- Avoid models smaller than 3B parameters; they tend to ignore system prompt constraints.
