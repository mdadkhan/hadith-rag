# hadith-rag — Local AI Search for Sahih Bukhari & Sahih Muslim

> Ask questions about Sahih Bukhari and Sahih Muslim in plain English and get cited answers — fully offline, no API key required.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5%2B-orange)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Unified documentation for `ingest_bukhari_v1_lc.py` and `ingest_muslim_v1_lc.py`.

---

## What Is This?

**hadith-rag** is a fully local Retrieval-Augmented Generation (RAG) pipeline for the canonical Hadith collections — Sahih Bukhari and Sahih Muslim. It parses bilingual Arabic-English PDFs, indexes every hadith individually with rich metadata (book, chapter, narrator, reference), and lets you query them using natural language with cited answers powered by a local LLM (Ollama).

No cloud API. No subscriptions. Everything runs on your machine.

---

## Features

- **Hadith-aware parsing** — detects book, chapter, and hadith boundaries from PDF structure; not generic chunking
- **Hybrid retrieval** — combines BM25 keyword search and vector similarity for high-recall results
- **Reciprocal Rank Fusion (RRF)** — merges keyword and semantic ranked lists into a single score
- **Cross-encoder re-ranking** — reranks top candidates with a dedicated relevance model before passing to LLM
- **Rich metadata per hadith** — volume, book number, book title, chapter, hadith number, narrator, page, reference string
- **Fully local** — embeddings via HuggingFace sentence-transformers, LLM via Ollama (llama3.2, mistral, etc.)
- **Cited answers** — LLM output always includes `[Book | Hadith reference | Page N]` citations
- **Incremental ingestion** — re-run without duplicating; stable ID scheme for deduplication
- **Source management** — delete one PDF's data without wiping the entire database
- **Interactive query mode** — REPL interface; models loaded once, reused for every question

---

## Example Output

```
$ python3 query.py "What did the Prophet say about cleanliness?"

Searching for: "What did the Prophet say about cleanliness?"
────────────────────────────────────────────────────
Top 5 chunks after hybrid search + re-ranking:

[1] sahih-al-bukhari-volume-1.pdf  |  Page 45  |  Vol.1, Bk.4, Ch.1, No.142
────────────────────────────────
Narrated Abu Malik Al-Ash'ari: The Prophet said, "Cleanliness is half of faith..."

[2] sahih-muslim-volume-1.pdf  |  Page 112  |  Vol.1, Bk.2, Ch.1, No.432
────────────────────────────────
...purity is half of faith...

============================================================
ANSWER
============================================================
The Prophet (ﷺ) said that cleanliness (purity) is half of faith
[sahih-al-bukhari-volume-1.pdf | Vol.1, Bk.4, Ch.1, No.142 | Page 45].
This is corroborated in Sahih Muslim where the same teaching appears
[sahih-muslim-volume-1.pdf | Vol.1, Bk.2, Ch.1, No.432 | Page 112].
```

---

## Overview

Both scripts ingest bilingual hadith PDFs (Arabic-English) into a persistent hybrid retrieval system:
- **Vector storage**: ChromaDB (semantic search via HuggingFaceEmbeddings)
- **Lexical storage**: BM25 corpus file (keyword-based retrieval)
- **Document model**: LangChain Document with rich metadata

Each hadith becomes one searchable unit with:
- Full text (Arabic + English)
- Metadata (volume, book, chapter, hadith number, page, narrator)
- Embeddings stored in ChromaDB

---

## Supported PDFs

### Bukhari (ingest_bukhari_v1_lc.py)
- **Location**: `data/bukhari/*.pdf`
- **Format**: Arabic-English bilingual typeset
- **Edition**: Dr. Muhammad Muhsin Khan translation
- **Example**: `sahih-al-bukhari-volume-1.pdf`
- **Structure**: Running headers with "N—THE BOOK OF XXXX [Arabic]"

### Muslim (ingest_muslim_v1_lc.py)
- **Location**: `data/muslim/*.pdf`
- **Format**: Arabic-English bilingual typeset
- **Edition**: Abdul Hamid Siddiqi translation
- **Example**: `sahih-muslim-volume-1.pdf`
- **Structure**: Running headers "The Book Of XXXX  \<page_num\>"

---

## PDF Structure Parsing

### Bukhari
```
Running page header   → "N—THE BOOK OF XXXX  [Arabic]  <page_num>"
Book intro section    → "N - THE BOOK OF XXXX" (full paragraph)
Chapter marker        → "(N) CHAPTER. [title text]"
Hadith boundary       → "N. Narrated <narrator_name>:"
Arabic text           → Inline with English (preserved)
```

### Muslim
```
Running page header   → "The Book Of XXXX  <tab>  <page_num>"
Book intro section    → "N. The Book Of XXXX" (once per book)
Chapter marker        → "Chapter N. Title text"
Hadith boundary       → "[globalN] localN - (variant) Text…"
Arabic text           → Font-encoded glyphs (stripped during ingest)
```

---

## Metadata Per Hadith

Both scripts store the same metadata structure (with field names):

| Field | Type | Example | Notes |
|-------|------|---------|-------|
| `volume` | int | 1 | Extracted from filename |
| `book_num` | int | 2 | Sequential book number in volume |
| `book_title` | str | "The Book of Belief" | From running header |
| `chapter_num` | int | 3 | Chapter within book |
| `chapter_title` | str | "Faith is a Practical Matter" | May be empty |
| `hadith_num` | int | 45 | Global sequential number |
| `hadith_local` | int | 12 | Local number within book (Muslim only) |
| `narrator` | str | "Abu Huraira" | Narrator name (Bukhari only) |
| `reference` | str | "Vol.1, Bk.2, Ch.3, No.45" | Formatted reference string |
| `page` | int | 42 | PDF page where hadith starts |
| `source` | str | "sahih-al-bukhari-volume-1.pdf" | Original PDF filename |

---

## Usage

### Ingest Bukhari

```bash
# Ingest all Bukhari PDFs from data/bukhari/
python3 ingest_bukhari_v1_lc.py

# Ingest specific volume only
python3 ingest_bukhari_v1_lc.py --file sahih-al-bukhari-volume-1.pdf

# Wipe database and reprocess
python3 ingest_bukhari_v1_lc.py --reset

# Parse only (no embedding or storage)
python3 ingest_bukhari_v1_lc.py --dry-run

# Dry-run a specific file
python3 ingest_bukhari_v1_lc.py --dry-run --file vol-1.pdf
```

### Ingest Muslim

```bash
# Ingest all Muslim volumes from data/muslim/
python3 ingest_muslim_v1_lc.py

# Ingest specific volume only
python3 ingest_muslim_v1_lc.py --file sahih-muslim-volume-1.pdf

# Wipe database and reprocess
python3 ingest_muslim_v1_lc.py --reset

# Parse only (no embedding or storage)
python3 ingest_muslim_v1_lc.py --dry-run

# Dry-run a specific file
python3 ingest_muslim_v1_lc.py --dry-run --file volume-1.pdf
```

---

## Code Flow (Both Scripts)

### 1. High-Level Architecture

```
CLI args (--reset, --file, --dry-run)
    ↓
run_pipeline()
    ├─ Resolve PDF list (--file or glob data/*/volume-*.pdf)
    ├─ Optional: wipe Chroma collection + BM25 file
    ├─ Load embedding model (HuggingFaceEmbeddings)
    ├─ Load existing BM25 corpus from disk
    │
    └─ For each PDF:
        ├─ _process_one(pdf, volume)
        │   ├─ extract_pages(pdf)              → raw page text + page numbers
        │   ├─ _extract_book_titles(pages)     → {book_num: "title"}
        │   ├─ _build_page_book_map(pages)     → {page: book_num}
        │   ├─ build_flat_text(pages)          → (flat_text, offsets)
        │   └─ parse_hadiths(flat_text, ...)   → list[Document]
        │
        └─ embed_and_store(documents)
            ├─ Create/open ChromaDB collection
            ├─ Batch dedupe IDs
            ├─ Generate embeddings via HuggingFaceEmbeddings
            ├─ Store in ChromaDB (vectors + docs + metadata)
            ├─ Append keyword rows to BM25 corpus
            └─ return stored_count
    │
    └─ save_bm25_corpus(filename)
    └─ Print final summary
```

### 2. Function-Level Call Flow

```
main
  ↓
run_pipeline(reset, dry_run)
  ├─ _volume_from_filename(filename)     # parse volume number
  ├─ _process_one(pdf_path, volume)
  │   ├─ extract_pages(pdf)
  │   ├─ _extract_book_titles(pages)
  │   ├─ _build_page_book_map(pages)
  │   ├─ build_flat_text(pages)
  │   │   └─ _clean_page_text(text)    # strip headers/noise
  │   └─ parse_hadiths(flat_text, ...)
  │       ├─ _build_chapter_index()
  │       ├─ _page_for_offset()        # map offset → page
  │       └─ _lookup()                 # resolve chapter metadata
  │
  ├─ embed_and_store(documents)
  ├─ load_bm25_corpus()
  └─ save_bm25_corpus()
```

### 3. Sequence Diagram (Runtime)

```
User/CLI
    │
    ├─ 1. Start with args (--reset, --file, --dry-run)
    │
run_pipeline
    ├─ 2. Resolve PDF list
    ├─ 3. Optional: reset Chroma + BM25 file
    ├─ 4. Load embedding model
    ├─ 5. Load existing BM25 corpus
    │
    └─ For each PDF:
        │
        _process_one
            ├─ 6. extract_pages()
            ├─ 7. _extract_book_titles()
            ├─ 8. _build_page_book_map()
            ├─ 9. build_flat_text()
            ├─ 10. parse_hadiths()
            │       → list[Document] with metadata
            │
            └─ 11. embed_and_store(documents)
                    │
                    embed_and_store
                        ├─ 12. Create/open ChromaDB collection
                        ├─ 13. Batch dedupe IDs
                        ├─ 14. Generate embeddings
                        ├─ 15. collection.add(ids, vectors, docs, metadata)
                        └─ 16. Append to BM25 corpus
                    │
                    ├─ 17. Return stored count
                    │
    │
    ├─ 18. Accumulate grand total
    └─ 19. save_bm25_corpus()
         └─ 20. Print summary paths
```

---

## Core Data Structures

### pages
- **Type**: `list[dict]`
- **Shape**: `[{"page": int, "text": str}, ...]`
- **Purpose**: Raw page text with explicit page numbers for downstream mapping

### offsets
- **Type**: `list[tuple[int, int]]`
- **Shape**: `[(flat_start_offset, page_number), ...]`
- **Purpose**: Maps hadith text offsets to original PDF pages (enables fast binary search)

### book_titles
- **Type**: `dict[int, str]`
- **Shape**: `{book_num: "The Book of ..."}`
- **Purpose**: Canonical human-readable book titles

### page_book_map
- **Type**: `dict[int, int]` (Bukhari) or `dict[int, tuple[int, str]]` (Muslim)
- **Shape**: `{page_num: book_num}` or `{page_num: (book_num, book_title)}`
- **Purpose**: Maps each page to its active book number using running headers

### chapter_map & chapter_starts
- **Type**: `dict[int, tuple[int, str]]` and `list[int]`
- **Purpose**: Resolves chapter metadata from nearest preceding chapter marker

### documents
- **Type**: `list[Document]` (LangChain)
- **Purpose**: One parsed hadith per document with retrieval metadata

### bm25_corpus
- **Type**: `list[dict]`
- **Shape per row**: `{id, text, source, page, reference}`
- **Purpose**: Lexical retrieval rows for BM25-style keyword ranking (separate from vectors)

### existing_ids
- **Type**: `set[str]`
- **Purpose**: Prevents duplicate insertion across batches and repeated runs

---

## Models and Algorithms

### Embedding Model
- **Library**: `HuggingFaceEmbeddings` from `langchain-community`
- **Config**: `model_name=EMBED_MODEL` with `normalize_embeddings=True`
- **Dimension**: 384-dimensional vectors (default sentence-transformers model)
- **Similarity**: Cosine distance

### Vector Storage Backend
- **Library**: ChromaDB persistent collection
- **Index type**: HNSW (Hierarchical Navigable Small World)
- **Distance metric**: `hnsw:space = cosine`

### Lexical Storage
- **Format**: JSON-serialized BM25 corpus rows
- **Fields per row**: `{id, text, source, page, reference}`
- **Purpose**: Enables keyword-based retrieval complementing vector search

### Text Segmentation
- **Algorithm**: Regex-based boundary detection
- **Hadith markers**:
  - **Bukhari**: `"N. Narrated <narrator>:"` (where N is number)
  - **Muslim**: `"[N]"` or number prefix patterns
- **Approach**: Stateless regex split (not semantic chunking)

### Metadata Lookup
- **Algorithm**: Binary search (`bisect_right`)
- **Applied to**: Sorted offsets and chapter markers
- **Speed**: O(log n) instead of O(n) linear scan

### Deduplication
- **Algorithm**: Set-based duplicate ID checks
- **Applied across**: Existing corpus and current batch
- **Format**: Stable IDs like `filename__vN_bNN_hNNNN`

---

## Storage Architecture (Hybrid Retrieval)

### Vector Side (Semantic)
- **Store**: ChromaDB collection under `CHROMA_DIR`
- **Contents**: Embeddings + full document text + metadata
- **Query**: Cosine similarity on 384-dim vectors
- **Best for**: Semantic meaning, paraphrases, conceptual questions

### Keyword Side (Lexical)
- **Store**: BM25 corpus file at `BM25_FILE`
- **Format**: JSON list of dicts
- **Contents**: Plain text, source, page, reference
- **Query**: Term frequency, substring matching
- **Best for**: Exact hadith numbers, narrator names, precise keywords

### Hybrid Query
Combine both during retrieval:
1. Fetch top-k results from vector similarity
2. Fetch top-m results from BM25 keywords
3. Merge and rank by combined score

---

## Document ID Format

Stable ID scheme ensures deduplication and traceability:

```
<filename>__v<volume>_b<book>_h<hadith>

Example (Bukhari):
  sahih-al-bukhari-volume-1.pdf__v1_b2_h45

Example (Muslim):
  sahih-muslim-volume-1.pdf__v1_b3_h234
```

---

## Output Artifacts

After successful ingestion:

### ChromaDB Collection
- **Location**: `CHROMA_DIR` (from config.py, usually `rag_knowledge_base/`)
- **Contents**: 
  - Vector embeddings (384-dim)
  - Document text per hadith
  - Metadata (volume, book, chapter, reference, page, source, etc.)
- **Queryable**: Via `Chroma` client with `.similarity_search(query, k=n)`

### BM25 Corpus File
- **Location**: `BM25_FILE` (from config.py, usually `bm25_corpus.json`)
- **Format**: JSON-serialized list of dicts
- **Each row**: `{id, text, source, page, reference}`
- **Queryable**: Via string search or BM25 ranking library

### Progress/Logs
- **Stdout**: Summary of processed PDFs, stored hadith counts, errors/skips

---

## Error Handling & Skip Paths

| Condition | Action | Message |
|-----------|--------|---------|
| Data directory missing | Exit with error | "data/bukhari/ not found" or "data/muslim/ not found" |
| No PDFs found | Exit with error | "No PDF files in data/bukhari/" |
| Volume parse fails | Skip file with warning | "Could not determine volume from filename" |
| No hadith boundaries | Skip file with warning | "No hadith boundaries found in {filename}" |
| Duplicate IDs | Skip silently | (logged internally, not added to collection) |
| PDF parse error (fitz) | Exception propagates | User sees detailed stack trace |

---

## Dependencies

### Required Python Packages
```
pymupdf>=1.27.0              # PDF text extraction
langchain>=0.1.0             # Document abstraction
langchain-community>=0.1.0   # HuggingFaceEmbeddings
langchain-chroma>=0.1.0      # ChromaDB wrapper
chromadb>=1.5.0              # Vector storage
sentence-transformers>=2.7.0 # Embedding model
transformers>=4.40.0         # Required by sentence-transformers
torch>=2.2.2                 # ML backend
```

### Installation
```bash
pip3 install -r requirements_enhanced.txt
# or individually:
pip3 install pymupdf langchain langchain-community langchain-chroma chromadb "sentence-transformers==2.7.0" "transformers==4.40.0"
```

---

## Configuration

All settings read from `config.py`:

```python
# Data directories
DATA_DIR              # Root for PDF folders
EMBED_MODEL          # HuggingFace model name (e.g., "all-MiniLM-L6-v2")
CHROMA_DIR           # Persistent ChromaDB location
BM25_FILE            # JSON file for lexical retrieval
COLLECTION           # ChromaDB collection name
```

Customize in `config.py` or `config.ini` before running ingestion scripts.

---

## Differences Between Bukhari and Muslim Scripts

| Aspect | Bukhari | Muslim |
|--------|---------|--------|
| **PDF location** | `data/bukhari/` | `data/muslim/` |
| **Narration marker** | "N. Narrated \<name\>:" | "[N] N - (variant)" |
| **Arabic handling** | Preserved inline | Stripped (garbled font) |
| **Chapter format** | "(N) CHAPTER." | "Chapter N." |
| **Metadata field** | `narrator` | `hadith_local` |
| **Header format** | "N—THE BOOK [Arabic]" | "The Book Of XXXX" |
| **Book intro** | "N - THE BOOK OF" | "N. The Book Of" |

Both normalize to the same metadata output; differences are internal parsing logic.

---

## Typical Workflow

1. **Place PDFs** in `data/bukhari/` or `data/muslim/`
2. **Run ingestion** with one command:
   ```bash
   python3 ingest_bukhari_v1_lc.py
   # or
   python3 ingest_muslim_v1_lc.py
   ```
3. **Query via retrieval** in your app:
   ```python
   from langchain_chroma import Chroma
   from langchain_community.embeddings import HuggingFaceEmbeddings
   
   embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
   vectorstore = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=CHROMA_DIR)
   results = vectorstore.similarity_search("Tell me about faith", k=5)
   ```

---

## Troubleshooting

### Issue: "No PDF files found"
- Check PDFs are in `data/bukhari/` or `data/muslim/`
- Ensure filenames contain volume number (e.g., `volume-1.pdf`)

### Issue: "Could not determine volume from filename"
- Filename must include a volume number pattern (e.g., `vol-1`, `volume-1`, `v1`)

### Issue: "No hadith boundaries found"
- PDF structure may not match expected format
- Check if narration markers (Bukhari: "N. Narrated", Muslim: "[N]") are present

### Issue: Embedding takes very long
- First run downloads embedding model (~1.5GB)
- Subsequent runs use cached model
- Use `--dry-run` to parse without embedding

### Issue: ChromaDB collection exists, want fresh start
- Use `--reset` flag to delete old collection and BM25 file:
  ```bash
  python3 ingest_bukhari_v1_lc.py --reset
  ```

---

## Relevant Files

| File | Role |
|------|------|
| `ingest_bukhari_v1_lc.py` | Ingest Sahih Bukhari volumes (this doc) |
| `ingest_muslim_v1_lc.py` | Ingest Sahih Muslim volumes (this doc) |
| `config.py` | Shared constants: paths, models, Ollama settings |
| `query.py` | Query pipeline: BM25 + vector → RRF → re-rank → Ollama LLM |
| `delete_source.py` | Remove a source PDF's documents from ChromaDB and BM25 index |
| `PIPELINE.md` | General PDF → Vector DB overview |
| `QUERY_CODE_FLOW.md` | How to query stored hadiths |
| `INGEST_BUKHARI_V1_CODE_FLOW.md` | Detailed mermaid diagrams for Bukhari v1 |

---

## Query Pipeline (query.py)

After ingestion, use `query.py` to ask questions against the stored hadith database.

### Usage

```bash
# Full pipeline: hybrid retrieval + Ollama LLM answer
python3 query.py "What did the Prophet say about honesty?"

# Show retrieved chunks only (skip LLM)
python3 query.py "What did the Prophet say about honesty?" --no-llm

# Interactive mode (ask multiple questions without reloading models)
python3 query.py -i
```

### Query Pipeline Steps

```
Question (user)
    ↓
Step 1 — Hybrid Retrieval
    ├─ BM25 keyword search (rank-bm25)
    │     └─ tokenize query → score entire corpus → top-k IDs
    ├─ Vector similarity search (ChromaDB + BGE embeddings)
    │     └─ encode query with BGE prefix → cosine search → top-k IDs
    └─ Reciprocal Rank Fusion (RRF)
          └─ merge ranked lists → unified score per ID

    ↓
Step 2 — Cross-Encoder Re-ranking
    └─ ms-marco-MiniLM cross-encoder scores (question, chunk) pairs
       → re-rank merged candidates → final top-k chunks

    ↓
Step 3 — LLM Answer Generation (Ollama)
    └─ Build context from top-k chunks with hadith references
    └─ Send to Ollama with strict RAG prompt
    └─ Return grounded answer with citations [Book | Hadith ref | Page N]
```

### Source Filtering

Disable specific books from retrieval without re-ingesting:
```python
# In config.py or config.ini
DISABLED_SOURCES = ["sahih-al-bukhari-volume-2.pdf"]  # excluded from BM25 + vector search
```

---

## Managing Sources (delete_source.py)

Use `delete_source.py` to remove a specific PDF's documents from both ChromaDB and the BM25 index without wiping the entire database.

### Usage

```bash
# List all sources currently stored in ChromaDB and BM25 index
python3 delete_source.py --list

# Preview what would be deleted (no changes made)
python3 delete_source.py --dry-run "sahih-al-bukhari-volume-1.pdf"

# Delete all documents for a specific Bukhari volume
python3 delete_source.py "sahih-al-bukhari-volume-1.pdf"

# Delete all documents for a specific Muslim volume
python3 delete_source.py "sahih-muslim-volume-1.pdf"
```

> **Note:** The source name must exactly match the `source` field stored in the DB (i.e. the original PDF filename). Use `--list` first to confirm the exact name.

### What It Deletes

For the given source filename, it removes:
- All matching documents from the **ChromaDB** collection (batched in groups of 500)
- All matching entries from the **BM25 corpus JSON** file

### When to Use

| Scenario | Command |
|----------|---------|
| Re-ingest a single volume after fixing the PDF | `delete_source.py "vol-1.pdf"` then re-run ingest |
| Check what's currently stored | `delete_source.py --list` |
| Verify deletion scope before committing | `delete_source.py --dry-run "vol-1.pdf"` |
| Partial reset (one volume only, not all) | `delete_source.py "vol-1.pdf"` (avoids `--reset` which wipes everything) |

---

## Ollama Integration

Ollama provides the local LLM for answer generation. No external API key required.

### Installation

```bash
# macOS
brew install ollama

# Or download from https://ollama.com
```

### Pull Models

```bash
# Default model used by query.py
ollama pull llama3.2

# Alternative models
ollama pull mistral
ollama pull phi3
ollama pull llama3.1
```

### Start Ollama Server

```bash
# Start in background (usually auto-started on macOS)
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### Configuration

Controlled via `config.py` or environment variables:

```bash
# Override default model
export OLLAMA_MODEL=mistral
python3 query.py "your question"

# Override host (e.g., remote Ollama server)
export OLLAMA_HOST=http://192.168.1.100:11434
python3 query.py "your question"
```

| Config | Default | Purpose |
|--------|---------|---------|
| `OLLAMA_MODEL` | `llama3.2` | Which model to use for answer generation |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

### RAG Prompt Template

`query.py` uses a strict RAG prompt to prevent hallucination:

```
System: You are a precise document assistant.
        Answer using CONTEXT blocks only.
        Do NOT use external knowledge.
        Cite every claim as [Book | Hadith reference | Page N].

User:   CONTEXT:
        [CONTEXT 1] Source: sahih-al-bukhari-volume-1.pdf | Page: 42 | Hadith: Vol.1, Bk.2, Ch.3, No.45
        <hadith text>
        ...

        QUESTION: <user question>
```

---

## See Also

- [PIPELINE.md](PIPELINE.md) — General PDF → Vector DB → LLM pipeline overview
- [QUERY_CODE_FLOW.md](QUERY_CODE_FLOW.md) — How to query stored hadiths
- [INGEST_BUKHARI_V1_CODE_FLOW.md](INGEST_BUKHARI_V1_CODE_FLOW.md) — Detailed mermaid diagrams for Bukhari v1 ingestion
- [config.py](config.py) — Tunable configuration

---

## GitHub Topics

If publishing to GitHub, add these topics to maximize discoverability:

```
hadith  quran  islamic-ai  rag  retrieval-augmented-generation  langchain
chromadb  ollama  bm25  hybrid-search  nlp  python  local-llm
sahih-bukhari  sahih-muslim  islamic-studies  pdf-parsing
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

See [LICENSE](LICENSE) for full terms.
