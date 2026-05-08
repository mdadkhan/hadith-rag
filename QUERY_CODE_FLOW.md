# query.py Sequence Diagram and Code Flow

This document explains how `query.py` executes from CLI input to retrieved chunks and final answer generation.

## 1) End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User/CLI
    participant Q as query.py
    participant C as Config + DB Utils
    participant V as Chroma Vector DB
    participant B as BM25 Corpus (JSON)
    participant E as Embedding Model
    participant R as Cross-Encoder
    participant O as Ollama

    U->>Q: 1. Run command with question or -i
    Q->>Q: 2. Parse CLI args

    alt Interactive mode
        Q->>Q: 3. run_interactive()
    else Single-shot mode
        Q->>Q: 3. run_query(question)
    end

    Q->>Q: 4. _load_models_and_collection()
    Q->>C: 5. load_embed_model(EMBED_MODEL)
    C-->>Q: 6. Embedding model loaded
    Q->>C: 7. load_rerank_model(RERANK_MODEL)
    C-->>Q: 8. Re-ranker loaded
    Q->>C: 9. get_collection()
    C->>V: 10. Open/create Chroma collection
    V-->>Q: 11. Return collection handle

    Q->>Q: 12. _execute_query(question)
    Q->>Q: 13. hybrid_retrieve(question)

    Q->>B: 14. load_bm25_corpus()
    B-->>Q: 15. Return BM25 rows
    Q->>Q: 16. Apply disabled-source filtering
    Q->>Q: 17. Tokenize corpus + question
    Q->>Q: 18. BM25Okapi scoring and top-K IDs

    Q->>E: 19. Encode prefixed query (BGE prefix)
    E-->>Q: 20. Query embedding
    Q->>V: 21. Vector top-K query (optional source filter)
    V-->>Q: 22. Vector IDs + docs + metadata

    Q->>Q: 23. RRF fuse BM25 and vector ranked IDs
    Q->>Q: 24. Build candidate ID -> text/metadata map
    Q->>R: 25. Cross-encoder re-rank candidates
    R-->>Q: 26. Ranked scores
    Q->>Q: 27. Select FINAL_TOP_K chunks

    Q->>Q: 28. Print retrieved chunks preview

    alt --no-llm
        Q-->>U: 29. Return retrieved chunks only
    else With LLM answer
        Q->>Q: 29. generate_answer(question, top_chunks)
        Q->>O: 30. chat(system+context+question)
        O-->>Q: 31. Cited answer text
        Q-->>U: 32. Print final answer
    end
```

## 2) Code Flow (Function by Function)

```mermaid
flowchart TD
    A[CLI entry] --> B{args.interactive OR no question?}
    B -->|Yes| C[run_interactive]
    B -->|No| D[run_query]

    C --> E[_load_models_and_collection]
    D --> E

    E --> F[load_embed_model]
    E --> G[load_rerank_model]
    E --> H[get_collection]

    C --> I[loop input questions]
    I --> J[_execute_query]
    D --> J

    J --> K[hybrid_retrieve]
    K --> L[load_bm25_corpus]
    K --> M[BM25Okapi keyword retrieval]
    K --> N[Embedding encode + Chroma vector retrieval]
    K --> O[RRF merge]
    K --> P[Cross-encoder rerank]
    K --> Q[top_chunks]

    J --> R{no_llm?}
    R -->|Yes| S[stop after retrieval preview]
    R -->|No| T[generate_answer]
    T --> U[Ollama chat]
    U --> V[print cited answer]
```

## 3) Algorithms and Models Used

- Lexical retrieval: `BM25Okapi` from `rank_bm25`.
- Semantic retrieval: embedding vector search against Chroma collection.
- Query embedding: `SentenceTransformer` loaded via `load_embed_model(EMBED_MODEL)`.
- BGE retrieval trick: prefixes query with `BGE_QUERY_PREFIX` before embedding.
- Fusion: Reciprocal Rank Fusion (RRF) with standard constant `k=60`.
- Final ranking: `CrossEncoder` loaded via `load_rerank_model(RERANK_MODEL)`.
- Answer generation: Ollama chat using `OLLAMA_MODEL` at `OLLAMA_HOST`.

## 4) What the Pipeline Actually Does

1. Loads models and vector collection once (especially useful for interactive mode).
2. Reads the persisted BM25 corpus JSON from disk.
3. Runs two retrieval paths in parallel logic:
   - Keyword path: BM25 top candidates.
   - Semantic path: vector top candidates from Chroma.
4. Merges candidate IDs via RRF.
5. Re-ranks merged candidates with a cross-encoder.
6. Displays the best chunks with source/page/reference metadata.
7. Optionally sends context to Ollama and prints a cited final answer.

## 5) Key Inputs and Outputs

- Input:
  - User question string.
  - Chroma collection documents and metadata.
  - BM25 corpus JSON rows.

- Output:
  - `top_chunks`: best retrieved passages after hybrid + rerank.
  - Optional final natural-language answer with citations.

## 6) Important Runtime Branches

- If vector DB is empty: exits and asks to ingest first.
- If BM25 corpus is empty: warns and returns no results.
- If all sources are disabled by filter: warns and returns no results.
- If `--no-llm` is set: retrieval only, no generation.
