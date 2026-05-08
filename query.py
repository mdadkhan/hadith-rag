"""
query.py
--------
Query pipeline: hybrid search (BM25 + vector) → RRF → re-rank → LLM answer.

Usage:
    python3 query.py "your question"          # full pipeline with LLM answer
    python3 query.py "your question" --no-llm # show retrieved chunks only
    python3 query.py -i                     # interactive mode, ask multiple questions without reloading models

Install dependencies (once):
    pip3 install sentence-transformers rank-bm25 chromadb ollama
"""

import argparse
import re
import sys

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from config import (
    EMBED_MODEL,
    RERANK_MODEL,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    BGE_QUERY_PREFIX,
    RETRIEVAL_K,
    FINAL_TOP_K,
    DISABLED_SOURCES,
    get_collection,
    load_bm25_corpus,
    load_embed_model,
    load_rerank_model,
)


# ===========================================================================
# Step 1 — Hybrid retrieval: BM25 + Vector  →  RRF  →  Cross-encoder re-rank
# ===========================================================================

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


# Arabic Unicode blocks: Arabic, Supplement, Extended-A, Presentation Forms A/B, symbols
_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)


def _strip_arabic(text: str) -> str:
    """Remove Arabic characters for clean terminal display."""
    cleaned = _ARABIC_RE.sub("", text)
    # Collapse multiple blank lines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """
    Combine multiple ranked lists of document IDs into a single RRF score dict.
    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_retrieve(
    question: str,
    collection,
    embed_model: SentenceTransformer,
    rerank_model: CrossEncoder,
    retrieval_k: int = RETRIEVAL_K,
    final_top_k: int = FINAL_TOP_K,
) -> list[dict]:
    """
    Full retrieval pipeline:
      1. BM25 keyword search over the full corpus → top retrieval_k IDs
      2. Vector similarity search (BGE)           → top retrieval_k IDs
      3. Reciprocal Rank Fusion (RRF) merges both lists
      4. Cross-encoder re-ranks the merged candidates
      5. Return final_top_k best chunks with metadata
    """
    bm25_corpus = load_bm25_corpus()

    if not bm25_corpus:
        print("[Warning] BM25 keyword index is empty — run ingest_bukhari_v1_lc.py or ingest_muslim_v1_lc.py first.")
        return []

    # Apply source filter: drop disabled books from BM25 corpus
    if DISABLED_SOURCES:
        bm25_corpus = [e for e in bm25_corpus if e["source"] not in DISABLED_SOURCES]
        if not bm25_corpus:
            print("[Warning] All sources are disabled — nothing to search.")
            return []

    # ── 1. BM25 keyword search ───────────────────────────────────────────────
    tokenized_corpus = [_tokenize(entry["text"]) for entry in bm25_corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = _tokenize(question)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_indices = np.argsort(bm25_scores)[::-1][:retrieval_k]
    bm25_ranked_ids  = [bm25_corpus[i]["id"] for i in bm25_top_indices]

    # ── 2. Vector search ────────────────────────────────────────────────────
    # BGE query prefix improves retrieval accuracy significantly
    query_with_prefix = BGE_QUERY_PREFIX + question
    q_embedding = embed_model.encode(
        [query_with_prefix], normalize_embeddings=True
    ).tolist()

    # Build ChromaDB where-filter to exclude disabled sources
    if DISABLED_SOURCES:
        enabled_sources = [
            e["source"] for e in
            {e["source"]: e for e in load_bm25_corpus()}.values()
            if e["source"] not in DISABLED_SOURCES
        ]
        where_filter = {"source": {"$in": enabled_sources}} if enabled_sources else None
    else:
        where_filter = None

    vector_query_kwargs = dict(
        query_embeddings=q_embedding,
        n_results=min(retrieval_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where_filter:
        vector_query_kwargs["where"] = where_filter

    vector_results = collection.query(**vector_query_kwargs)
    vector_ranked_ids = vector_results["ids"][0]

    # ── 3. RRF fusion ────────────────────────────────────────────────────────
    rrf_scores = _reciprocal_rank_fusion([bm25_ranked_ids, vector_ranked_ids])

    all_candidate_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    # Build a lookup from ID → (text, metadata)
    id_to_doc: dict[str, dict] = {}
    for entry in bm25_corpus:
        id_to_doc[entry["id"]] = {
            "text":      entry["text"],
            "source":    entry["source"],
            "page":      entry["page"],
            "reference": entry.get("reference", ""),
        }
    for doc_id, doc_text, meta in zip(
        vector_results["ids"][0],
        vector_results["documents"][0],
        vector_results["metadatas"][0],
    ):
        id_to_doc[doc_id] = {
            "text":      doc_text,
            "source":    meta["source"],
            "page":      meta["page"],
            "reference": meta.get("reference", ""),
        }

    candidates = [id_to_doc[cid] for cid in all_candidate_ids if cid in id_to_doc]

    if not candidates:
        return []

    # ── 4. Cross-encoder re-ranking ──────────────────────────────────────────
    pairs  = [(question, c["text"]) for c in candidates]
    scores = rerank_model.predict(pairs, show_progress_bar=False)

    ranked     = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in ranked[:final_top_k]]

    return top_chunks


# ===========================================================================
# Step 2 — Answer generation with source citations
# ===========================================================================
'''
_SYSTEM_PROMPT = """\
You are a precise document assistant. Follow these rules without exception:
1. Answer ONLY using the information in the CONTEXT blocks provided.
2. Do NOT use external knowledge or assumptions.
3. After each claim, cite its source in this format: [SourceFile | Page N]
4. If the answer cannot be found in the context, respond with exactly:
   "I could not find an answer in the provided documents."
5. Keep your answer concise and factual.
"""
'''
_SYSTEM_PROMPT = """\
You are a precise document assistant. Follow these rules without exception:
1. Answer using the information in the CONTEXT blocks provided.
2. Do NOT use external knowledge or assumptions.
3. After each claim, cite its source in this format: [Book | Hadith reference | Page N]
"""

def generate_answer(question: str, top_chunks: list[dict]) -> str:
    """Send retrieved chunks to Ollama and return a cited answer."""
    if not top_chunks:
        return "No relevant content found in the knowledge base."

    context_blocks = []
    for i, chunk in enumerate(top_chunks, start=1):
        ref  = chunk.get("reference", "")
        ref_part = f" | Hadith: {ref}" if ref else ""
        header = f"[CONTEXT {i}] Source: {chunk['source']} | Page: {chunk['page']}{ref_part}"
        context_blocks.append(f"{header}\n{chunk['text']}")

    context_text = "\n\n".join(context_blocks)

    user_message = (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer strictly based on the context above. "
        f"Cite [Book | Hadith reference | Page N] after each claim:"
    )

    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            options={"temperature": 0.0},
        )
        return response["message"]["content"].strip()
    except ollama.ResponseError as e:
        return f"[Ollama error] {e.error}"
    except Exception as e:
        return f"[Ollama error] {e}"


# ===========================================================================
# Query entry point
# ===========================================================================

def _execute_query(
    question: str,
    collection,
    embed_model,
    rerank_model,
    no_llm: bool = False,
) -> None:
    """Run one question against already-loaded models and collection."""
    print(f"\nSearching for: \"{question}\"")
    print("─" * 60)

    top_chunks = hybrid_retrieve(
        question, collection, embed_model, rerank_model,
        retrieval_k=RETRIEVAL_K, final_top_k=FINAL_TOP_K,
    )

    if not top_chunks:
        print("No results found.")
        return

    # ── Show retrieved chunks ────────────────────────────────────────────────
    print(f"\nTop {len(top_chunks)} chunks after hybrid search + re-ranking:\n")
    for i, chunk in enumerate(top_chunks, start=1):
        ref = chunk.get("reference", "")
        ref_str = f"  |  {ref}" if ref else ""
        print(f"[{i}] {chunk['source']}  |  Page {chunk['page']}{ref_str}")
        print("─" * 40)
        preview = _strip_arabic(chunk["text"])[:350]
        print(preview + ("..." if len(chunk["text"]) > 350 else ""))
        print()

    if no_llm:
        return

    # ── Generate answer with citations ───────────────────────────────────────
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    answer = generate_answer(question, top_chunks)
    print(answer)
    print()


def _load_models_and_collection():
    """Load models and DB once. Returns (embed_model, rerank_model, collection)."""
    print(f"Loading embedding model '{EMBED_MODEL}'...")
    embed_model  = load_embed_model(EMBED_MODEL)

    print(f"Loading re-ranking model '{RERANK_MODEL}'...")
    rerank_model = load_rerank_model(RERANK_MODEL)

    _, collection = get_collection()

    if collection.count() == 0:
        print("Vector DB is empty. Run ingest_bukhari_v1_lc.py or ingest_muslim_v1_lc.py first.")
        sys.exit(1)

    return embed_model, rerank_model, collection


def run_query(question: str, no_llm: bool = False) -> None:
    """Single-shot query: loads models, runs one question, exits."""
    embed_model, rerank_model, collection = _load_models_and_collection()
    _execute_query(question, collection, embed_model, rerank_model, no_llm)


def run_interactive(no_llm: bool = False) -> None:
    """
    Interactive REPL: models are loaded ONCE and reused for every question.
    Models stay in RAM — no reload penalty between queries.
    Type 'quit' or press Ctrl+C to exit.
    """
    embed_model, rerank_model, collection = _load_models_and_collection()

    print("\nModels loaded. Type your question (or 'quit' to exit).")
    print("=" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        _execute_query(question, collection, embed_model, rerank_model, no_llm)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the RAG knowledge base and get a cited LLM answer"
    )
    parser.add_argument(
        "question",
        nargs="?",
        type=str,
        help="The question to ask (omit to start interactive mode)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive REPL — models loaded once, reused for every query",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Show retrieved chunks only — skip Ollama answer generation",
    )
    args = parser.parse_args()

    if args.interactive or not args.question:
        run_interactive(no_llm=args.no_llm)
    else:
        run_query(args.question, no_llm=args.no_llm)
