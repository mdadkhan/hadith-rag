"""
config.py
---------
Shared configuration, constants, and lightweight DB utilities for the
enhanced RAG pipeline.

Imported by both ingest.py and query.py.
"""

import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# HuggingFace connectivity  (set env vars before any model is loaded)
# ---------------------------------------------------------------------------

# Optional mirror — set HF_ENDPOINT=https://hf-mirror.com if huggingface.co
# is blocked or slow.  Leave unset to use the default HuggingFace CDN.
_hf_endpoint = os.environ.get("HF_ENDPOINT")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).parent

DATA_DIR   = _BASE_DIR / "data"

# Vector DB and keyword index directories
CHROMA_DIR = _BASE_DIR / "rag_knowledge_base"
BM25_FILE  = CHROMA_DIR / "bm25_keyword_index.json"

COLLECTION = "pdf_knowledge_base"

# ---------------------------------------------------------------------------
# Models  (override with env vars)
# ---------------------------------------------------------------------------

EMBED_MODEL  = os.environ.get("EMBED_MODEL",  "BAAI/bge-base-en-v1.5")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2") # best for prototyping; switch to a larger model for better relevance at scale.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")

# BGE models work best with this query prefix for asymmetric retrieval
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Chunking (semantic)
# ---------------------------------------------------------------------------

# BREAKPOINT_PERCENTILE : similarity drop threshold for detecting topic shifts.
#   Lower (e.g. 70) = more sensitive → more, smaller chunks.
#   Higher (e.g. 95) = less sensitive → fewer, larger chunks.
BREAKPOINT_PERCENTILE = float(os.environ.get("BREAKPOINT_PERCENTILE", "85"))
MIN_CHUNK_CHARS       = int(os.environ.get("MIN_CHUNK_CHARS", "100"))
MAX_CHUNK_CHARS       = int(os.environ.get("MAX_CHUNK_CHARS", "1000"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# RETRIEVAL_K : candidates fetched from EACH of BM25 and vector search.
#               These are then merged (RRF) and re-ranked down to FINAL_TOP_K.
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "20"))
FINAL_TOP_K = int(os.environ.get("FINAL_TOP_K", "5"))

# ---------------------------------------------------------------------------
# Source filtering
# ---------------------------------------------------------------------------

# PDF filenames to EXCLUDE from query results.
# Add or remove entries from the set below to enable/disable books.
# The DISABLED_SOURCES env var (comma-separated) is merged in at runtime.
DISABLED_SOURCES: set[str] = {
    "Ryadh_Saliheen.pdf",
    "Sahih Bukhari.pdf",
    "Sahih Muslim.pdf",
    # "sahih-al-bukhari-volume-1.pdf",
    # "sahih-al-bukhari-volume-2.pdf",
    # "sahih-al-bukhari-volume-3.pdf",
    # "sahih-al-bukhari-volume-4.pdf",
    # "sahih-al-bukhari-volume-5.pdf",
    # "sahih-al-bukhari-volume-6.pdf",
    # "sahih-al-bukhari-volume-7.pdf",
    # "sahih-al-bukhari-volume-8.pdf",
    # "sahih-al-bukhari-volume-9.pdf",
}

# Merge in any sources disabled via environment variable (comma-separated)
_disabled_env = os.environ.get("DISABLED_SOURCES", "")
DISABLED_SOURCES |= {s.strip() for s in _disabled_env.split(",") if s.strip()}

# ---------------------------------------------------------------------------
# Shared DB utilities  (used by both ingest.py and query.py)
# ---------------------------------------------------------------------------

def get_collection(reset: bool = False):
    """
    Open (or create) the ChromaDB collection.
    If reset=True, delete the existing collection and BM25 index first.
    Returns (client, collection).
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"Deleted collection '{COLLECTION}'")
        except Exception:
            pass
        if BM25_FILE.exists():
            BM25_FILE.unlink()
            print("Deleted BM25 keyword index")

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def load_bm25_corpus() -> list[dict]:
    """Load the persisted BM25 keyword index from disk."""
    if BM25_FILE.exists():
        try:
            return json.loads(BM25_FILE.read_text())
        except Exception:
            return []
    return []


def save_bm25_corpus(corpus: list[dict]) -> None:
    """Persist the BM25 keyword index to disk."""
    BM25_FILE.write_text(json.dumps(corpus))


# ---------------------------------------------------------------------------
# Model loading helpers  (online → local-cache fallback)
# ---------------------------------------------------------------------------

def load_embed_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    """
    Load a SentenceTransformer embedding model.
    If HuggingFace is unreachable, falls back to the local cache.
    Set HF_ENDPOINT env var to use a mirror (e.g. https://hf-mirror.com).
    """
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        if "timeout" in str(e).lower() or "connect" in str(e).lower():
            print(f"  [Warning] HuggingFace unreachable — loading '{model_name}' from local cache...")
            return SentenceTransformer(model_name, local_files_only=True)
        raise


def load_rerank_model(model_name: str = RERANK_MODEL) -> CrossEncoder:
    """
    Load a CrossEncoder re-ranking model.
    If HuggingFace is unreachable, falls back to the local cache.
    """
    try:
        return CrossEncoder(model_name)
    except Exception as e:
        if "timeout" in str(e).lower() or "connect" in str(e).lower():
            print(f"  [Warning] HuggingFace unreachable — loading '{model_name}' from local cache...")
            return CrossEncoder(model_name, local_files_only=True)
        raise
