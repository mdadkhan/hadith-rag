"""
delete_source.py
----------------
Remove all documents for a specific PDF source from:
  - ChromaDB vector store
  - BM25 keyword index (JSON)

Usage:
    python3 delete_source.py "Sahih Bukhari.pdf"
    python3 delete_source.py "Sahih Muslim.pdf"
    python3 delete_source.py "Ryadh_Saliheen.pdf"

    python3 delete_source.py --list          # show all sources currently in the DB
    python3 delete_source.py --dry-run "..."  # preview what would be deleted
"""

import argparse
import sys

from config import get_collection, load_bm25_corpus, save_bm25_corpus


def list_sources(collection) -> None:
    total = collection.count()
    if total == 0:
        print("Vector DB is empty.")
        return

    # Fetch all metadata (no embeddings needed)
    result = collection.get(include=["metadatas"])
    sources: dict[str, int] = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "<unknown>")
        sources[src] = sources.get(src, 0) + 1

    print(f"\nVector DB  —  {total} total documents across {len(sources)} source(s):\n")
    for src, count in sorted(sources.items()):
        print(f"  {count:>6} docs  │  {src}")

    bm25 = load_bm25_corpus()
    bm25_sources: dict[str, int] = {}
    for entry in bm25:
        src = entry.get("source", "<unknown>")
        bm25_sources[src] = bm25_sources.get(src, 0) + 1

    print(f"\nBM25 index —  {len(bm25)} total entries across {len(bm25_sources)} source(s):\n")
    for src, count in sorted(bm25_sources.items()):
        print(f"  {count:>6} entries  │  {src}")
    print()


def delete_source(source_name: str, collection, dry_run: bool = False) -> None:
    # ── ChromaDB ──────────────────────────────────────────────────────────────
    result = collection.get(
        where={"source": source_name},
        include=["metadatas"],
    )
    chroma_ids = result["ids"]

    if not chroma_ids:
        print(f"No documents found in ChromaDB with source='{source_name}'")
    else:
        print(f"ChromaDB: found {len(chroma_ids)} documents for '{source_name}'")
        if not dry_run:
            # Delete in batches to avoid potential limits
            batch_size = 500
            for i in range(0, len(chroma_ids), batch_size):
                collection.delete(ids=chroma_ids[i : i + batch_size])
            print(f"  → Deleted {len(chroma_ids)} documents from ChromaDB")
        else:
            print(f"  [dry-run] Would delete {len(chroma_ids)} documents from ChromaDB")

    # ── BM25 index ────────────────────────────────────────────────────────────
    corpus = load_bm25_corpus()
    keep   = [e for e in corpus if e.get("source") != source_name]
    removed = len(corpus) - len(keep)

    if removed == 0:
        print(f"No entries found in BM25 index with source='{source_name}'")
    else:
        print(f"BM25 index: found {removed} entries for '{source_name}'")
        if not dry_run:
            save_bm25_corpus(keep)
            print(f"  → Deleted {removed} entries from BM25 index")
        else:
            print(f"  [dry-run] Would delete {removed} entries from BM25 index")

    if not dry_run and (chroma_ids or removed):
        remaining = collection.count()
        print(f"\nDone. Vector DB now has {remaining} documents.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete all documents for a PDF source from ChromaDB + BM25 index"
    )
    parser.add_argument(
        "source", nargs="?",
        help="Exact PDF filename to delete, e.g. 'Sahih Bukhari.pdf'"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all sources currently in the DB and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be deleted without making any changes"
    )
    args = parser.parse_args()

    _, collection = get_collection()

    if args.list:
        list_sources(collection)
        sys.exit(0)

    if not args.source:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Deleting source: '{args.source}'\n")
    delete_source(args.source, collection, dry_run=args.dry_run)
