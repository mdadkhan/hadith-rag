"""
ingest_muslim_v1_lc.py
----------------------
LangChain ingestion for the Sahih Muslim bilingual typeset volumes stored in
data/muslim/  (e.g. sahih-muslim-volume-1.pdf …).

PDF structure:
  - Running page header : "The Book Of XXXX  <tab>  <page_num>"  (two lines)
  - Book intro section  : "N. The Book Of XXXX"  (appears once at book start)
  - Chapter marker      : "Chapter N. Title text"
  - Hadith boundary     : "[globalN] localN - (variant) Text…"
  - Arabic text         : garbled font-encoded glyphs (stripped during ingest)

One hadith = one LangChain Document.

Metadata stored per document:
    volume        : volume number extracted from filename (int)
    book_num      : sequential book number within the volume (int)
    book_title    : canonical book title (str)
    chapter_num   : chapter number within the book (int)
    chapter_title : chapter title text (str, may be empty)
    hadith_num    : global sequential hadith number [N] from PDF (int)
    hadith_local  : local hadith number within book section (int)
    reference     : e.g. "Vol.1, Bk.2, Ch.3, No.534" (str)
    page          : PDF page where hadith starts (int)
    source        : PDF filename (str)

Usage:
    python3 ingest_muslim_v1_lc.py                         # ingest all volumes
    python3 ingest_muslim_v1_lc.py --file volume-1.pdf     # ingest one specific volume
    python3 ingest_muslim_v1_lc.py --reset                 # wipe DB + BM25, reprocess all
    python3 ingest_muslim_v1_lc.py --dry-run               # parse only, no embedding
    python3 ingest_muslim_v1_lc.py --dry-run --file v1.pdf # dry-run a specific file
"""

import argparse
import bisect
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    DATA_DIR,
    CHROMA_DIR,
    BM25_FILE,
    EMBED_MODEL,
    COLLECTION,
    load_bm25_corpus,
    save_bm25_corpus,
)

# ---------------------------------------------------------------------------
# Source directory
# ---------------------------------------------------------------------------

MUSLIM_DIR = DATA_DIR / "muslim"


def _volume_from_filename(name: str) -> int:
    """Extract volume number from filename like 'sahih-muslim-volume-1.pdf'."""
    m = re.search(r"volume[\s-]*(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Running page header — two lines:
#   "The Book Of Faith \t\n92 \t\n"
# Strip both the title line and the following page-number line.
_PAGE_HEADER_RE = re.compile(
    r"(?m)^The Book Of[^\n]*\n[ \t]*\d+[ \t]*\n",
    re.IGNORECASE,
)

# Intro section header (pages before the first book):
#   "Narrating from the Trustworthy... 44 \t\n"
_INTRO_HEADER_RE = re.compile(
    r"(?m)^Narrating from the Trustworthy[^\n]*\n[ \t]*\d+[ \t]*\n",
    re.IGNORECASE,
)

# Hadith boundary:  "[globalN] localN - (variant)"
_HADITH_RE = re.compile(
    r"(?m)^\[(\d+)\]\s+(\d+)\s*-\s*\([\d.]+\)",
)

# Chapter heading: "Chapter N. Title text"
_CHAPTER_RE = re.compile(
    r"(?m)^Chapter\s+(\d+)[.]\s*(.*)$",
)

# TOC lines (dots + page number)
_TOC_RE = re.compile(r"\.{3,}\s*\d+\s*$")


# ---------------------------------------------------------------------------
# Step 1 — Extract raw text per page
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        pages.append({"page": page_num, "text": page.get_text()})
    doc.close()
    print(f"  Loaded {len(pages)} PDF pages from '{pdf_path.name}'")
    return pages


# ---------------------------------------------------------------------------
# Step 2 — Build flat text with page-offset tracking
# ---------------------------------------------------------------------------

def _clean_page_text(text: str) -> str:
    """Remove running page headers and garbled Arabic-font noise lines."""
    text = _PAGE_HEADER_RE.sub("", text)
    text = _INTRO_HEADER_RE.sub("", text)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # Keep the line if it contains at least one English word of 3+ chars
        if re.search(r"[A-Za-z]{3,}", line) or re.search(r"\d{2,}", line) or line.strip() == "":
            cleaned.append(line)
        # else: silently drop garbled Arabic glyph noise
    return "\n".join(cleaned)


def build_flat_text(pages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Strip headers + noise, join all pages into one flat string."""
    flat, offsets = "", []
    for p in pages:
        cleaned = _clean_page_text(p["text"])
        offsets.append((len(flat), p["page"]))
        flat += cleaned + "\n"
    return flat, offsets


def _page_for_offset(offset: int, offsets: list[tuple[int, int]]) -> int:
    starts = [o[0] for o in offsets]
    idx = bisect.bisect_right(starts, offset) - 1
    return offsets[max(idx, 0)][1]


# ---------------------------------------------------------------------------
# Step 3 — Detect book boundaries (two-pass: TOC titles + content start pages)
# ---------------------------------------------------------------------------

# "N. The Book Of TITLE" — appears in TOC and at each book's intro page
_BOOK_INTRO_RE = re.compile(
    r"(?m)^(\d+)[.]\s+The\s+Book\s+Of\s+(.+?)(?:\s*\n|$)",
    re.IGNORECASE,
)

# First line of a hadith — used to find where content starts (skip TOC pages)
_HADITH_FIRST_LINE_RE = re.compile(r"^\[\d+\]\s+\d+\s*-\s*\([\d.]+\)")


def _build_page_book_map(pages: list[dict]) -> dict[int, tuple[int, str]]:
    """
    Two-pass book detection robust against OCR-garbled running headers.

    Pass 1: Extract canonical titles from TOC pages (pages before the first
            hadith). These are printed clearly and reliably.
    Pass 2: Find content-area book start pages by looking for the second
            occurrence of "N. The Book Of X" (after the TOC).
    Book numbers use the actual Islamic book numbers from the TOC (e.g.,
    12, 13, 14 for Zakât, Fasting, Hajj in volume 3).
    """
    # ── Pass 0: find the page of the first actual hadith ─────────────────
    first_content_page = pages[-1]["page"]
    for p in pages:
        for line in p["text"].splitlines():
            if _HADITH_FIRST_LINE_RE.match(line.strip()):
                first_content_page = p["page"]
                break
        if p["page"] == first_content_page:
            break

    # ── Pass 1: canonical titles from TOC (before first hadith page) ─────
    canonical: dict[int, str] = {}  # Islamic book_num → title
    for p in pages:
        if p["page"] >= first_content_page:
            break
        for m in _BOOK_INTRO_RE.finditer(p["text"]):
            n = int(m.group(1))
            raw = m.group(2).strip()
            # Strip trailing Arabic or garbled suffix after a dash/bracket
            raw = re.sub(r"\s+[-–—(].*$", "", raw).strip()
            if 1 <= n <= 50 and re.search(r"[A-Za-z]{3,}", raw) and n not in canonical:
                canonical[n] = "The Book Of " + raw

    # ── Pass 2: find where each book starts in content area ──────────────
    book_starts: dict[int, int] = {}  # Islamic book_num → start page
    for p in pages:
        if p["page"] < first_content_page:
            continue
        for m in _BOOK_INTRO_RE.finditer(p["text"]):
            n = int(m.group(1))
            title_check = m.group(2).strip()
            if (n in canonical
                    and n not in book_starts
                    and re.search(r"[A-Za-z]{3,}", title_check)):
                book_starts[n] = p["page"]

    # ── Build page → (book_num, title) from sorted page ranges ───────────
    if not book_starts:
        # Fallback: no book structure detected
        first_title = canonical.get(min(canonical), "Book 1") if canonical else "Book 1"
        first_num   = min(canonical) if canonical else 1
        return {p["page"]: (first_num, first_title) for p in pages}

    sorted_books = sorted(book_starts.items(), key=lambda x: x[1])
    page_book: dict[int, tuple[int, str]] = {}
    total_pages = pages[-1]["page"]

    for idx, (bnum, start_pg) in enumerate(sorted_books):
        end_pg = sorted_books[idx + 1][1] if idx + 1 < len(sorted_books) else total_pages + 1
        title = canonical.get(bnum, f"Book {bnum}")
        for pg in range(start_pg, end_pg):
            page_book[pg] = (bnum, title)

    # Pages before the first detected book (intro hadiths) → first book
    first_bnum, _ = sorted_books[0]
    first_title    = canonical.get(first_bnum, f"Book {first_bnum}")
    for p in pages:
        if p["page"] not in page_book:
            page_book[p["page"]] = (first_bnum, first_title)

    return page_book


# ---------------------------------------------------------------------------
# Step 4 — Build chapter index
# ---------------------------------------------------------------------------

def _build_chapter_index(
    flat_text: str,
    min_offset: int = 0,
) -> tuple[list[int], dict[int, tuple[int, str]]]:
    """
    Find 'Chapter N. Title' markers at or after min_offset.
    Skips TOC lines (dots + page number). Strips trailing word-wrap hyphens.
    """
    chapter_map: dict[int, tuple[int, str]] = {}
    for m in _CHAPTER_RE.finditer(flat_text):
        if m.start() < min_offset:
            continue
        title = m.group(2).strip().rstrip("-").strip()
        if _TOC_RE.search(title):
            continue
        chapter_map[m.start()] = (int(m.group(1)), title)
    return sorted(chapter_map.keys()), chapter_map


def _lookup(hadith_offset: int, starts: list[int], index_map: dict, default):
    """Return value from index_map whose key is the largest <= hadith_offset."""
    idx = bisect.bisect_right(starts, hadith_offset) - 1
    return index_map[starts[idx]] if idx >= 0 else default


# ---------------------------------------------------------------------------
# Step 5 — Parse hadiths → LangChain Documents
# ---------------------------------------------------------------------------

def parse_hadiths(
    flat_text: str,
    offsets: list[tuple[int, int]],
    page_book_map: dict[int, tuple[int, str]],
    pdf_file: Path,
    volume: int,
) -> list[Document]:
    """
    Split flat text on every '[N] M - (K)' boundary.
    Each block becomes one LangChain Document.
    """
    boundaries = [
        (m.start(), int(m.group(1)), int(m.group(2)))
        for m in _HADITH_RE.finditer(flat_text)
    ]

    if not boundaries:
        return []

    first_hadith_offset = boundaries[0][0]
    chapter_starts, chapter_map = _build_chapter_index(
        flat_text, min_offset=first_hadith_offset
    )

    documents: list[Document] = []

    for idx, (start, global_num, local_num) in enumerate(boundaries):
        end   = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(flat_text)
        block = flat_text[start:end].strip()

        page = _page_for_offset(start, offsets)
        book_num, book_title = page_book_map.get(page, (0, "Unknown"))

        chapter_num, chapter_title = _lookup(start, chapter_starts, chapter_map, (0, ""))

        reference = f"Vol.{volume}, Bk.{book_num}, Ch.{chapter_num}, No.{global_num}"

        documents.append(Document(
            page_content=block,
            metadata={
                "volume"       : volume,
                "book_num"     : book_num,
                "book_title"   : book_title,
                "chapter_num"  : chapter_num,
                "chapter_title": chapter_title,
                "hadith_num"   : global_num,
                "hadith_local" : local_num,
                "reference"    : reference,
                "page"         : page,
                "source"       : pdf_file.name,
            },
        ))

    unique_books = len({d.metadata["book_num"] for d in documents})
    print(f"  Parsed {len(documents)} hadiths across {unique_books} books")
    return documents


# ---------------------------------------------------------------------------
# Step 6 — Embed + store using raw ChromaDB (bypasses LangChain upsert bug)
# ---------------------------------------------------------------------------

def embed_and_store(
    documents: list[Document],
    pdf_file: Path,
    volume: int,
    embeddings: HuggingFaceEmbeddings,
    collection,                 # raw chromadb Collection (None on first call)
    bm25_corpus: list,
    existing_ids: set,
    batch_size: int = 256,
) -> tuple[int, object]:
    """
    Embed documents and add them to ChromaDB + BM25 corpus.
    Uses raw chromadb collection.add() to avoid the LangChain upsert
    compaction bug that occurs on existing collections.
    Returns (count_stored, collection).
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Open (or create) raw collection once — reused across all volumes
    if collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    total = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]

        # Stable IDs: filename__vN_bNN_hNNNN  (global hadith num)
        raw_ids = [
            f"{pdf_file.name}__v{volume}"
            f"_b{d.metadata['book_num']:02d}"
            f"_h{d.metadata['hadith_num']:04d}"
            for d in batch
        ]

        # Deduplicate within batch and against already-stored IDs
        seen_in_batch: set = set()
        ids, deduped = [], []
        skipped = 0
        for doc_id, doc in zip(raw_ids, batch):
            if doc_id in existing_ids or doc_id in seen_in_batch:
                skipped += 1
                continue
            seen_in_batch.add(doc_id)
            ids.append(doc_id)
            deduped.append(doc)
        if skipped:
            print(f"    [Vol.{volume}] Skipped {skipped} duplicate(s) in batch {i // batch_size + 1}")

        if not ids:
            continue

        # Generate embeddings manually then call add() — avoids upsert compaction bug
        texts      = [doc.page_content for doc in deduped]
        vectors    = embeddings.embed_documents(texts)
        metadatas  = [doc.metadata for doc in deduped]

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

        for doc_id, doc in zip(ids, deduped):
            bm25_corpus.append({
                "id"       : doc_id,
                "text"     : doc.page_content,
                "source"   : doc.metadata["source"],
                "page"     : doc.metadata["page"],
                "reference": doc.metadata["reference"],
            })
            existing_ids.add(doc_id)

        total += len(ids)
        print(f"    Stored batch {i // batch_size + 1}: {total}/{len(documents)} hadiths", end="\r")

    print()
    return total, collection


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------

def _process_one(
    pdf_file: Path,
    volume: int,
    dry_run: bool,
    embeddings,
    vectorstore,
    bm25_corpus: list,
    existing_ids: set,
) -> tuple[int, object]:
    """Parse and optionally ingest a single volume. Returns (count, vectorstore)."""
    print(f"\n[Vol.{volume}] Processing: {pdf_file.name}")

    pages         = extract_pages(pdf_file)
    page_book_map = _build_page_book_map(pages)
    flat_text, offsets = build_flat_text(pages)
    documents     = parse_hadiths(flat_text, offsets, page_book_map, pdf_file, volume)

    if not documents:
        print(f"  WARNING: No hadiths parsed in {pdf_file.name} — skipping.")
        return 0, vectorstore

    if dry_run:
        print(f"\n  === Sample hadiths (first 3) ===")
        for doc in documents[:3]:
            m = doc.metadata
            print(f"\n  {m['reference']}  |  {m['book_title']}")
            print(f"    Chapter  : {m['chapter_num']} — {m['chapter_title'] or '(no title)'}")
            print(f"    Page     : {m['page']}")
            print(f"    Text     : {doc.page_content[:200]}...")
        print(f"\n  Total hadiths : {len(documents)}")
        print(f"  Total books   : {len({d.metadata['book_num'] for d in documents})}")
        # Show book breakdown
        from collections import Counter
        counts = Counter(
            (d.metadata["book_num"], d.metadata["book_title"]) for d in documents
        )
        for (bk, title), cnt in sorted(counts.items()):
            print(f"    Book {bk:2d}: {cnt:3d} hadiths  — {title}")
        return len(documents), vectorstore

    stored, vectorstore = embed_and_store(
        documents, pdf_file, volume,
        embeddings, vectorstore,
        bm25_corpus, existing_ids,
    )
    print(f"  Stored {stored} hadiths")
    return stored, vectorstore


def run_pipeline(
    reset: bool = False,
    dry_run: bool = False,
    specific_file: str | None = None,
) -> None:
    if not MUSLIM_DIR.exists():
        sys.exit(f"ERROR: Muslim directory not found at {MUSLIM_DIR}")

    # Collect PDFs to process
    if specific_file:
        candidate = Path(specific_file)
        if not candidate.is_absolute():
            candidate = MUSLIM_DIR / candidate
        if not candidate.exists():
            sys.exit(f"ERROR: File not found: {candidate}")
        pdf_files = [candidate]
    else:
        pdf_files = sorted(MUSLIM_DIR.glob("*.pdf"))
        if not pdf_files:
            sys.exit(f"ERROR: No PDF files found in {MUSLIM_DIR}")

    print(f"Found {len(pdf_files)} PDF(s) to process in {MUSLIM_DIR}")
    for f in pdf_files:
        print(f"  {f.name}")

    # One-time setup
    embeddings   = None
    vectorstore  = None
    bm25_corpus  = []
    existing_ids: set = set()

    if not dry_run:
        if reset:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            try:
                client.delete_collection(COLLECTION)
                print(f"\nDeleted existing collection '{COLLECTION}'")
            except Exception:
                pass
            if BM25_FILE.exists():
                BM25_FILE.unlink()
                print("Deleted BM25 keyword index")

        print(f"\nLoading embedding model '{EMBED_MODEL}'...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        bm25_corpus  = load_bm25_corpus()
        existing_ids = {entry["id"] for entry in bm25_corpus}

    grand_total = 0
    for pdf_file in pdf_files:
        volume = _volume_from_filename(pdf_file.name)
        if volume == 0:
            print(f"  WARNING: Could not extract volume number from '{pdf_file.name}' — skipping.")
            continue
        stored, vectorstore = _process_one(
            pdf_file, volume, dry_run,
            embeddings, vectorstore,
            bm25_corpus, existing_ids,
        )
        grand_total += stored

    if not dry_run:
        save_bm25_corpus(bm25_corpus)
        print(f"\n{'='*60}")
        print(f"Grand total: {grand_total} hadiths ingested across {len(pdf_files)} volume(s)")
        print(f"Vector DB   : {CHROMA_DIR.resolve()}")
        print(f"Keyword idx : {BM25_FILE.resolve()}")
    else:
        print(f"\nDry-run complete. {grand_total} hadiths parsed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Sahih Muslim volumes from data/muslim/ — one hadith per document"
    )
    parser.add_argument("--reset",   action="store_true",
                        help="Wipe vector DB and BM25 index, then reprocess all volumes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse only — no embedding or storing")
    parser.add_argument("--file",    metavar="FILENAME",
                        help="Process a single PDF (filename or path within data/muslim/)")
    args = parser.parse_args()
    run_pipeline(reset=args.reset, dry_run=args.dry_run, specific_file=args.file)
