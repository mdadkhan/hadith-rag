"""
ingest_bukhari_v1_lc.py
-----------------------
LangChain ingestion for 'sahih-al-bukhari-volume-1.pdf'
(Arabic-English bilingual typeset edition, Dr. Muhammad Muhsin Khan translation).

PDF structure:
  - Running page header : "N—THE BOOK OF XXXX  [Arabic]  <page_num>"
  - Book intro section  : "N - THE BOOK OF XXXX" (full intro paragraph)
  - Chapter marker      : "(N) CHAPTER. [title text]"
  - Hadith boundary     : "N. Narrated <narrator_name>:"
  - Arabic text         : interspersed inline with English (kept as-is)

One hadith = one LangChain Document.

Metadata stored per document:
    volume        : 1 (fixed for this file)
    book_num      : book number (int, 1-10 in this volume)
    book_title    : book title string (str)
    chapter_num   : chapter number within the book (int)
    chapter_title : chapter title text (str, may be empty)
    hadith_num    : global sequential hadith number (int)
    narrator      : narrator name (str)
    reference     : e.g. "Vol.1, Bk.2, Ch.3, No.45" (str)
    page          : PDF page where hadith starts (int)
    source        : PDF filename (str)

Usage:
    python3 ingest_bukhari_v1_lc.py              # ingest
    python3 ingest_bukhari_v1_lc.py --reset      # wipe DB + BM25, reprocess
    python3 ingest_bukhari_v1_lc.py --dry-run    # parse only, no storing
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

BUKHARI_DIR = DATA_DIR / "bukhari"


def _volume_from_filename(name: str) -> int:
    """Extract the volume number from a filename like 'sahih-al-bukhari-volume-3.pdf'."""
    m = re.search(r"volume[\s-]*(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Running page header on every content page:
# "1—THE BOOK OF REVELATION   [Arabic/noise]   46"
# Stripped to clean the flat text.
_PAGE_HEADER_RE = re.compile(
    r"(?m)^\d+\s*[-\u2014]\s*THE BOOK OF[^\n]*\n",
    re.IGNORECASE,
)

# Book section intro: "N - THE BOOK OF XXXX" (appears once at the start of each book)
# Broader match so we can get canonical book title + number.
_BOOK_INTRO_RE = re.compile(
    r"(?m)^(\d+)\s*[-\u2014]\s*THE BOOK OF\s+(.+?)(?:\s*\(i\.e\.[^)]*\))?\s*$",
    re.IGNORECASE,
)

# Chapter heading: "(N) CHAPTER. Optional title text"
# IMPORTANT: use [ .]* not [\s.]* — \s matches newlines and would
# consume into the next line, picking up the hadith text as the title.
_CHAPTER_RE = re.compile(
    r"(?m)^\((\d+)\)\s+CHAPTER[. ]*(.*)$",
)

# TOC-style lines (dots + page number at end) — skip these
_TOC_RE = re.compile(r"\.{3,}\s*\d+\s*$")

# Arabic font-noise: In this PDF the Arabic glyphs are encoded with a custom
# font that lacks Unicode mapping.  PyMuPDF extracts them as short
# garbled-Latin fragments (e.g. ":Jl", "L.", "jith", ":JU").
# These lines consist almost entirely of 1-2 char tokens and punctuation
# with no recognisable English word ≥ 3 chars.  Strip whole lines that match.
_NOISE_LINE_RE = re.compile(
    r"(?m)^(?:[A-Za-z0-9]{1,2}[.:\-\t ]*){2,}\s*$"
)
# Hadith boundary: "N. Narrated Narrator Name"
_HADITH_RE = re.compile(
    r"(?m)^(\d+)\.\s+Narrated\s+(.+?)(?:\s*[:\u2014]|\s*$)",
)

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
    # Strip lines that are pure font-noise (garbled Arabic glyph encodings)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # Keep the line if it contains at least one English word of 3+ chars
        if re.search(r"[A-Za-z]{3,}", line) or re.search(r"\d{2,}", line) or line.strip() == "":
            cleaned.append(line)
        # else: silently drop the noise line
    return "\n".join(cleaned)


def build_flat_text(pages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Strip running page headers + Arabic noise, then join all pages."""
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
# Step 3 — Build indices for books and chapters
# ---------------------------------------------------------------------------

def _extract_book_titles(pages: list[dict]) -> dict[int, str]:
    """
    Scan raw page text (before header stripping) to build book_num → title map.
    Uses the first clean non-TOC occurrence of each book number.
    Works for any volume — no hardcoded title list.
    """
    book_titles: dict[int, str] = {}
    for p in pages:
        for line in p["text"].splitlines():
            m = _BOOK_INTRO_RE.match(line.strip())
            if m:
                bnum = int(m.group(1))
                title = m.group(2).strip().title()
                if _TOC_RE.search(title):
                    continue
                if bnum not in book_titles:
                    book_titles[bnum] = f"The Book of {title}"
    return book_titles


def _build_page_book_map(pages: list[dict]) -> dict[int, int]:
    """
    Scan raw page text (before header stripping) to build a
    page_num → book_num map by tracking the running page header.
    The running header "N—THE BOOK OF X" on each content page tells
    us which book that page belongs to.
    """
    page_book: dict[int, int] = {}
    current_book = 0
    header_re = re.compile(
        r"^(\d+)\s*[-\u2014]\s*THE BOOK OF",
        re.IGNORECASE,
    )
    for p in pages:
        first_line = p["text"].split("\n")[0].strip() if p["text"] else ""
        m = header_re.match(first_line)
        if m:
            current_book = int(m.group(1))
        if current_book > 0:
            page_book[p["page"]] = current_book
    return page_book


def _build_chapter_index(
    flat_text: str,
) -> tuple[list[int], dict[int, tuple[int, str]]]:
    """
    Find chapter markers across the full flat text.
    min_offset=0 so TOC chapters are included; bisect always finds
    the content chapter (closest before each hadith), not the distant TOC entry.
    Skips TOC lines (dots + page number). Strips word-wrap hyphens.
    """
    chapter_map: dict[int, tuple[int, str]] = {}
    for m in _CHAPTER_RE.finditer(flat_text):
        title = m.group(2).strip().rstrip("-").strip()
        if _TOC_RE.search(title):
            continue
        chapter_map[m.start()] = (int(m.group(1)), title)
    chapter_starts = sorted(chapter_map.keys())
    return chapter_starts, chapter_map


def _lookup(
    hadith_offset: int,
    starts: list[int],
    index_map: dict,
    default,
):
    """Return the value from index_map whose key is the largest <= hadith_offset."""
    idx = bisect.bisect_right(starts, hadith_offset) - 1
    if idx < 0:
        return default
    return index_map[starts[idx]]


# ---------------------------------------------------------------------------
# Step 4 — Parse hadiths → LangChain Documents
# ---------------------------------------------------------------------------

def parse_hadiths(
    flat_text: str,
    offsets: list[tuple[int, int]],
    book_titles: dict[int, str],
    page_book_map: dict[int, int],
    pdf_file: Path,
    volume: int,
) -> list[Document]:
    """
    Split the flat text on every "N. Narrated X" boundary.
    Each block becomes one LangChain Document.
    """
    boundaries = [
        (m.start(), int(m.group(1)), m.group(2).strip())
        for m in _HADITH_RE.finditer(flat_text)
    ]

    if not boundaries:
        return []

    # Build chapter index across the full flat text (bisect handles TOC vs content)
    chapter_starts, chapter_map = _build_chapter_index(flat_text)

    documents: list[Document] = []

    for idx, (start, hadith_num, narrator_raw) in enumerate(boundaries):
        end   = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(flat_text)
        block = flat_text[start:end].strip()

        # Clean narrator: take text up to first newline or special char
        narrator = re.split(r"[\n\u2014:]", narrator_raw)[0].strip()
        narrator = re.sub(r"\s+", " ", narrator)

        # Look up book from page-level map (tracks running page header)
        page       = _page_for_offset(start, offsets)
        book_num   = page_book_map.get(page, 1)
        book_title = book_titles.get(book_num, f"Book {book_num}")

        chapter_info  = _lookup(start, chapter_starts, chapter_map, (0, ""))
        chapter_num, chapter_title = chapter_info

        reference = f"Vol.{volume}, Bk.{book_num}, Ch.{chapter_num}, No.{hadith_num}"

        documents.append(Document(
            page_content=block,
            metadata={
                "volume"       : volume,
                "book_num"     : book_num,
                "book_title"   : book_title,
                "chapter_num"  : chapter_num,
                "chapter_title": chapter_title,
                "hadith_num"   : hadith_num,
                "narrator"     : narrator,
                "reference"    : reference,
                "page"         : page,
                "source"       : pdf_file.name,
            },
        ))

    unique_books = len({d.metadata["book_num"] for d in documents})
    print(f"  Parsed {len(documents)} hadiths across {unique_books} books")
    return documents


# ---------------------------------------------------------------------------
# Step 5 — Embed + store using raw ChromaDB (bypasses LangChain upsert bug)
# ---------------------------------------------------------------------------

def embed_and_store(
    documents: list[Document],
    pdf_file: Path,
    volume: int,
    embeddings: HuggingFaceEmbeddings,
    collection,                  # raw chromadb Collection (None on first call)
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

        # Stable IDs: filename__vN_bNN_hNNNN
        raw_ids = [
            f"{pdf_file.name}__v{volume}"
            f"_b{d.metadata['book_num']:02d}"
            f"_h{d.metadata['hadith_num']:04d}"
            for d in batch
        ]

        # Deduplicate: skip IDs already stored (cross-batch) or repeated in this batch
        seen_in_batch: set = set()
        ids, deduped_batch = [], []
        skipped = 0
        for doc_id, doc in zip(raw_ids, batch):
            if doc_id in existing_ids or doc_id in seen_in_batch:
                skipped += 1
                continue
            seen_in_batch.add(doc_id)
            ids.append(doc_id)
            deduped_batch.append(doc)
        if skipped:
            print(f"    [Vol.{volume}] Skipped {skipped} duplicate ID(s) in batch {i // batch_size + 1}")

        if not ids:
            continue

        # Generate embeddings manually then use add() — avoids upsert compaction bug
        texts     = [doc.page_content for doc in deduped_batch]
        vectors   = embeddings.embed_documents(texts)
        metadatas = [doc.metadata for doc in deduped_batch]

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

        for doc_id, doc in zip(ids, deduped_batch):
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
# Pipeline entry point
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
    """Parse and ingest a single PDF volume. Returns (hadiths_stored, vectorstore)."""
    print(f"\n[Vol.{volume}] Processing: {pdf_file.name}")

    pages         = extract_pages(pdf_file)
    book_titles   = _extract_book_titles(pages)
    page_book_map = _build_page_book_map(pages)
    flat_text, offsets = build_flat_text(pages)
    documents     = parse_hadiths(flat_text, offsets, book_titles, page_book_map, pdf_file, volume)

    if not documents:
        print(f"  WARNING: No hadiths parsed in {pdf_file.name} — skipping.")
        return 0, vectorstore

    if dry_run:
        print(f"\n  === Sample hadiths (first 3) ===")
        for doc in documents[:3]:
            m = doc.metadata
            print(f"\n  {m['reference']}  |  {m['book_title']}")
            print(f"    Chapter  : {m['chapter_num']} — {m['chapter_title'] or '(no title)'}")
            print(f"    Narrator : {m['narrator']}")
            print(f"    Page     : {m['page']}")
            print(f"    Text     : {doc.page_content[:200]}...")
        print(f"\n  Total hadiths : {len(documents)}")
        print(f"  Total books   : {len({d.metadata['book_num'] for d in documents})}")
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
    if not BUKHARI_DIR.exists():
        sys.exit(f"ERROR: Bukhari directory not found at {BUKHARI_DIR}")

    # Collect PDFs to process
    if specific_file:
        # Allow bare name, relative, or absolute path
        candidate = Path(specific_file)
        if not candidate.is_absolute():
            candidate = BUKHARI_DIR / candidate
        if not candidate.exists():
            sys.exit(f"ERROR: File not found: {candidate}")
        pdf_files = [candidate]
    else:
        pdf_files = sorted(BUKHARI_DIR.glob("*.pdf"))
        if not pdf_files:
            sys.exit(f"ERROR: No PDF files found in {BUKHARI_DIR}")

    print(f"Found {len(pdf_files)} PDF(s) to process in {BUKHARI_DIR}")
    for f in pdf_files:
        print(f"  {f.name}")

    # One-time setup (skip for dry-run)
    embeddings  = None
    vectorstore = None
    bm25_corpus = []
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
        print(f"\nDry-run complete. {grand_total} hadiths would be ingested.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Sahih Al-Bukhari volumes from data/bukhari/ — one hadith per document"
    )
    parser.add_argument("--reset",   action="store_true",
                        help="Wipe vector DB and BM25 index, then reprocess all volumes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse only — no embedding or storing")
    parser.add_argument("--file",    metavar="FILENAME",
                        help="Process a single PDF (filename or path within data/bukhari/)")
    args = parser.parse_args()
    run_pipeline(reset=args.reset, dry_run=args.dry_run, specific_file=args.file)
