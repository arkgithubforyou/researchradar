"""Re-chunk all papers with full text using fixed-size strategy.

Clears old abstract-only chunks, creates new 512-token chunks from full text,
resets ChromaDB, and re-embeds everything.

Usage:
    python scripts/rechunk.py                          # Full pipeline
    python scripts/rechunk.py --skip-embeddings        # Chunks only (no embed)
    python scripts/rechunk.py --strategy section       # Section-aware chunks
    python scripts/rechunk.py --batch-size 128         # Bigger embedding batches
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.ingestion.base_loader import PaperRecord
from src.ingestion.chunking import chunk_paper, CHUNKING_STRATEGIES
from src.ingestion.embeddings import EmbeddingGenerator
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_papers_from_db(db: SQLiteDB) -> list[PaperRecord]:
    """Load all papers from SQLite as PaperRecord objects."""
    conn = db.get_connection()
    try:
        rows = conn.execute("""
            SELECT id, title, abstract, year, venue, url, full_text
            FROM papers
            ORDER BY year DESC, id
        """).fetchall()

        papers = []
        for row in rows:
            paper = PaperRecord(
                source=row["id"].split("::")[0] if "::" in row["id"] else "unknown",
                source_id=row["id"].split("::")[-1] if "::" in row["id"] else row["id"],
                title=row["title"] or "",
                abstract=row["abstract"] or "",
                year=row["year"],
                venue=row["venue"] or "",
                url=row["url"] or "",
                full_text=row["full_text"] or "",
                authors=[],
            )
            papers.append(paper)
        return papers
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Re-chunk papers with full text")
    parser.add_argument(
        "--strategy", choices=list(CHUNKING_STRATEGIES.keys()), default="fixed",
        help="Chunking strategy (default: fixed)",
    )
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    config = get_config()
    db = SQLiteDB(config.sqlite_db_path)

    # ── Step 1: Load papers ────────────────────────────────────────────
    logger.info("=== Step 1: Loading papers from DB ===")
    papers = load_papers_from_db(db)
    with_ft = sum(1 for p in papers if p.has_full_text())
    logger.info("Loaded %d papers (%d with full text)", len(papers), with_ft)

    # ── Step 2: Clear old chunks ───────────────────────────────────────
    logger.info("=== Step 2: Clearing old chunks ===")
    conn = db.get_connection()
    old_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()
    logger.info("Deleted %d old chunks", old_count)

    # ── Step 3: Re-chunk with full text ────────────────────────────────
    logger.info("=== Step 3: Chunking papers (strategy=%s) ===", args.strategy)
    start = time.time()
    all_chunks = []
    for i, paper in enumerate(papers):
        chunks = chunk_paper(paper, strategy=args.strategy)
        all_chunks.extend(chunks)
        if (i + 1) % 5000 == 0:
            logger.info("Chunked %d/%d papers → %d chunks so far",
                        i + 1, len(papers), len(all_chunks))

    elapsed = time.time() - start
    logger.info("Chunked %d papers → %d chunks in %.1fs (avg %.1f chunks/paper)",
                len(papers), len(all_chunks), elapsed,
                len(all_chunks) / len(papers) if papers else 0)

    # Insert into SQLite
    logger.info("Inserting chunks into SQLite...")
    db.insert_chunks(all_chunks)
    logger.info("SQLite: %d chunks stored", db.get_chunk_count())

    # ── Step 4: Reset ChromaDB and re-embed ────────────────────────────
    if not args.skip_embeddings:
        logger.info("=== Step 4: Re-embedding %d chunks ===", len(all_chunks))

        chroma = ChromaStore(config.chroma_db_path)
        logger.info("Resetting ChromaDB collection...")
        chroma.reset()

        embedder = EmbeddingGenerator(config.embedding_model)

        # Retrieve chunks with metadata for ChromaDB
        chunks_with_meta = db.get_all_chunks()
        logger.info("Embedding %d chunks (batch_size=%d)...", len(chunks_with_meta), args.batch_size)
        embedder.embed_and_store(chunks_with_meta, chroma, batch_size=args.batch_size)
        logger.info("ChromaDB: %d embeddings stored", chroma.count())
    else:
        logger.info("=== Step 4: Skipped (--skip-embeddings) ===")

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("=== Re-chunking complete ===")
    logger.info("Papers: %d | Chunks: %d | Strategy: %s",
                db.get_paper_count(), db.get_chunk_count(), args.strategy)


if __name__ == "__main__":
    main()
