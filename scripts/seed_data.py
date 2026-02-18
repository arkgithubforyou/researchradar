"""Build-time data seeding for HF Spaces deployment.

Downloads a small set of recent ACL papers, ingests them into SQLite + ChromaDB,
and runs regex-based entity enrichment. Designed to run during Docker build
so the Space starts with demo data.

Usage:
    python scripts/seed_data.py [--max-papers 500] [--year-from 2022] [--year-to 2024]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.ingestion.chunking import chunk_papers
from src.ingestion.embeddings import EmbeddingGenerator
from src.ingestion.load_acl_anthology import ACLAnthologyLoader
from src.enrichment.pipeline import EnrichmentPipeline
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEMO_VENUES = ["acl", "emnlp", "naacl", "findings", "eacl", "coling"]


def main():
    parser = argparse.ArgumentParser(description="Seed ResearchRadar with demo data")
    parser.add_argument("--max-papers", type=int, default=500)
    parser.add_argument("--year-from", type=int, default=2022)
    parser.add_argument("--year-to", type=int, default=2024)
    args = parser.parse_args()

    config = get_config()

    # ── Step 1: Load papers from ACL Anthology ────────────────────────
    logger.info("=== Step 1: Loading papers from ACL Anthology ===")
    loader = ACLAnthologyLoader()
    papers = loader.load(
        year_from=args.year_from,
        year_to=args.year_to,
        venues=DEMO_VENUES,
        max_papers=args.max_papers,
    )
    logger.info("Loaded %d papers", len(papers))

    if not papers:
        logger.error("No papers loaded. Exiting.")
        sys.exit(1)

    # ── Step 2: Store in SQLite ───────────────────────────────────────
    logger.info("=== Step 2: Storing in SQLite ===")
    db = SQLiteDB(config.sqlite_db_path)
    db.create_schema()
    db.insert_papers(papers)
    logger.info("SQLite: %d papers stored", db.get_paper_count())

    # ── Step 3: Chunk papers ──────────────────────────────────────────
    logger.info("=== Step 3: Chunking papers ===")
    chunks = chunk_papers(papers, strategy="abstract")
    db.insert_chunks(chunks)
    logger.info("SQLite: %d chunks stored", db.get_chunk_count())

    # ── Step 4: Generate embeddings ───────────────────────────────────
    logger.info("=== Step 4: Generating embeddings ===")
    chroma = ChromaStore(config.chroma_db_path)
    embedder = EmbeddingGenerator(config.embedding_model)
    chunks_with_meta = db.get_all_chunks()
    embedder.embed_and_store(chunks_with_meta, chroma)
    logger.info("ChromaDB: %d embeddings stored", chroma.count())

    # ── Step 5: Regex-based entity enrichment ─────────────────────────
    logger.info("=== Step 5: Enriching with regex extraction ===")
    enricher = EnrichmentPipeline(
        db=db,
        llm_backend=None,  # No LLM at build time — regex only
        use_regex_fallback=True,
    )
    stats = enricher.enrich_all()
    logger.info(
        "Enrichment: %d methods, %d datasets, %d tasks, %d topics",
        stats.total_methods,
        stats.total_datasets,
        stats.total_tasks,
        stats.total_topics,
    )

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("=== Seeding complete ===")
    logger.info(
        "Papers: %d | Chunks: %d | Embeddings: %d",
        db.get_paper_count(),
        db.get_chunk_count(),
        chroma.count(),
    )


if __name__ == "__main__":
    main()
