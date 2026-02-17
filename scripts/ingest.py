"""CLI: Full ingestion pipeline.

Loads papers from a data source, stores in SQLite, chunks them,
generates embeddings, and stores in ChromaDB.

Usage:
    python scripts/ingest.py --source hf --parquet-path data/raw/acl-publication-info.74k.parquet
    python scripts/ingest.py --source hf --parquet-path data/raw/acl-publication-info.74k.parquet --year-from 2018 --year-to 2022 --max-papers 5000
    python scripts/ingest.py --source acl --year-from 2023 --year-to 2025
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.ingestion.base_loader import DataLoader
from src.ingestion.chunking import chunk_papers
from src.ingestion.embeddings import EmbeddingGenerator
from src.ingestion.load_acl_anthology import ACLAnthologyLoader
from src.ingestion.load_hf_data import HFDataLoader
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_loader(args, config) -> DataLoader:
    """Create the appropriate data loader from CLI args."""
    if args.source == "hf":
        if not args.parquet_path:
            raise ValueError("--parquet-path is required for HuggingFace source")
        return HFDataLoader(args.parquet_path)
    elif args.source == "acl":
        return ACLAnthologyLoader()
    else:
        raise ValueError(f"Unknown source: {args.source}")


def main():
    parser = argparse.ArgumentParser(description="ResearchRadar ingestion pipeline")
    parser.add_argument(
        "--source", required=True, choices=["hf", "acl"],
        help="Data source: 'hf' (HuggingFace parquet) or 'acl' (ACL Anthology package)",
    )
    parser.add_argument("--parquet-path", type=str, help="Path to HF parquet file")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument(
        "--venues", type=str, nargs="+", default=None,
        help="Venue filter (e.g., acl emnlp naacl)",
    )
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument(
        "--chunk-strategy", choices=["abstract", "fixed", "section"], default="abstract",
    )
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")

    args = parser.parse_args()
    config = get_config()

    # 1. Load papers
    logger.info("=== Step 1: Loading papers ===")
    loader = get_loader(args, config)
    papers = loader.load(
        year_from=args.year_from,
        year_to=args.year_to,
        venues=args.venues,
        max_papers=args.max_papers,
    )
    logger.info("Loaded %d papers from %s", len(papers), loader.source_name)

    if not papers:
        logger.warning("No papers loaded. Exiting.")
        return

    # 2. Store in SQLite
    logger.info("=== Step 2: Storing in SQLite ===")
    db = SQLiteDB(config.sqlite_db_path)
    db.create_schema()
    db.insert_papers(papers)
    logger.info("SQLite: %d papers stored", db.get_paper_count())

    # 3. Chunk papers
    logger.info("=== Step 3: Chunking papers (strategy=%s) ===", args.chunk_strategy)
    chunks = chunk_papers(papers, strategy=args.chunk_strategy)
    db.insert_chunks(chunks)
    logger.info("SQLite: %d chunks stored", db.get_chunk_count())

    # 4. Generate embeddings and store in ChromaDB
    if not args.skip_embeddings:
        logger.info("=== Step 4: Generating embeddings ===")
        chroma = ChromaStore(config.chroma_db_path)
        embedder = EmbeddingGenerator(config.embedding_model)

        # Retrieve chunks with metadata for ChromaDB
        chunks_with_meta = db.get_all_chunks()
        embedder.embed_and_store(chunks_with_meta, chroma)
        logger.info("ChromaDB: %d embeddings stored", chroma.count())
    else:
        logger.info("=== Step 4: Skipped (--skip-embeddings) ===")

    logger.info("=== Ingestion complete ===")
    logger.info("Papers: %d | Chunks: %d", db.get_paper_count(), db.get_chunk_count())


if __name__ == "__main__":
    main()
