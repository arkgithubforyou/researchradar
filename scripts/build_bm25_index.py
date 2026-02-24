"""CLI: Build and serialize the BM25 index for fast startup.

Builds the BM25 index from all chunks in SQLite (slow NLTK tokenization),
then saves it as a pickle file. The serialized index can be loaded at
startup in ~1s instead of rebuilding from scratch (~15 min).

Usage:
    python -m scripts.build_bm25_index
    python scripts/build_bm25_index.py
    python scripts/build_bm25_index.py --output data/bm25_index.pkl
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.retrieval.bm25_retriever import BM25Retriever
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build and serialize the BM25 index")
    parser.add_argument(
        "--output", type=str, default=None, help="Output path (default: config.bm25_index_path)"
    )
    args = parser.parse_args()

    config = get_config()
    output_path = Path(args.output) if args.output else config.bm25_index_path

    logger.info("Opening SQLite DB at %s", config.sqlite_db_path)
    db = SQLiteDB(config.sqlite_db_path)

    retriever = BM25Retriever(db)

    logger.info("Building BM25 index (this will take a while)...")
    t0 = time.perf_counter()
    n_chunks = retriever.build_index()
    elapsed = time.perf_counter() - t0
    logger.info("BM25 index built: %d chunks in %.1fs", n_chunks, elapsed)

    retriever.save_index(output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Done. Output: %s (%.1f MB)", output_path, size_mb)


if __name__ == "__main__":
    main()
