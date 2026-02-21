"""Embed all chunks into ChromaDB in memory-safe mega-batches.

Processes chunks in mega-batches (default 10K) to avoid OOM:
  encode batch → store in ChromaDB → free memory → next batch

Usage:
    python scripts/embed_chunks.py                    # defaults
    python scripts/embed_chunks.py --mega-batch 5000  # smaller mega-batches
    python scripts/embed_chunks.py --encode-batch 256 # bigger GPU batches
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB")
    parser.add_argument("--mega-batch", type=int, default=10000,
                        help="Chunks per mega-batch (encode+store cycle)")
    parser.add_argument("--encode-batch", type=int, default=128,
                        help="GPU encoding batch size")
    parser.add_argument("--chroma-batch", type=int, default=500,
                        help="ChromaDB insertion batch size")
    parser.add_argument("--no-reset", action="store_true",
                        help="Don't reset ChromaDB (resume mode)")
    args = parser.parse_args()

    config = get_config()
    db = SQLiteDB(config.sqlite_db_path)
    chroma = ChromaStore(config.chroma_db_path)

    # Reset ChromaDB unless resuming
    if not args.no_reset:
        logger.info("Resetting ChromaDB collection...")
        chroma.reset()

    # Load all chunks from DB
    logger.info("Loading chunks from SQLite...")
    all_chunks = db.get_all_chunks()
    total = len(all_chunks)
    logger.info("Total chunks: %d", total)

    # Load model
    logger.info("Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.embedding_model)
    dim = model.get_sentence_embedding_dimension()
    logger.info("Model loaded on %s (dim=%d)", model.device, dim)

    # Process in mega-batches
    start_time = time.time()
    total_stored = 0

    for mega_start in range(0, total, args.mega_batch):
        mega_end = min(mega_start + args.mega_batch, total)
        batch_chunks = all_chunks[mega_start:mega_end]
        batch_size = len(batch_chunks)

        logger.info("=== Mega-batch %d-%d / %d (%d chunks) ===",
                     mega_start, mega_end, total, batch_size)

        # Extract texts
        texts = [c["chunk_text"] for c in batch_chunks]

        # Encode on GPU
        t0 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=args.encode_batch,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        encode_time = time.time() - t0
        logger.info("Encoded %d chunks in %.1fs (%.0f chunks/sec)",
                     batch_size, encode_time, batch_size / encode_time)

        # Prepare ChromaDB data (no documents — text lives in SQLite only)
        ids = []
        metadatas = []
        emb_list = []

        for i, chunk in enumerate(batch_chunks):
            chunk_id = str(chunk.get("id", f"{chunk['paper_id']}_chunk_{chunk['chunk_index']}"))
            ids.append(chunk_id)
            emb_list.append(embeddings[i].tolist())
            metadatas.append({
                "paper_id": chunk["paper_id"],
                "chunk_type": chunk.get("chunk_type", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "year": chunk.get("year", 0),
                "venue": chunk.get("venue", ""),
                "title": chunk.get("title", ""),
            })

        # Store in ChromaDB in sub-batches
        t0 = time.time()
        chroma.add_embeddings(
            ids=ids,
            embeddings=emb_list,
            metadatas=metadatas,
            batch_size=args.chroma_batch,
        )
        store_time = time.time() - t0
        total_stored += batch_size
        logger.info("Stored in ChromaDB in %.1fs. Total stored: %d/%d",
                     store_time, total_stored, total)

        # Free memory
        del texts, embeddings, ids, metadatas, emb_list, batch_chunks
        gc.collect()

    elapsed = time.time() - start_time
    final_count = chroma.count()
    logger.info("=== EMBEDDING COMPLETE ===")
    logger.info("Total: %d embeddings in %.1fs (%.1f chunks/sec)",
                final_count, elapsed, total / elapsed)
    logger.info("ChromaDB count: %d", final_count)


if __name__ == "__main__":
    main()
