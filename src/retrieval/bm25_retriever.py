"""BM25 sparse retrieval using rank-bm25.

Builds an in-memory BM25 index from SQLite chunk texts and returns
scored chunk IDs for a given query.
"""

import logging
import pickle
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from src.storage.sqlite_db import SQLiteDB

logger = logging.getLogger(__name__)

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for BM25."""
    return word_tokenize(text.lower())


class BM25Retriever:
    """BM25 sparse retriever backed by SQLite chunk texts."""

    def __init__(self, db: SQLiteDB):
        self.db = db
        self._index: BM25Okapi | None = None
        self._chunk_ids: list[int] = []

    @property
    def is_built(self) -> bool:
        return self._index is not None

    def build_index(self) -> int:
        """Build BM25 index from all chunks in SQLite.

        Returns:
            Number of chunks indexed.
        """
        texts, ids = self.db.get_chunk_texts_and_ids()
        if not texts:
            logger.warning("No chunks found — BM25 index is empty")
            self._index = None
            self._chunk_ids = []
            return 0

        tokenized = [_tokenize(t) for t in texts]
        self._index = BM25Okapi(tokenized)
        self._chunk_ids = ids
        logger.info("BM25 index built with %d chunks", len(ids))
        return len(ids)

    def save_index(self, path: Path) -> None:
        """Serialize the BM25 index and chunk IDs to a pickle file."""
        if not self.is_built:
            raise RuntimeError("BM25 index not built — nothing to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((self._index, self._chunk_ids), f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("BM25 index saved to %s (%.1f MB, %d chunks)", path, size_mb, len(self._chunk_ids))

    def load_index(self, path: Path) -> int:
        """Load a pre-built BM25 index from a pickle file.

        Returns:
            Number of chunks in the loaded index.
        """
        with open(path, "rb") as f:
            self._index, self._chunk_ids = pickle.load(f)
        logger.info("Loaded pre-built BM25 index (%d chunks) from %s", len(self._chunk_ids), path)
        return len(self._chunk_ids)

    def search(
        self, query: str, top_k: int = 50
    ) -> list[dict]:
        """Search the BM25 index.

        Args:
            query: Raw query string.
            top_k: Maximum results to return.

        Returns:
            List of dicts with keys: chunk_id, score.
            Sorted by descending score.
        """
        if not self.is_built:
            raise RuntimeError("BM25 index not built — call build_index() first")

        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Pair (chunk_id, score), drop zeros, sort descending
        scored = [
            {"chunk_id": self._chunk_ids[i], "score": float(scores[i])}
            for i in range(len(scores))
            if scores[i] > 0
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
