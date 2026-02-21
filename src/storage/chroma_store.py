"""ChromaDB vector store operations.

Handles collection management, embedding insertion, and similarity queries.
Embeddings are generated externally (see embeddings.py) and stored here.
"""

import logging
import sqlite3
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)


def _migrate_chroma_schema(persist_path: str) -> None:
    """Add missing columns to ChromaDB's internal SQLite if needed.

    chromadb 0.4.x expects a ``topic`` column on the ``collections`` table.
    Pre-built data created with an older version may lack this column,
    causing ``sqlite3.OperationalError: no such column: collections.topic``
    at query time.  We patch it up before ChromaDB opens the DB.
    """
    db_file = Path(persist_path) / "chroma.sqlite3"
    if not db_file.exists():
        return
    try:
        conn = sqlite3.connect(str(db_file))
        cols = [row[1] for row in conn.execute("PRAGMA table_info(collections)").fetchall()]
        if "topic" not in cols:
            logger.warning("ChromaDB migration: adding missing 'topic' column to collections table")
            conn.execute("ALTER TABLE collections ADD COLUMN topic TEXT NOT NULL DEFAULT ''")
            conn.commit()
        conn.close()
    except Exception as exc:
        logger.error("ChromaDB schema migration failed: %s", exc)


class ChromaStore:
    """ChromaDB interface for ResearchRadar."""

    COLLECTION_NAME = "paper_chunks"

    def __init__(self, persist_path: str | Path):
        self.persist_path = str(persist_path)
        self._client = None
        self._collection = None
        _migrate_chroma_schema(self.persist_path)

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_path)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        batch_size: int = 500,
    ):
        """Add embeddings to the collection in batches.

        Document texts are NOT stored in ChromaDB to avoid redundancy —
        they already live in SQLite and are resolved at query time by
        the retrieval pipeline.  This cuts chroma.sqlite3 size by ~40%.

        Args:
            ids: Unique string IDs for each embedding (chunk IDs).
            embeddings: Pre-computed embedding vectors.
            metadatas: Metadata dicts (paper_id, year, venue, chunk_type).
            batch_size: Number of embeddings per batch.
        """
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )
        logger.info("Added %d embeddings to ChromaDB", total)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 20,
        where: dict | None = None,
    ) -> dict:
        """Query the collection by embedding similarity.

        Args:
            query_embedding: Query vector.
            n_results: Number of results to return.
            where: Optional metadata filter (e.g., {"year": {"$gte": 2020}}).

        Returns:
            ChromaDB query result dict with ids, metadatas, distances.
            Documents are NOT stored/returned — text lives in SQLite only.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def get_by_paper_id(self, paper_id: str) -> dict:
        """Get all chunk embeddings/metadata for a specific paper."""
        return self.collection.get(
            where={"paper_id": paper_id},
            include=["metadatas", "embeddings"],
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass  # Collection may not exist
        self._collection = None
        logger.info("ChromaDB collection reset")
