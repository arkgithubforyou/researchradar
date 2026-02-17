"""ChromaDB vector store operations.

Handles collection management, embedding insertion, and similarity queries.
Embeddings are generated externally (see embeddings.py) and stored here.
"""

import logging
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)


class ChromaStore:
    """ChromaDB interface for ResearchRadar."""

    COLLECTION_NAME = "paper_chunks"

    def __init__(self, persist_path: str | Path):
        self.persist_path = str(persist_path)
        self._client = None
        self._collection = None

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
        documents: list[str],
        metadatas: list[dict],
        batch_size: int = 500,
    ):
        """Add embeddings to the collection in batches.

        Args:
            ids: Unique string IDs for each embedding (chunk IDs).
            embeddings: Pre-computed embedding vectors.
            documents: Original chunk texts.
            metadatas: Metadata dicts (paper_id, year, venue, chunk_type).
            batch_size: Number of embeddings per batch.
        """
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
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
            ChromaDB query result dict with ids, documents, metadatas, distances.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def get_by_paper_id(self, paper_id: str) -> dict:
        """Get all chunks for a specific paper."""
        return self.collection.get(
            where={"paper_id": paper_id},
            include=["documents", "metadatas", "embeddings"],
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self._collection = None
        logger.info("ChromaDB collection reset")
