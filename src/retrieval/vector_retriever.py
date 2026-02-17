"""Vector retrieval via ChromaDB cosine similarity.

Thin wrapper that encodes a query and returns scored chunk IDs
from the existing ChromaDB collection.
"""

import logging

from src.ingestion.embeddings import EmbeddingGenerator
from src.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Dense vector retriever using ChromaDB."""

    def __init__(
        self,
        chroma_store: ChromaStore,
        embedding_generator: EmbeddingGenerator,
    ):
        self.chroma_store = chroma_store
        self.embedding_generator = embedding_generator

    def search(
        self,
        query: str,
        top_k: int = 50,
        where: dict | None = None,
    ) -> list[dict]:
        """Search by embedding similarity.

        Args:
            query: Raw query string.
            top_k: Maximum results to return.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of dicts with keys: chunk_id, score, document, metadata.
            Score is 1 - cosine_distance (higher = more similar).
            Sorted by descending score.
        """
        query_embedding = self.embedding_generator.encode_query(query)
        results = self.chroma_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        # ChromaDB returns distances (lower = better for cosine).
        # Convert to similarity scores: score = 1 - distance.
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        scored = []
        for i in range(len(ids)):
            scored.append({
                "chunk_id": _parse_chunk_id(ids[i]),
                "score": 1.0 - distances[i],
                "document": documents[i],
                "metadata": metadatas[i],
            })

        return scored


def _parse_chunk_id(raw_id: str) -> int | str:
    """Parse ChromaDB ID back to integer chunk ID if possible."""
    try:
        return int(raw_id)
    except (ValueError, TypeError):
        return raw_id
