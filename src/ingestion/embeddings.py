"""Embedding generation using sentence-transformers.

Encodes chunk texts into vectors and stores them in ChromaDB.
"""

import logging

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for chunks and store in ChromaDB."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info("Embedding model loaded (dim=%d)", self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Encode texts into embedding vectors.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.

        Returns:
            List of embedding vectors as lists of floats.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string."""
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def embed_and_store(
        self,
        chunks: list[dict],
        chroma_store: ChromaStore,
        batch_size: int = 64,
    ):
        """Encode chunks and store embeddings in ChromaDB.

        Args:
            chunks: List of chunk dicts from chunking module. Each must have:
                    paper_id, chunk_text, chunk_type, chunk_index, token_count.
                    Optionally: year, venue, title (from joined query).
            chroma_store: ChromaStore instance.
            batch_size: Batch size for encoding.
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return

        texts = [c["chunk_text"] for c in chunks]

        logger.info("Encoding %d chunks with %s", len(texts), self.model_name)
        all_embeddings = self.encode(texts, batch_size=batch_size)

        # Prepare ChromaDB data
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for i, chunk in enumerate(tqdm(chunks, desc="Preparing ChromaDB data")):
            # Use chunk table ID if available, otherwise generate one
            chunk_id = str(chunk.get("id", f"{chunk['paper_id']}_chunk_{chunk['chunk_index']}"))

            ids.append(chunk_id)
            documents.append(chunk["chunk_text"])
            embeddings.append(all_embeddings[i])
            metadatas.append({
                "paper_id": chunk["paper_id"],
                "chunk_type": chunk.get("chunk_type", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "year": chunk.get("year", 0),
                "venue": chunk.get("venue", ""),
                "title": chunk.get("title", ""),
            })

        chroma_store.add_embeddings(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Stored %d embeddings in ChromaDB", len(ids))
