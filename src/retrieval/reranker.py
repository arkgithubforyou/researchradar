"""Cross-encoder reranker using sentence-transformers.

Takes a query + candidate chunk texts and rescores them with a
cross-encoder model (ms-marco-MiniLM-L-6-v2 by default).
"""

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank candidate chunks using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info("Loading cross-encoder model: %s", model_name)
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info("Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        chunk_ids: list[int | str],
        chunk_texts: list[str],
        top_k: int = 20,
    ) -> list[dict]:
        """Rerank chunks by cross-encoder relevance.

        Args:
            query: The user query.
            chunk_ids: Corresponding chunk IDs.
            chunk_texts: Text of each candidate chunk.
            top_k: Number of results to return after reranking.

        Returns:
            List of {"chunk_id", "ce_score"} sorted by descending score.
        """
        if not chunk_ids:
            return []

        pairs = [[query, text] for text in chunk_texts]
        scores = self.model.predict(pairs)

        scored = [
            {"chunk_id": chunk_ids[i], "ce_score": float(scores[i])}
            for i in range(len(chunk_ids))
        ]
        scored.sort(key=lambda x: x["ce_score"], reverse=True)
        return scored[:top_k]
