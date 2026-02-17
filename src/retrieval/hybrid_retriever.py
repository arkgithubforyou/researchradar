"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Combines BM25 sparse scores and vector dense scores into a single
ranked list using the RRF formula:
    RRF_score(d) = sum( 1 / (k + rank_i(d)) ) for each ranker i

Reference: Cormack, Clarke & Buettcher (2009).
"""

import logging

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60  # Standard RRF constant


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Fuse multiple ranked lists using RRF.

    Args:
        ranked_lists: Each list contains dicts with at least "chunk_id".
                      Items are ordered by rank (best first).
        k: RRF smoothing constant (default 60).

    Returns:
        Fused list of {"chunk_id": ..., "rrf_score": ...} sorted descending.
    """
    fused_scores: dict[int | str, float] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            cid = item["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)

    results = [
        {"chunk_id": cid, "rrf_score": score}
        for cid, score in fused_scores.items()
    ]
    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


class HybridRetriever:
    """Combines BM25 and vector retrievers via RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 50,
        bm25_weight: int = 1,
        vector_weight: int = 1,
        where: dict | None = None,
    ) -> list[dict]:
        """Run hybrid search combining BM25 and vector results.

        Args:
            query: Raw query string.
            top_k: Final number of results to return.
            bm25_weight: Number of times to include BM25 list in RRF
                         (>1 upweights sparse signal).
            vector_weight: Number of times to include vector list in RRF.
            where: Optional metadata filter passed to vector retriever.

        Returns:
            List of {"chunk_id", "rrf_score"} sorted by descending rrf_score.
        """
        # Fetch more candidates from each retriever than final top_k
        fetch_k = min(top_k * 3, 200)

        bm25_results = self.bm25.search(query, top_k=fetch_k)
        vector_results = self.vector.search(query, top_k=fetch_k, where=where)

        logger.debug(
            "BM25 returned %d, vector returned %d candidates",
            len(bm25_results),
            len(vector_results),
        )

        ranked_lists = (
            [bm25_results] * bm25_weight + [vector_results] * vector_weight
        )

        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        return fused[:top_k]
