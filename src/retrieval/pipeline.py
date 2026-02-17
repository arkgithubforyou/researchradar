"""End-to-end retrieval pipeline.

Orchestrates BM25 → vector → hybrid RRF → cross-encoder reranking
and resolves chunk IDs to full chunk data from SQLite.
"""

import logging
from dataclasses import dataclass, field

from src.ingestion.embeddings import EmbeddingGenerator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_retriever import VectorRetriever
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_db import SQLiteDB

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with all metadata and scores."""

    chunk_id: int | str
    paper_id: str
    chunk_text: str
    chunk_type: str
    title: str
    year: int
    venue: str
    rrf_score: float = 0.0
    ce_score: float | None = None


@dataclass
class RetrievalConfig:
    """Knobs for the retrieval pipeline."""

    bm25_top_k: int = 100
    vector_top_k: int = 100
    rrf_k: int = 60
    bm25_weight: int = 1
    vector_weight: int = 1
    hybrid_top_k: int = 50
    rerank_top_k: int = 20
    use_reranker: bool = True


class RetrievalPipeline:
    """Full retrieval pipeline: BM25 + vector → RRF → cross-encoder."""

    def __init__(
        self,
        db: SQLiteDB,
        chroma_store: ChromaStore,
        embedding_generator: EmbeddingGenerator,
        reranker: CrossEncoderReranker | None = None,
        config: RetrievalConfig | None = None,
    ):
        self.db = db
        self.config = config or RetrievalConfig()

        # Build sub-components
        self.bm25 = BM25Retriever(db)
        self.vector = VectorRetriever(chroma_store, embedding_generator)
        self.hybrid = HybridRetriever(
            self.bm25, self.vector, rrf_k=self.config.rrf_k
        )
        self.reranker = reranker

        # Chunk lookup cache (populated lazily)
        self._chunk_map: dict[int, dict] | None = None

    def build_index(self) -> int:
        """Build the BM25 index. Must be called before search."""
        return self.bm25.build_index()

    def _ensure_chunk_map(self):
        """Lazily load all chunks into a lookup dict keyed by chunk ID."""
        if self._chunk_map is not None:
            return
        all_chunks = self.db.get_all_chunks()
        self._chunk_map = {c["id"]: c for c in all_chunks}
        logger.info("Loaded %d chunks into lookup map", len(self._chunk_map))

    def _resolve_chunk(self, chunk_id: int | str) -> dict | None:
        """Look up full chunk data by ID."""
        self._ensure_chunk_map()
        return self._chunk_map.get(chunk_id)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Run the full retrieval pipeline.

        Steps:
            1. BM25 + vector search (via HybridRetriever with RRF)
            2. Resolve chunk IDs to texts
            3. Cross-encoder reranking (if enabled and reranker provided)

        Args:
            query: User query string.
            top_k: Override final number of results (default: config.rerank_top_k).
            where: Optional metadata filter for vector search.

        Returns:
            List of RetrievalResult sorted by final score (descending).
        """
        if not self.bm25.is_built:
            raise RuntimeError("BM25 index not built — call build_index() first")

        final_k = top_k or self.config.rerank_top_k

        # Step 1: Hybrid retrieval
        hybrid_results = self.hybrid.search(
            query,
            top_k=self.config.hybrid_top_k,
            bm25_weight=self.config.bm25_weight,
            vector_weight=self.config.vector_weight,
            where=where,
        )

        if not hybrid_results:
            return []

        # Step 2: Resolve chunk IDs to text + metadata
        resolved = []
        chunk_ids_for_rerank = []
        texts_for_rerank = []

        for item in hybrid_results:
            chunk = self._resolve_chunk(item["chunk_id"])
            if chunk is None:
                continue
            resolved.append({
                "chunk_id": item["chunk_id"],
                "rrf_score": item["rrf_score"],
                "chunk": chunk,
            })
            chunk_ids_for_rerank.append(item["chunk_id"])
            texts_for_rerank.append(chunk["chunk_text"])

        # Step 3: Cross-encoder reranking
        if self.config.use_reranker and self.reranker and texts_for_rerank:
            reranked = self.reranker.rerank(
                query=query,
                chunk_ids=chunk_ids_for_rerank,
                chunk_texts=texts_for_rerank,
                top_k=final_k,
            )

            # Build lookup for CE scores
            ce_scores = {r["chunk_id"]: r["ce_score"] for r in reranked}
            reranked_ids = {r["chunk_id"] for r in reranked}

            # Filter resolved to only reranked items, attach CE score
            results = []
            for item in resolved:
                if item["chunk_id"] in reranked_ids:
                    c = item["chunk"]
                    results.append(RetrievalResult(
                        chunk_id=item["chunk_id"],
                        paper_id=c["paper_id"],
                        chunk_text=c["chunk_text"],
                        chunk_type=c.get("chunk_type", ""),
                        title=c.get("title", ""),
                        year=c.get("year", 0),
                        venue=c.get("venue", ""),
                        rrf_score=item["rrf_score"],
                        ce_score=ce_scores.get(item["chunk_id"]),
                    ))

            # Sort by CE score
            results.sort(key=lambda r: r.ce_score or 0, reverse=True)
            return results[:final_k]

        # No reranker — return RRF-ranked results
        results = []
        for item in resolved[:final_k]:
            c = item["chunk"]
            results.append(RetrievalResult(
                chunk_id=item["chunk_id"],
                paper_id=c["paper_id"],
                chunk_text=c["chunk_text"],
                chunk_type=c.get("chunk_type", ""),
                title=c.get("title", ""),
                year=c.get("year", 0),
                venue=c.get("venue", ""),
                rrf_score=item["rrf_score"],
            ))

        return results
