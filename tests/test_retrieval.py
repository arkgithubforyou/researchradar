"""Tests for the Phase 2 retrieval pipeline.

Covers BM25, vector, hybrid RRF, cross-encoder reranker, and the
full pipeline orchestrator.  All tests use in-memory fixtures —
no model downloads or GPU required.
"""

from unittest.mock import MagicMock

import pytest

from src.retrieval.bm25_retriever import BM25Retriever, _tokenize
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.pipeline import RetrievalConfig, RetrievalPipeline, RetrievalResult
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_retriever import VectorRetriever, _parse_chunk_id


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def seeded_db(tmp_db, sample_papers):
    """SQLiteDB with papers + chunks inserted."""
    tmp_db.insert_papers(sample_papers)
    chunks = [
        {
            "paper_id": "hf_acl_ocl::P18-1001",
            "chunk_text": "We propose a novel transformer architecture for machine translation.",
            "chunk_type": "abstract",
            "chunk_index": 0,
            "token_count": 10,
        },
        {
            "paper_id": "hf_acl_ocl::D19-1234",
            "chunk_text": "We propose LoRA, a parameter-efficient fine-tuning method using low-rank matrices.",
            "chunk_type": "abstract",
            "chunk_index": 0,
            "token_count": 12,
        },
        {
            "paper_id": "hf_acl_ocl::D19-1234",
            "chunk_text": "LoRA decomposes weight updates into low-rank matrices for efficient adaptation.",
            "chunk_type": "full_text",
            "chunk_index": 1,
            "token_count": 11,
        },
        {
            "paper_id": "acl_anthology::2022.acl-long.100",
            "chunk_text": "This paper surveys contrastive learning approaches applied to NLP tasks.",
            "chunk_type": "abstract",
            "chunk_index": 0,
            "token_count": 10,
        },
    ]
    tmp_db.insert_chunks(chunks)
    return tmp_db


@pytest.fixture
def mock_chroma(tmp_path):
    """A real ChromaStore with fake embeddings added."""
    from src.storage.chroma_store import ChromaStore

    store = ChromaStore(tmp_path / "chroma_test")
    store.add_embeddings(
        ids=["1", "2", "3", "4"],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
        ],
        documents=[
            "We propose a novel transformer architecture for machine translation.",
            "We propose LoRA, a parameter-efficient fine-tuning method using low-rank matrices.",
            "LoRA decomposes weight updates into low-rank matrices for efficient adaptation.",
            "This paper surveys contrastive learning approaches applied to NLP tasks.",
        ],
        metadatas=[
            {"paper_id": "hf_acl_ocl::P18-1001", "chunk_type": "abstract", "chunk_index": 0, "year": 2018, "venue": "acl", "title": "Attention Is All You Need (Not Really)"},
            {"paper_id": "hf_acl_ocl::D19-1234", "chunk_type": "abstract", "chunk_index": 0, "year": 2019, "venue": "emnlp", "title": "LoRA: Low-Rank Adaptation of Large Language Models"},
            {"paper_id": "hf_acl_ocl::D19-1234", "chunk_type": "full_text", "chunk_index": 1, "year": 2019, "venue": "emnlp", "title": "LoRA: Low-Rank Adaptation of Large Language Models"},
            {"paper_id": "acl_anthology::2022.acl-long.100", "chunk_type": "abstract", "chunk_index": 0, "year": 2022, "venue": "acl", "title": "Contrastive Learning for NLP: A Survey"},
        ],
    )
    return store


@pytest.fixture
def mock_embedding_generator():
    """EmbeddingGenerator mock that returns a fixed vector."""
    gen = MagicMock()
    gen.encode_query.return_value = [0.0, 1.0, 0.0]
    return gen


# ── Tokenization ─────────────────────────────────────────────────────


class TestTokenize:
    def test_lowercase(self):
        tokens = _tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_punctuation_split(self):
        tokens = _tokenize("fine-tuning, LoRA.")
        assert "lora" in tokens


# ── BM25 Retriever ───────────────────────────────────────────────────


class TestBM25Retriever:
    def test_build_index(self, seeded_db):
        bm25 = BM25Retriever(seeded_db)
        assert not bm25.is_built
        count = bm25.build_index()
        assert count == 4
        assert bm25.is_built

    def test_search_before_build_raises(self, seeded_db):
        bm25 = BM25Retriever(seeded_db)
        with pytest.raises(RuntimeError, match="not built"):
            bm25.search("transformer")

    def test_search_returns_results(self, seeded_db):
        bm25 = BM25Retriever(seeded_db)
        bm25.build_index()
        results = bm25.search("transformer architecture")
        assert len(results) > 0
        assert all("chunk_id" in r and "score" in r for r in results)
        assert results[0]["score"] > 0

    def test_search_ranking(self, seeded_db):
        bm25 = BM25Retriever(seeded_db)
        bm25.build_index()
        # "transformer" is unique to chunk 1 — BM25 IDF will score it
        results = bm25.search("novel transformer architecture")
        assert len(results) >= 1
        assert results[0]["chunk_id"] == 1

    def test_search_respects_top_k(self, seeded_db):
        bm25 = BM25Retriever(seeded_db)
        bm25.build_index()
        results = bm25.search("propose", top_k=2)
        assert len(results) <= 2

    def test_empty_db(self, tmp_db):
        bm25 = BM25Retriever(tmp_db)
        count = bm25.build_index()
        assert count == 0
        assert not bm25.is_built

    def test_no_match_returns_empty(self, seeded_db):
        bm25 = BM25Retriever(seeded_db)
        bm25.build_index()
        results = bm25.search("xyznonexistent")
        assert results == []


# ── Vector Retriever ─────────────────────────────────────────────────


class TestVectorRetriever:
    def test_search_returns_results(self, mock_chroma, mock_embedding_generator):
        vr = VectorRetriever(mock_chroma, mock_embedding_generator)
        results = vr.search("LoRA", top_k=4)
        assert len(results) > 0
        assert all("chunk_id" in r and "score" in r for r in results)

    def test_search_ranking(self, mock_chroma, mock_embedding_generator):
        """Query vector [0, 1, 0] should match LoRA chunks best."""
        vr = VectorRetriever(mock_chroma, mock_embedding_generator)
        results = vr.search("LoRA", top_k=4)
        top_id = results[0]["chunk_id"]
        assert top_id == 2 or top_id == "2"

    def test_metadata_returned(self, mock_chroma, mock_embedding_generator):
        vr = VectorRetriever(mock_chroma, mock_embedding_generator)
        results = vr.search("LoRA", top_k=1)
        assert "metadata" in results[0]
        assert "paper_id" in results[0]["metadata"]

    def test_document_returned(self, mock_chroma, mock_embedding_generator):
        vr = VectorRetriever(mock_chroma, mock_embedding_generator)
        results = vr.search("LoRA", top_k=1)
        assert "document" in results[0]
        assert len(results[0]["document"]) > 0


class TestParseChunkId:
    def test_integer_string(self):
        assert _parse_chunk_id("42") == 42

    def test_non_integer(self):
        assert _parse_chunk_id("abc_123") == "abc_123"

    def test_none(self):
        assert _parse_chunk_id(None) is None


# ── Reciprocal Rank Fusion ───────────────────────────────────────────


class TestRRF:
    def test_single_list(self):
        ranked = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
        fused = reciprocal_rank_fusion([ranked], k=60)
        assert len(fused) == 3
        assert fused[0]["chunk_id"] == "a"
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]

    def test_two_lists_agreement(self):
        """When both lists agree, top item should have highest score."""
        list1 = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        list2 = [{"chunk_id": "a"}, {"chunk_id": "c"}]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert fused[0]["chunk_id"] == "a"
        a_score = fused[0]["rrf_score"]
        assert a_score == pytest.approx(2.0 / 61)

    def test_disjoint_lists(self):
        list1 = [{"chunk_id": "a"}]
        list2 = [{"chunk_id": "b"}]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert len(fused) == 2
        assert fused[0]["rrf_score"] == fused[1]["rrf_score"]

    def test_empty_list(self):
        fused = reciprocal_rank_fusion([], k=60)
        assert fused == []

    def test_k_parameter_effect(self):
        ranked = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        fused_k1 = reciprocal_rank_fusion([ranked], k=1)
        fused_k100 = reciprocal_rank_fusion([ranked], k=100)
        assert fused_k1[0]["rrf_score"] > fused_k100[0]["rrf_score"]

    def test_weighted_fusion(self):
        """Repeating a list simulates upweighting."""
        list1 = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        fused_1x = reciprocal_rank_fusion([list1], k=60)
        fused_2x = reciprocal_rank_fusion([list1, list1], k=60)
        assert fused_2x[0]["rrf_score"] == pytest.approx(
            2 * fused_1x[0]["rrf_score"]
        )


# ── Hybrid Retriever ─────────────────────────────────────────────────


class TestHybridRetriever:
    def test_search(self, seeded_db, mock_chroma, mock_embedding_generator):
        bm25 = BM25Retriever(seeded_db)
        bm25.build_index()
        vr = VectorRetriever(mock_chroma, mock_embedding_generator)
        hybrid = HybridRetriever(bm25, vr)

        results = hybrid.search("LoRA low-rank", top_k=4)
        assert len(results) > 0
        assert all("chunk_id" in r and "rrf_score" in r for r in results)

    def test_top_k_respected(self, seeded_db, mock_chroma, mock_embedding_generator):
        bm25 = BM25Retriever(seeded_db)
        bm25.build_index()
        vr = VectorRetriever(mock_chroma, mock_embedding_generator)
        hybrid = HybridRetriever(bm25, vr)

        results = hybrid.search("LoRA", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_descending(self, seeded_db, mock_chroma, mock_embedding_generator):
        bm25 = BM25Retriever(seeded_db)
        bm25.build_index()
        vr = VectorRetriever(mock_chroma, mock_embedding_generator)
        hybrid = HybridRetriever(bm25, vr)

        results = hybrid.search("propose", top_k=4)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ── Cross-Encoder Reranker ───────────────────────────────────────────


class TestCrossEncoderReranker:
    def test_rerank_with_mock(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = MagicMock()
        reranker.model.predict.return_value = [0.9, 0.1, 0.5]

        results = reranker.rerank(
            query="test query",
            chunk_ids=[1, 2, 3],
            chunk_texts=["text a", "text b", "text c"],
            top_k=2,
        )
        assert len(results) == 2
        assert results[0]["chunk_id"] == 1
        assert results[0]["ce_score"] == pytest.approx(0.9)
        assert results[1]["chunk_id"] == 3

    def test_rerank_empty(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = MagicMock()
        results = reranker.rerank("q", [], [], top_k=5)
        assert results == []

    def test_top_k_limits_output(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = MagicMock()
        reranker.model.predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

        results = reranker.rerank(
            query="q",
            chunk_ids=[1, 2, 3, 4, 5],
            chunk_texts=["a", "b", "c", "d", "e"],
            top_k=3,
        )
        assert len(results) == 3
        assert results[0]["ce_score"] >= results[-1]["ce_score"]


# ── Full Pipeline ────────────────────────────────────────────────────


class TestRetrievalPipeline:
    def test_search_without_reranker(self, seeded_db, mock_chroma, mock_embedding_generator):
        config = RetrievalConfig(use_reranker=False, hybrid_top_k=10, rerank_top_k=4)
        pipeline = RetrievalPipeline(
            db=seeded_db,
            chroma_store=mock_chroma,
            embedding_generator=mock_embedding_generator,
            config=config,
        )
        pipeline.build_index()

        results = pipeline.search("LoRA low-rank matrices")
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.ce_score is None for r in results)
        assert all(r.paper_id for r in results)
        assert all(r.chunk_text for r in results)

    def test_search_with_reranker(self, seeded_db, mock_chroma, mock_embedding_generator):
        mock_reranker = MagicMock(spec=CrossEncoderReranker)
        mock_reranker.rerank.return_value = [
            {"chunk_id": 2, "ce_score": 0.95},
            {"chunk_id": 3, "ce_score": 0.80},
        ]

        config = RetrievalConfig(use_reranker=True, hybrid_top_k=10, rerank_top_k=4)
        pipeline = RetrievalPipeline(
            db=seeded_db,
            chroma_store=mock_chroma,
            embedding_generator=mock_embedding_generator,
            reranker=mock_reranker,
            config=config,
        )
        pipeline.build_index()

        results = pipeline.search("LoRA")
        assert len(results) > 0
        assert results[0].ce_score == pytest.approx(0.95)
        ce_scores = [r.ce_score for r in results]
        assert ce_scores == sorted(ce_scores, reverse=True)

    def test_search_before_build_raises(self, seeded_db, mock_chroma, mock_embedding_generator):
        pipeline = RetrievalPipeline(
            db=seeded_db,
            chroma_store=mock_chroma,
            embedding_generator=mock_embedding_generator,
        )
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.search("anything")

    def test_build_index_returns_count(self, seeded_db, mock_chroma, mock_embedding_generator):
        pipeline = RetrievalPipeline(
            db=seeded_db,
            chroma_store=mock_chroma,
            embedding_generator=mock_embedding_generator,
        )
        count = pipeline.build_index()
        assert count == 4

    def test_retrieval_result_fields(self, seeded_db, mock_chroma, mock_embedding_generator):
        config = RetrievalConfig(use_reranker=False)
        pipeline = RetrievalPipeline(
            db=seeded_db,
            chroma_store=mock_chroma,
            embedding_generator=mock_embedding_generator,
            config=config,
        )
        pipeline.build_index()

        results = pipeline.search("transformer")
        if results:
            r = results[0]
            assert isinstance(r.chunk_id, (int, str))
            assert isinstance(r.paper_id, str)
            assert isinstance(r.chunk_text, str)
            assert isinstance(r.rrf_score, float)

    def test_empty_query(self, seeded_db, mock_chroma, mock_embedding_generator):
        """Even a vague query should not crash."""
        config = RetrievalConfig(use_reranker=False)
        pipeline = RetrievalPipeline(
            db=seeded_db,
            chroma_store=mock_chroma,
            embedding_generator=mock_embedding_generator,
            config=config,
        )
        pipeline.build_index()
        results = pipeline.search("a")
        assert isinstance(results, list)


class TestRetrievalConfig:
    def test_defaults(self):
        config = RetrievalConfig()
        assert config.bm25_top_k == 100
        assert config.vector_top_k == 100
        assert config.rrf_k == 60
        assert config.hybrid_top_k == 50
        assert config.rerank_top_k == 20
        assert config.use_reranker is True

    def test_custom_values(self):
        config = RetrievalConfig(rrf_k=30, rerank_top_k=10)
        assert config.rrf_k == 30
        assert config.rerank_top_k == 10
