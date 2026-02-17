"""Tests for ChromaDB store."""

import pytest

from src.storage.chroma_store import ChromaStore


@pytest.fixture
def chroma(tmp_path) -> ChromaStore:
    return ChromaStore(tmp_path / "test_chroma")


class TestChromaStore:
    def test_empty_collection(self, chroma):
        assert chroma.count() == 0

    def test_add_and_count(self, chroma):
        chroma.add_embeddings(
            ids=["chunk_1", "chunk_2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            documents=["Hello world", "Goodbye world"],
            metadatas=[
                {"paper_id": "test::1", "chunk_type": "abstract", "year": 2020, "venue": "acl", "chunk_index": 0, "title": "T1"},
                {"paper_id": "test::2", "chunk_type": "abstract", "year": 2021, "venue": "emnlp", "chunk_index": 0, "title": "T2"},
            ],
        )
        assert chroma.count() == 2

    def test_query_returns_results(self, chroma):
        chroma.add_embeddings(
            ids=["c1", "c2", "c3"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
            documents=["transformer model", "dataset analysis", "attention mechanism"],
            metadatas=[
                {"paper_id": "t::1", "chunk_type": "abstract", "year": 2020, "venue": "acl", "chunk_index": 0, "title": "P1"},
                {"paper_id": "t::2", "chunk_type": "abstract", "year": 2021, "venue": "emnlp", "chunk_index": 0, "title": "P2"},
                {"paper_id": "t::3", "chunk_type": "abstract", "year": 2022, "venue": "acl", "chunk_index": 0, "title": "P3"},
            ],
        )
        results = chroma.query(query_embedding=[1.0, 0.0, 0.0], n_results=2)
        assert len(results["ids"][0]) == 2
        # Closest to [1,0,0] should be c1 and c3
        assert "c1" in results["ids"][0]
        assert "c3" in results["ids"][0]

    def test_query_with_metadata_filter(self, chroma):
        chroma.add_embeddings(
            ids=["c1", "c2"],
            embeddings=[[1.0, 0.0], [0.9, 0.1]],
            documents=["doc1", "doc2"],
            metadatas=[
                {"paper_id": "t::1", "chunk_type": "abstract", "year": 2020, "venue": "acl", "chunk_index": 0, "title": "P1"},
                {"paper_id": "t::2", "chunk_type": "abstract", "year": 2022, "venue": "emnlp", "chunk_index": 0, "title": "P2"},
            ],
        )
        results = chroma.query(
            query_embedding=[1.0, 0.0], n_results=10,
            where={"year": {"$gte": 2021}},
        )
        assert len(results["ids"][0]) == 1
        assert results["ids"][0][0] == "c2"

    def test_get_by_paper_id(self, chroma):
        chroma.add_embeddings(
            ids=["c1", "c2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            documents=["doc1", "doc2"],
            metadatas=[
                {"paper_id": "test::1", "chunk_type": "abstract", "year": 2020, "venue": "acl", "chunk_index": 0, "title": "T1"},
                {"paper_id": "test::2", "chunk_type": "abstract", "year": 2021, "venue": "emnlp", "chunk_index": 0, "title": "T2"},
            ],
        )
        results = chroma.get_by_paper_id("test::1")
        assert len(results["ids"]) == 1
        assert results["ids"][0] == "c1"

    def test_reset(self, chroma):
        chroma.add_embeddings(
            ids=["c1"],
            embeddings=[[0.1, 0.2]],
            documents=["doc1"],
            metadatas=[{"paper_id": "t::1", "chunk_type": "abstract", "year": 2020, "venue": "acl", "chunk_index": 0, "title": "T"}],
        )
        assert chroma.count() == 1
        chroma.reset()
        assert chroma.count() == 0
