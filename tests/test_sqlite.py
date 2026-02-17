"""Tests for SQLite storage layer."""

import pytest

from src.ingestion.base_loader import PaperRecord
from src.storage.sqlite_db import SQLiteDB


class TestSQLiteSchema:
    def test_create_schema(self, tmp_db):
        # Schema should be created without error
        assert tmp_db.get_paper_count() == 0
        assert tmp_db.get_chunk_count() == 0

    def test_create_schema_idempotent(self, tmp_db):
        # Calling create_schema again should not error
        tmp_db.create_schema()
        assert tmp_db.get_paper_count() == 0


class TestPaperIngestion:
    def test_insert_papers(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        assert tmp_db.get_paper_count() == 3

    def test_insert_duplicate_papers_ignored(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        tmp_db.insert_papers(sample_papers)  # same papers again
        assert tmp_db.get_paper_count() == 3

    def test_get_paper_by_id(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        paper = tmp_db.get_paper_by_id("hf_acl_ocl::P18-1001")
        assert paper is not None
        assert paper["title"] == "Attention Is All You Need (Not Really)"
        assert paper["year"] == 2018
        assert paper["venue"] == "acl"
        assert len(paper["authors"]) == 2
        assert "Alice Smith" in paper["authors"]

    def test_get_paper_by_id_not_found(self, tmp_db):
        assert tmp_db.get_paper_by_id("nonexistent") is None

    def test_paper_sources_stored(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        p1 = tmp_db.get_paper_by_id("hf_acl_ocl::P18-1001")
        p2 = tmp_db.get_paper_by_id("acl_anthology::2022.acl-long.100")
        assert p1["source"] == "hf_acl_ocl"
        assert p2["source"] == "acl_anthology"


class TestChunkOperations:
    def test_insert_and_count_chunks(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        chunks = [
            {
                "paper_id": "hf_acl_ocl::P18-1001",
                "chunk_text": "We propose a novel transformer architecture.",
                "chunk_type": "abstract",
                "chunk_index": 0,
                "token_count": 8,
            },
            {
                "paper_id": "hf_acl_ocl::P18-1001",
                "chunk_text": "The model uses self-attention.",
                "chunk_type": "method",
                "chunk_index": 1,
                "token_count": 6,
            },
        ]
        tmp_db.insert_chunks(chunks)
        assert tmp_db.get_chunk_count() == 2

    def test_get_all_chunks_joined(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        chunks = [
            {
                "paper_id": "hf_acl_ocl::P18-1001",
                "chunk_text": "Transformer abstract text.",
                "chunk_type": "abstract",
                "chunk_index": 0,
                "token_count": 4,
            },
        ]
        tmp_db.insert_chunks(chunks)
        results = tmp_db.get_all_chunks()
        assert len(results) == 1
        assert results[0]["title"] == "Attention Is All You Need (Not Really)"
        assert results[0]["year"] == 2018
        assert results[0]["venue"] == "acl"

    def test_get_chunk_texts_and_ids(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        chunks = [
            {"paper_id": "hf_acl_ocl::P18-1001", "chunk_text": "Text A", "chunk_type": "abstract", "chunk_index": 0, "token_count": 2},
            {"paper_id": "hf_acl_ocl::D19-1234", "chunk_text": "Text B", "chunk_type": "abstract", "chunk_index": 0, "token_count": 2},
        ]
        tmp_db.insert_chunks(chunks)
        texts, ids = tmp_db.get_chunk_texts_and_ids()
        assert len(texts) == 2
        assert len(ids) == 2
        assert "Text A" in texts
        assert "Text B" in texts


class TestBrowsePapers:
    def test_browse_all(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        results = tmp_db.browse_papers()
        assert len(results) == 3

    def test_browse_by_venue(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        results = tmp_db.browse_papers(venue="acl")
        assert len(results) == 2
        assert all(r["venue"] == "acl" for r in results)

    def test_browse_by_year(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        results = tmp_db.browse_papers(year=2018)
        assert len(results) == 1
        assert results[0]["year"] == 2018

    def test_browse_limit_offset(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        results = tmp_db.browse_papers(limit=1, offset=0)
        assert len(results) == 1


class TestEnrichmentOperations:
    def test_insert_and_query_methods(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        pid = "hf_acl_ocl::P18-1001"
        tmp_db.insert_methods(pid, [
            {"method_name": "transformer", "method_type": "model"},
            {"method_name": "self-attention", "method_type": "technique"},
        ])
        paper = tmp_db.get_paper_by_id(pid)
        assert len(paper["methods"]) == 2

    def test_insert_and_query_datasets(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        pid = "hf_acl_ocl::P18-1001"
        tmp_db.insert_datasets(pid, [
            {"dataset_name": "WMT14", "task_type": "translation"},
        ])
        paper = tmp_db.get_paper_by_id(pid)
        assert len(paper["datasets"]) == 1
        assert paper["datasets"][0]["dataset_name"] == "WMT14"

    def test_insert_tasks(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        pid = "hf_acl_ocl::P18-1001"
        tmp_db.insert_tasks(pid, ["machine translation", "language modeling"])
        # Tasks are not returned by get_paper_by_id, just verify no error
        assert tmp_db.get_paper_count() == 3

    def test_insert_and_query_topics(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        pid = "hf_acl_ocl::P18-1001"
        tmp_db.insert_topics(pid, ["multimodal", "fairness"])
        # Verify no error and count
        stats = tmp_db.get_enrichment_stats()
        assert stats["total_topics"] == 2

    def test_enrichment_stats(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        pid = "hf_acl_ocl::P18-1001"
        tmp_db.insert_methods(pid, [{"method_name": "BERT", "method_type": "model"}])
        tmp_db.insert_datasets(pid, [{"dataset_name": "SQuAD", "task_type": "QA"}])
        tmp_db.insert_tasks(pid, ["question answering"])
        tmp_db.insert_topics(pid, ["multimodal"])
        stats = tmp_db.get_enrichment_stats()
        assert stats["total_papers"] == 3
        assert stats["total_methods"] == 1
        assert stats["total_datasets"] == 1
        assert stats["total_tasks"] == 1
        assert stats["total_topics"] == 1
        assert stats["papers_with_methods"] == 1


class TestAnalytics:
    def test_papers_per_venue_per_year(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        results = tmp_db.papers_per_venue_per_year()
        assert len(results) > 0
        # Should have venue and year keys
        assert "venue" in results[0]
        assert "year" in results[0]
        assert "paper_count" in results[0]

    def test_top_methods_by_year(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        tmp_db.insert_methods("hf_acl_ocl::P18-1001", [{"method_name": "BERT", "method_type": "model"}])
        tmp_db.insert_methods("hf_acl_ocl::D19-1234", [{"method_name": "BERT", "method_type": "model"}])
        results = tmp_db.top_methods_by_year(top_n=5)
        assert len(results) > 0
        assert results[0]["method_name"] == "BERT"

    def test_method_trend(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        tmp_db.insert_methods("hf_acl_ocl::P18-1001", [{"method_name": "BERT", "method_type": "model"}])
        results = tmp_db.method_trend("BERT")
        assert len(results) == 1
        assert results[0]["year"] == 2018
        assert results[0]["paper_count"] == 1

    def test_method_trend_empty(self, tmp_db):
        results = tmp_db.method_trend("nonexistent")
        assert results == []

    def test_top_datasets_by_year(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        tmp_db.insert_datasets("hf_acl_ocl::P18-1001", [{"dataset_name": "SQuAD", "task_type": "QA"}])
        tmp_db.insert_datasets("hf_acl_ocl::D19-1234", [{"dataset_name": "SQuAD", "task_type": "QA"}])
        results = tmp_db.top_datasets_by_year(top_n=5)
        assert len(results) > 0
        assert results[0]["dataset_name"] == "SQuAD"

    def test_dataset_trend(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        tmp_db.insert_datasets("hf_acl_ocl::P18-1001", [{"dataset_name": "GLUE", "task_type": "NLI"}])
        results = tmp_db.dataset_trend("GLUE")
        assert len(results) == 1
        assert results[0]["year"] == 2018


class TestCooccurrenceAnalytics:
    def _setup_enrichment(self, db, sample_papers):
        """Helper to populate enrichment tables for testing."""
        db.insert_papers(sample_papers)
        # Paper 1: BERT + SQuAD + QA
        db.insert_methods("hf_acl_ocl::P18-1001", [
            {"method_name": "BERT", "method_type": "model"},
            {"method_name": "Transformer", "method_type": "model"},
        ])
        db.insert_datasets("hf_acl_ocl::P18-1001", [
            {"dataset_name": "SQuAD", "task_type": "QA"},
        ])
        db.insert_tasks("hf_acl_ocl::P18-1001", ["machine translation"])
        db.insert_topics("hf_acl_ocl::P18-1001", ["multimodal", "low-resource"])

        # Paper 2: BERT + LoRA + GLUE
        db.insert_methods("hf_acl_ocl::D19-1234", [
            {"method_name": "BERT", "method_type": "model"},
            {"method_name": "LoRA", "method_type": "technique"},
        ])
        db.insert_datasets("hf_acl_ocl::D19-1234", [
            {"dataset_name": "GLUE", "task_type": "NLI"},
            {"dataset_name": "SQuAD", "task_type": "QA"},
        ])
        db.insert_tasks("hf_acl_ocl::D19-1234", ["text classification"])
        db.insert_topics("hf_acl_ocl::D19-1234", ["low-resource", "efficiency"])

        # Paper 3: contrastive learning + no datasets
        db.insert_methods("acl_anthology::2022.acl-long.100", [
            {"method_name": "contrastive learning", "method_type": "technique"},
        ])
        db.insert_tasks("acl_anthology::2022.acl-long.100", ["text classification"])
        db.insert_topics("acl_anthology::2022.acl-long.100", ["multimodal"])

    def test_method_dataset_cooccurrence(self, tmp_db, sample_papers):
        self._setup_enrichment(tmp_db, sample_papers)
        results = tmp_db.method_dataset_cooccurrence(top_n=10)
        assert len(results) > 0
        # BERT+SQuAD should appear (co-occurs in 2 papers)
        top = results[0]
        assert "method_name" in top
        assert "dataset_name" in top
        assert "co_count" in top
        assert top["co_count"] >= 1

    def test_method_task_cooccurrence(self, tmp_db, sample_papers):
        self._setup_enrichment(tmp_db, sample_papers)
        results = tmp_db.method_task_cooccurrence(top_n=10)
        assert len(results) > 0
        assert "method_name" in results[0]
        assert "task_name" in results[0]
        assert "co_count" in results[0]

    def test_top_authors(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        results = tmp_db.top_authors(top_n=5)
        assert len(results) > 0
        assert "name" in results[0]
        assert "paper_count" in results[0]

    def test_author_collaboration_pairs(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        results = tmp_db.author_collaboration_pairs(top_n=5)
        # Paper 1 has 2 authors, paper 3 has 2 authors → at least 2 pairs
        assert len(results) >= 2
        assert "author_a" in results[0]
        assert "author_b" in results[0]
        assert "shared_papers" in results[0]

    def test_top_tasks_by_year(self, tmp_db, sample_papers):
        self._setup_enrichment(tmp_db, sample_papers)
        results = tmp_db.top_tasks_by_year(top_n=5)
        assert len(results) > 0
        assert "task_name" in results[0]
        assert "year" in results[0]
        assert "count" in results[0]

    def test_task_trend(self, tmp_db, sample_papers):
        self._setup_enrichment(tmp_db, sample_papers)
        results = tmp_db.task_trend("text classification")
        assert len(results) > 0
        assert "year" in results[0]
        assert "paper_count" in results[0]

    def test_task_trend_empty(self, tmp_db):
        results = tmp_db.task_trend("nonexistent_task")
        assert results == []

    def test_venue_method_profile(self, tmp_db, sample_papers):
        self._setup_enrichment(tmp_db, sample_papers)
        results = tmp_db.venue_method_profile("acl", top_n=5)
        assert len(results) > 0
        assert "method_name" in results[0]
        assert "paper_count" in results[0]

    def test_venue_method_profile_empty(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        results = tmp_db.venue_method_profile("nonexistent", top_n=5)
        assert results == []

    def test_top_topics_by_year(self, tmp_db, sample_papers):
        self._setup_enrichment(tmp_db, sample_papers)
        results = tmp_db.top_topics_by_year(top_n=5)
        assert len(results) > 0
        assert "topic_name" in results[0]
        assert "year" in results[0]
        assert "count" in results[0]

    def test_topic_trend(self, tmp_db, sample_papers):
        self._setup_enrichment(tmp_db, sample_papers)
        results = tmp_db.topic_trend("multimodal")
        assert len(results) > 0
        assert "year" in results[0]
        assert "paper_count" in results[0]

    def test_topic_trend_empty(self, tmp_db):
        results = tmp_db.topic_trend("nonexistent_topic")
        assert results == []

    def test_year_over_year_growth(self, tmp_db, sample_papers):
        tmp_db.insert_papers(sample_papers)
        results = tmp_db.year_over_year_growth()
        assert len(results) > 0
        assert "year" in results[0]
        assert "paper_count" in results[0]
        # First year has no previous → growth_pct is None
        assert results[0]["growth_pct"] is None
        # Subsequent years should have growth_pct
        if len(results) > 1:
            assert "growth_pct" in results[1]
