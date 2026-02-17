"""Tests for the Phase 5 FastAPI API layer.

Covers all endpoints: health, search, papers, and analytics.
All tests use mocks — no model loading, no real database, no LLM calls.
Uses httpx + FastAPI TestClient pattern.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.models import (
    CooccurrenceRow,
    EnrichmentStatsResponse,
    GrowthPoint,
    HealthResponse,
    PaperDetail,
    PaperListResponse,
    PaperSummary,
    RankedEntity,
    SearchResponse,
    SourcePaper,
    TrendPoint,
    TrendResponse,
)
from src.generation.rag_engine import RAGEngine, RAGResponse
from src.storage.sqlite_db import SQLiteDB


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """A mock SQLiteDB for dependency injection."""
    db = MagicMock(spec=SQLiteDB)
    db.get_paper_count.return_value = 42
    db.get_chunk_count.return_value = 100
    return db


@pytest.fixture
def mock_rag_engine():
    """A mock RAG engine that returns a canned response."""
    engine = MagicMock(spec=RAGEngine)
    engine.query.return_value = RAGResponse(
        answer="Transformers use self-attention [1].",
        sources=[
            {
                "paper_id": "hf_acl_ocl::P18-1001",
                "title": "Attention Is All You Need",
                "year": 2018,
                "venue": "acl",
                "chunk_type": "abstract",
            }
        ],
        model="mock-model",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )
    return engine


@pytest.fixture
def client(mock_db, mock_rag_engine):
    """TestClient with mocked dependencies."""
    app = create_app()

    # Override dependency injection
    from src.api import deps

    deps._db = mock_db
    deps._rag_engine = mock_rag_engine

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # Reset singletons
    deps._db = None
    deps._rag_engine = None


# ── Health endpoint ──────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_ok(self, client, mock_db):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["paper_count"] == 42
        assert data["chunk_count"] == 100

    def test_health_response_model(self, client):
        resp = client.get("/api/health")
        health = HealthResponse(**resp.json())
        assert health.status == "ok"


# ── Search endpoint ──────────────────────────────────────────────────


class TestSearchEndpoint:
    def test_search_basic(self, client, mock_rag_engine):
        resp = client.post("/api/search", json={"query": "What is attention?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["sources"]) == 1
        assert data["model"] == "mock-model"
        mock_rag_engine.query.assert_called_once()

    def test_search_with_top_k(self, client, mock_rag_engine):
        resp = client.post("/api/search", json={"query": "test", "top_k": 3})
        assert resp.status_code == 200
        call_kwargs = mock_rag_engine.query.call_args
        assert call_kwargs.kwargs["top_k"] == 3

    def test_search_with_year_filter(self, client, mock_rag_engine):
        resp = client.post("/api/search", json={
            "query": "test", "year_min": 2020, "year_max": 2023,
        })
        assert resp.status_code == 200
        call_kwargs = mock_rag_engine.query.call_args
        where = call_kwargs.kwargs["where"]
        assert where is not None

    def test_search_with_venue_filter(self, client, mock_rag_engine):
        resp = client.post("/api/search", json={
            "query": "test", "venue": "acl",
        })
        assert resp.status_code == 200
        call_kwargs = mock_rag_engine.query.call_args
        where = call_kwargs.kwargs["where"]
        assert where["venue"] == "acl"

    def test_search_empty_query_rejected(self, client):
        resp = client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422

    def test_search_top_k_bounds(self, client):
        resp = client.post("/api/search", json={"query": "test", "top_k": 0})
        assert resp.status_code == 422
        resp = client.post("/api/search", json={"query": "test", "top_k": 51})
        assert resp.status_code == 422

    def test_search_response_model(self, client):
        resp = client.post("/api/search", json={"query": "test"})
        search_resp = SearchResponse(**resp.json())
        assert isinstance(search_resp.sources[0], SourcePaper)

    def test_search_year_min_only(self, client, mock_rag_engine):
        resp = client.post("/api/search", json={
            "query": "test", "year_min": 2020,
        })
        assert resp.status_code == 200
        where = mock_rag_engine.query.call_args.kwargs["where"]
        assert where["year"]["$gte"] == 2020


# ── Papers endpoints ─────────────────────────────────────────────────


class TestPapersEndpoints:
    def test_browse_papers_default(self, client, mock_db):
        mock_db.browse_papers.return_value = [
            {
                "id": "test::1", "title": "Paper A", "abstract": "Abstract A",
                "year": 2020, "venue": "acl", "url": "http://example.com",
            },
        ]
        resp = client.get("/api/papers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["limit"] == 20
        assert data["offset"] == 0

    def test_browse_papers_with_filters(self, client, mock_db):
        mock_db.browse_papers.return_value = []
        resp = client.get("/api/papers", params={
            "venue": "emnlp", "year": 2022, "limit": 5, "offset": 10,
        })
        assert resp.status_code == 200
        mock_db.browse_papers.assert_called_once_with(
            venue="emnlp", year=2022, method=None, dataset=None,
            limit=5, offset=10,
        )

    def test_browse_papers_with_method_filter(self, client, mock_db):
        mock_db.browse_papers.return_value = []
        resp = client.get("/api/papers", params={"method": "BERT"})
        assert resp.status_code == 200
        mock_db.browse_papers.assert_called_once()
        call_kwargs = mock_db.browse_papers.call_args.kwargs
        assert call_kwargs["method"] == "BERT"

    def test_get_paper_found(self, client, mock_db):
        mock_db.get_paper_by_id.return_value = {
            "id": "test::1", "title": "Paper A", "abstract": "Abstract",
            "year": 2020, "venue": "acl", "url": None,
            "authors": ["Alice", "Bob"],
            "methods": [{"method_name": "BERT", "method_type": "model"}],
            "datasets": [{"dataset_name": "SQuAD", "task_type": "QA"}],
        }
        resp = client.get("/api/papers/test::1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test::1"
        assert data["authors"] == ["Alice", "Bob"]
        assert len(data["methods"]) == 1

    def test_get_paper_not_found(self, client, mock_db):
        mock_db.get_paper_by_id.return_value = None
        resp = client.get("/api/papers/nonexistent::999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_paper_detail_model(self, client, mock_db):
        mock_db.get_paper_by_id.return_value = {
            "id": "test::1", "title": "T", "abstract": "A",
            "year": 2020, "venue": "acl", "url": None,
            "authors": [], "methods": [], "datasets": [],
        }
        resp = client.get("/api/papers/test::1")
        detail = PaperDetail(**resp.json())
        assert detail.id == "test::1"


# ── Analytics endpoints ──────────────────────────────────────────────


class TestAnalyticsEndpoints:
    def test_method_trend(self, client, mock_db):
        mock_db.method_trend.return_value = [
            {"year": 2020, "paper_count": 10},
            {"year": 2021, "paper_count": 25},
        ]
        resp = client.post("/api/analytics/methods/trend", json={"name": "BERT"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "BERT"
        assert len(data["trend"]) == 2
        mock_db.method_trend.assert_called_once_with("BERT")

    def test_dataset_trend(self, client, mock_db):
        mock_db.dataset_trend.return_value = [{"year": 2021, "paper_count": 5}]
        resp = client.post("/api/analytics/datasets/trend", json={"name": "SQuAD"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "SQuAD"

    def test_task_trend(self, client, mock_db):
        mock_db.task_trend.return_value = []
        resp = client.post("/api/analytics/tasks/trend", json={"name": "QA"})
        assert resp.status_code == 200
        assert resp.json()["trend"] == []

    def test_topic_trend(self, client, mock_db):
        mock_db.topic_trend.return_value = [{"year": 2022, "paper_count": 8}]
        resp = client.post("/api/analytics/topics/trend", json={"name": "transformers"})
        assert resp.status_code == 200

    def test_top_methods(self, client, mock_db):
        mock_db.top_methods_by_year.return_value = [
            {"method_name": "BERT", "year": 2020, "count": 50, "rank": 1},
        ]
        resp = client.get("/api/analytics/methods/top", params={"top_n": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "BERT"

    def test_top_datasets(self, client, mock_db):
        mock_db.top_datasets_by_year.return_value = [
            {"dataset_name": "SQuAD", "year": 2020, "count": 30, "rank": 1},
        ]
        resp = client.get("/api/analytics/datasets/top")
        assert resp.status_code == 200
        assert resp.json()[0]["name"] == "SQuAD"

    def test_top_tasks(self, client, mock_db):
        mock_db.top_tasks_by_year.return_value = []
        resp = client.get("/api/analytics/tasks/top")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_top_topics(self, client, mock_db):
        mock_db.top_topics_by_year.return_value = [
            {"topic_name": "NLP", "year": 2021, "count": 20, "rank": 1},
        ]
        resp = client.get("/api/analytics/topics/top")
        assert resp.status_code == 200

    def test_method_dataset_cooccurrence(self, client, mock_db):
        mock_db.method_dataset_cooccurrence.return_value = [
            {"method_name": "BERT", "dataset_name": "SQuAD", "co_count": 15},
        ]
        resp = client.get("/api/analytics/cooccurrence/method-dataset")
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["entity_a"] == "BERT"
        assert data[0]["entity_b"] == "SQuAD"

    def test_method_task_cooccurrence(self, client, mock_db):
        mock_db.method_task_cooccurrence.return_value = [
            {"method_name": "GPT", "task_name": "generation", "co_count": 10},
        ]
        resp = client.get("/api/analytics/cooccurrence/method-task")
        assert resp.status_code == 200

    def test_enrichment_stats(self, client, mock_db):
        mock_db.get_enrichment_stats.return_value = {
            "total_papers": 100, "total_methods": 200,
            "total_datasets": 50, "total_tasks": 80,
            "total_topics": 120, "papers_with_methods": 90,
        }
        resp = client.get("/api/analytics/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_papers"] == 100

    def test_growth(self, client, mock_db):
        mock_db.year_over_year_growth.return_value = [
            {"year": 2020, "paper_count": 50, "prev_count": None, "growth_pct": None},
            {"year": 2021, "paper_count": 75, "prev_count": 50, "growth_pct": 50.0},
        ]
        resp = client.get("/api/analytics/growth")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[1]["growth_pct"] == 50.0

    def test_venue_paper_distribution(self, client, mock_db):
        mock_db.papers_per_venue_per_year.return_value = [
            {"venue": "acl", "year": 2020, "paper_count": 30},
        ]
        resp = client.get("/api/analytics/venues")
        assert resp.status_code == 200


# ── Trend request validation ─────────────────────────────────────────


class TestTrendRequestValidation:
    def test_empty_name_rejected(self, client):
        resp = client.post("/api/analytics/methods/trend", json={"name": ""})
        assert resp.status_code == 422


# ── Pydantic model unit tests ───────────────────────────────────────


class TestPydanticModels:
    def test_source_paper(self):
        sp = SourcePaper(
            paper_id="test::1", title="T", year=2020,
            venue="acl", chunk_type="abstract",
        )
        assert sp.paper_id == "test::1"

    def test_search_response(self):
        sr = SearchResponse(
            answer="test", sources=[], model="m", usage={},
        )
        assert sr.answer == "test"

    def test_trend_response(self):
        tr = TrendResponse(
            name="BERT",
            trend=[TrendPoint(year=2020, paper_count=10)],
        )
        assert len(tr.trend) == 1

    def test_paper_list_response(self):
        plr = PaperListResponse(
            papers=[PaperSummary(
                id="t::1", title="T", abstract=None,
                year=2020, venue=None, url=None,
            )],
            count=1, limit=20, offset=0,
        )
        assert plr.count == 1

    def test_health_response(self):
        hr = HealthResponse(status="ok", paper_count=10, chunk_count=20)
        assert hr.status == "ok"
