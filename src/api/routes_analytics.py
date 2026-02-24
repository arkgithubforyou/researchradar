"""Analytics endpoints — trends, rankings, co-occurrence."""

import logging

from fastapi import APIRouter, Depends, Query

from src.api.deps import get_db
from src.api.models import (
    CooccurrenceRow,
    EnrichmentStatsResponse,
    EntityListItem,
    GrowthPoint,
    RankedEntity,
    TrendPoint,
    TrendRequest,
    TrendResponse,
)
from src.storage.sqlite_db import SQLiteDB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ── Trend endpoints ──────────────────────────────────────────────────


@router.post("/methods/trend", response_model=TrendResponse)
def method_trend(req: TrendRequest, db: SQLiteDB = Depends(get_db)):
    """Track a method's adoption over time."""
    rows = db.method_trend(req.name)
    return TrendResponse(
        name=req.name,
        trend=[TrendPoint(**r) for r in rows],
    )


@router.post("/datasets/trend", response_model=TrendResponse)
def dataset_trend(req: TrendRequest, db: SQLiteDB = Depends(get_db)):
    """Track a dataset's usage over time."""
    rows = db.dataset_trend(req.name)
    return TrendResponse(
        name=req.name,
        trend=[TrendPoint(**r) for r in rows],
    )


@router.post("/tasks/trend", response_model=TrendResponse)
def task_trend(req: TrendRequest, db: SQLiteDB = Depends(get_db)):
    """Track a task's popularity over time."""
    rows = db.task_trend(req.name)
    return TrendResponse(
        name=req.name,
        trend=[TrendPoint(**r) for r in rows],
    )


@router.post("/topics/trend", response_model=TrendResponse)
def topic_trend(req: TrendRequest, db: SQLiteDB = Depends(get_db)):
    """Track a topic's popularity over time."""
    rows = db.topic_trend(req.name)
    return TrendResponse(
        name=req.name,
        trend=[TrendPoint(**r) for r in rows],
    )


# ── Ranking endpoints ────────────────────────────────────────────────


@router.get("/methods/top", response_model=list[RankedEntity])
def top_methods(
    top_n: int = Query(default=10, ge=1, le=100),
    db: SQLiteDB = Depends(get_db),
):
    """Top methods per year."""
    rows = db.top_methods_by_year(top_n)
    return [
        RankedEntity(name=r["method_name"], year=r["year"], count=r["count"], rank=r["rank"])
        for r in rows
    ]


@router.get("/datasets/top", response_model=list[RankedEntity])
def top_datasets(
    top_n: int = Query(default=10, ge=1, le=100),
    db: SQLiteDB = Depends(get_db),
):
    """Top datasets per year."""
    rows = db.top_datasets_by_year(top_n)
    return [
        RankedEntity(name=r["dataset_name"], year=r["year"], count=r["count"], rank=r["rank"])
        for r in rows
    ]


@router.get("/tasks/top", response_model=list[RankedEntity])
def top_tasks(
    top_n: int = Query(default=10, ge=1, le=100),
    db: SQLiteDB = Depends(get_db),
):
    """Top tasks per year."""
    rows = db.top_tasks_by_year(top_n)
    return [
        RankedEntity(name=r["task_name"], year=r["year"], count=r["count"], rank=r["rank"])
        for r in rows
    ]


@router.get("/topics/top", response_model=list[RankedEntity])
def top_topics(
    top_n: int = Query(default=10, ge=1, le=100),
    db: SQLiteDB = Depends(get_db),
):
    """Top topics per year."""
    rows = db.top_topics_by_year(top_n)
    return [
        RankedEntity(name=r["topic_name"], year=r["year"], count=r["count"], rank=r["rank"])
        for r in rows
    ]


# ── Entity list endpoints ────────────────────────────────────────────

VALID_ENTITY_TYPES = {"methods", "datasets", "tasks", "topics"}


@router.get("/{entity_type}/list", response_model=list[EntityListItem])
def entity_list(
    entity_type: str,
    limit: int = Query(default=500, ge=1, le=1000),
    db: SQLiteDB = Depends(get_db),
):
    """List all unique entities of a type with their paper counts."""
    if entity_type not in VALID_ENTITY_TYPES:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown entity type: {entity_type}")
    rows = db.get_entity_list(entity_type, limit=limit)
    return [EntityListItem(**r) for r in rows]


# ── Co-occurrence endpoints ──────────────────────────────────────────


@router.get("/cooccurrence/method-dataset", response_model=list[CooccurrenceRow])
def method_dataset_cooccurrence(
    top_n: int = Query(default=20, ge=1, le=200),
    db: SQLiteDB = Depends(get_db),
):
    """Most frequent method-dataset pairings."""
    rows = db.method_dataset_cooccurrence(top_n)
    return [
        CooccurrenceRow(entity_a=r["method_name"], entity_b=r["dataset_name"], co_count=r["co_count"])
        for r in rows
    ]


@router.get("/cooccurrence/method-task", response_model=list[CooccurrenceRow])
def method_task_cooccurrence(
    top_n: int = Query(default=20, ge=1, le=200),
    db: SQLiteDB = Depends(get_db),
):
    """Most frequent method-task pairings."""
    rows = db.method_task_cooccurrence(top_n)
    return [
        CooccurrenceRow(entity_a=r["method_name"], entity_b=r["task_name"], co_count=r["co_count"])
        for r in rows
    ]


# ── Aggregate stats ──────────────────────────────────────────────────


@router.get("/stats", response_model=EnrichmentStatsResponse)
def enrichment_stats(db: SQLiteDB = Depends(get_db)):
    """Get enrichment entity counts."""
    return EnrichmentStatsResponse(**db.get_enrichment_stats())


@router.get("/growth", response_model=list[GrowthPoint])
def year_over_year_growth(db: SQLiteDB = Depends(get_db)):
    """Paper count per year with year-over-year growth rate."""
    rows = db.year_over_year_growth()
    return [GrowthPoint(**r) for r in rows]


@router.get("/venues", response_model=list[dict])
def papers_per_venue_per_year(db: SQLiteDB = Depends(get_db)):
    """Paper distribution by venue and year."""
    return db.papers_per_venue_per_year()


@router.get("/venues-total", response_model=list[dict])
def papers_per_venue_total(db: SQLiteDB = Depends(get_db)):
    """Paper count per venue, aggregated across all years."""
    return db.papers_per_venue()
