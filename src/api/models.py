"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


# ── Request models ───────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """RAG search query."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    year_min: int | None = Field(default=None, ge=1900, le=2100)
    year_max: int | None = Field(default=None, ge=1900, le=2100)
    venue: str | None = None


class PaperBrowseRequest(BaseModel):
    """Filters for browsing papers."""

    venue: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)
    method: str | None = None
    dataset: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class TrendRequest(BaseModel):
    """Request for tracking a specific entity trend over time."""

    name: str = Field(..., min_length=1, max_length=500)


# ── Response models ──────────────────────────────────────────────────


class SourcePaper(BaseModel):
    """A paper cited as a source in a RAG response."""

    paper_id: str
    title: str
    year: int
    venue: str | None
    chunk_type: str


class SearchResponse(BaseModel):
    """RAG search result."""

    answer: str
    sources: list[SourcePaper]
    model: str
    usage: dict = Field(default_factory=dict)


class PaperDetail(BaseModel):
    """Full paper with authors and enrichment data."""

    id: str
    title: str
    abstract: str | None
    year: int | None
    venue: str | None
    url: str | None
    authors: list[str] = Field(default_factory=list)
    methods: list[dict] = Field(default_factory=list)
    datasets: list[dict] = Field(default_factory=list)


class PaperSummary(BaseModel):
    """Paper listing with basic metadata."""

    id: str
    title: str
    abstract: str | None
    year: int | None
    venue: str | None
    url: str | None


class PaperListResponse(BaseModel):
    """Paginated paper listing."""

    papers: list[PaperSummary]
    count: int
    limit: int
    offset: int


class TrendPoint(BaseModel):
    """A single year's data point in a trend."""

    year: int
    paper_count: int


class TrendResponse(BaseModel):
    """Trend data for an entity over time."""

    name: str
    trend: list[TrendPoint]


class RankedEntity(BaseModel):
    """An entity (method/dataset/task/topic) with count and rank."""

    name: str
    year: int
    count: int
    rank: int


class CooccurrenceRow(BaseModel):
    """A co-occurrence pair with frequency."""

    entity_a: str
    entity_b: str
    co_count: int


class EnrichmentStatsResponse(BaseModel):
    """Database enrichment statistics."""

    total_papers: int
    total_methods: int
    total_datasets: int
    total_tasks: int
    total_topics: int
    papers_with_methods: int


class GrowthPoint(BaseModel):
    """Year-over-year growth data."""

    year: int
    paper_count: int
    prev_count: int | None
    growth_pct: float | None


class HealthResponse(BaseModel):
    """System health status."""

    status: str
    paper_count: int
    chunk_count: int
