"""Paper endpoints — browse and fetch paper details."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_db
from src.api.models import (
    PaperBrowseRequest,
    PaperDetail,
    PaperListResponse,
    PaperSummary,
)
from src.storage.sqlite_db import SQLiteDB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("", response_model=PaperListResponse)
def browse_papers(
    venue: str | None = None,
    volume: str | None = None,
    year: int | None = None,
    method: str | None = None,
    dataset: str | None = None,
    author: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: SQLiteDB = Depends(get_db),
):
    """Browse papers with optional filters."""
    req = PaperBrowseRequest(
        venue=venue, volume=volume, year=year, method=method,
        dataset=dataset, author=author, limit=limit, offset=offset,
    )
    filter_kwargs = dict(
        venue=req.venue, volume=req.volume, year=req.year,
        method=req.method, dataset=req.dataset, author=req.author,
    )
    total = db.count_papers(**filter_kwargs)
    rows = db.browse_papers(**filter_kwargs, limit=req.limit, offset=req.offset)
    papers = [PaperSummary(**r) for r in rows]
    return PaperListResponse(
        papers=papers, count=total, limit=req.limit, offset=req.offset,
    )


@router.get("/{paper_id:path}", response_model=PaperDetail)
def get_paper(paper_id: str, db: SQLiteDB = Depends(get_db)):
    """Get full details for a single paper, including authors and enrichment."""
    paper = db.get_paper_by_id(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperDetail(
        id=paper["id"],
        title=paper["title"],
        abstract=paper.get("abstract"),
        year=paper.get("year"),
        venue=paper.get("venue"),
        url=paper.get("url"),
        authors=paper.get("authors", []),
        methods=paper.get("methods", []),
        datasets=paper.get("datasets", []),
    )
