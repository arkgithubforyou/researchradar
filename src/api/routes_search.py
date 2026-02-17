"""Search endpoint — RAG-powered question answering."""

import logging

from fastapi import APIRouter, Depends, Request

from src.api.deps import get_rag_engine
from src.api.models import SearchRequest, SearchResponse, SourcePaper
from src.api.rate_limit import search_limiter
from src.generation.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    request: Request,
    req: SearchRequest,
    engine: RAGEngine = Depends(get_rag_engine),
):
    """Answer a research question using RAG over the paper corpus."""
    search_limiter.check(request)
    # Build metadata filter from optional year/venue constraints
    where: dict | None = None
    filters = {}
    if req.year_min is not None:
        filters["year"] = {"$gte": req.year_min}
    if req.year_max is not None:
        if "year" in filters:
            # ChromaDB doesn't support compound $and on the same field in where,
            # so we use $and at the top level
            where = {"$and": [
                {"year": {"$gte": req.year_min}},
                {"year": {"$lte": req.year_max}},
            ]}
        else:
            filters["year"] = {"$lte": req.year_max}
    if req.venue:
        filters["venue"] = req.venue

    if where is None and filters:
        where = filters

    response = engine.query(question=req.query, top_k=req.top_k, where=where)

    return SearchResponse(
        answer=response.answer,
        sources=[SourcePaper(**s) for s in response.sources],
        model=response.model,
        usage=response.usage,
    )
