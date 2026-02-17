"""FastAPI application factory.

Creates the app, registers routers, and wires up lifespan events.
Serves the React frontend in production (static files + SPA fallback).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.deps import get_db, init_components, is_initialized
from src.api.models import HealthResponse
from src.api.routes_analytics import router as analytics_router
from src.api.routes_papers import router as papers_router
from src.api.routes_search import router as search_router

logger = logging.getLogger(__name__)

# Frontend build directory — check both dev and Docker locations
_FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"
_DOCKER_FRONTEND = Path("/app/frontend/dist")


def _get_frontend_dir() -> Path | None:
    """Return the frontend dist directory if it exists."""
    if _DOCKER_FRONTEND.is_dir():
        return _DOCKER_FRONTEND
    if _FRONTEND_DIST.is_dir():
        return _FRONTEND_DIST
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy components on startup, cleanup on shutdown."""
    if not is_initialized():
        logger.info("Starting ResearchRadar API — loading models...")
        init_components()
        logger.info("Startup complete")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ResearchRadar",
        description="RAG-powered research paper search, analytics, and trend tracking.",
        version="0.6.0",
        lifespan=lifespan,
    )

    # CORS — allow frontend dev server during development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routers
    app.include_router(search_router, prefix="/api")
    app.include_router(papers_router, prefix="/api")
    app.include_router(analytics_router, prefix="/api")

    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    def health_check():
        db = get_db()
        return HealthResponse(
            status="ok",
            paper_count=db.get_paper_count(),
            chunk_count=db.get_chunk_count(),
        )

    # ── Static file serving for the React SPA ────────────────────────
    frontend_dir = _get_frontend_dir()
    if frontend_dir is not None:
        logger.info("Serving frontend from %s", frontend_dir)

        # Mount /assets (hashed JS/CSS bundles)
        assets_dir = frontend_dir / "assets"
        if assets_dir.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="assets",
            )

        # SPA fallback: any non-API route returns index.html
        index_html = frontend_dir / "index.html"

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(request: Request, full_path: str):
            # Serve static file if it exists (favicon.svg, etc.)
            static_file = frontend_dir / full_path
            if full_path and static_file.is_file():
                return FileResponse(str(static_file))
            # Otherwise serve index.html for client-side routing
            return FileResponse(str(index_html))
    else:
        logger.info("No frontend build found — API-only mode")

    return app


app = create_app()
