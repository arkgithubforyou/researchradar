"""CLI: Entity enrichment pipeline.

Extracts methods, datasets, tasks, and topics from papers using LLM + regex.
Run after ingestion to populate the enrichment tables.

When --backend groq is used, Ollama is automatically configured as a fallback.
If Groq rate limits are hit, the pipeline seamlessly switches to Ollama for
the remaining papers.

Usage:
    # Regex-only (fast, no LLM needed)
    python scripts/enrich.py --mode regex

    # LLM with regex fallback (default backend from .env)
    python scripts/enrich.py --mode llm

    # LLM with specific backend (auto-fallback Groq → Ollama)
    python scripts/enrich.py --mode llm --backend groq

    # Limit for testing
    python scripts/enrich.py --mode regex --max-papers 100

    # Show analytics after enrichment
    python scripts/enrich.py --mode regex --show-stats
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.enrichment.pipeline import EnrichmentPipeline
from src.generation.llm_backend_base import LLMBackend
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _make_backend(name: str, config) -> LLMBackend | None:
    """Instantiate a single LLM backend by name."""
    if name == "ollama":
        from src.generation.ollama_backend import OllamaBackend

        return OllamaBackend(host=config.ollama_host)
    elif name == "groq":
        from src.generation.groq_backend import GroqBackend

        if not config.groq_api_key:
            logger.warning("GROQ_API_KEY not set — cannot create Groq backend.")
            return None
        return GroqBackend(api_key=config.groq_api_key)
    else:
        raise ValueError(f"Unknown backend: {name}")


def get_llm_backends(args, config) -> tuple[LLMBackend | None, LLMBackend | None]:
    """Create primary + fallback LLM backends from CLI args and config.

    Returns:
        (primary_backend, fallback_backend).  fallback is None if primary
        is not Groq or if Ollama is unavailable.
    """
    if args.mode == "regex":
        return None, None

    backend = args.backend or config.llm_backend
    primary = _make_backend(backend, config)

    if primary is None:
        logger.error("Cannot create primary backend %r. Use --mode regex.", backend)
        sys.exit(1)

    # Auto-create Ollama fallback when primary is Groq
    fallback = None
    if backend == "groq":
        fallback = _make_backend("ollama", config)
        if fallback is not None:
            logger.info("Ollama fallback ready (will activate if Groq quota exhausted)")

    return primary, fallback


def show_stats(db: SQLiteDB):
    """Print enrichment statistics and sample analytics."""
    stats = db.get_enrichment_stats()
    print("\n=== Enrichment Statistics ===")
    print(f"  Total papers:       {stats['total_papers']}")
    print(f"  Papers with methods: {stats['papers_with_methods']}")
    print(f"  Total methods:      {stats['total_methods']}")
    print(f"  Total datasets:     {stats['total_datasets']}")
    print(f"  Total tasks:        {stats['total_tasks']}")
    print(f"  Total topics:       {stats['total_topics']}")

    # Top methods
    top_methods = db.top_methods_by_year(top_n=5)
    if top_methods:
        years = sorted(set(r["year"] for r in top_methods if r["year"]))
        if years:
            latest = years[-1]
            print(f"\n=== Top Methods ({latest}) ===")
            for r in top_methods:
                if r["year"] == latest:
                    print(f"  #{r['rank']}: {r['method_name']} ({r['count']} papers)")

    # Top co-occurrences
    cooc = db.method_dataset_cooccurrence(top_n=10)
    if cooc:
        print("\n=== Top Method-Dataset Pairs ===")
        for r in cooc[:10]:
            print(f"  {r['method_name']} + {r['dataset_name']}: {r['co_count']} papers")

    # Year-over-year growth
    growth = db.year_over_year_growth()
    if growth:
        print("\n=== Year-over-Year Growth ===")
        for r in growth:
            g = f"{r['growth_pct']:+.1f}%" if r["growth_pct"] is not None else "N/A"
            print(f"  {r['year']}: {r['paper_count']} papers ({g})")


def main():
    parser = argparse.ArgumentParser(description="ResearchRadar entity enrichment")
    parser.add_argument(
        "--mode",
        choices=["llm", "regex"],
        default="llm",
        help="Extraction mode: 'llm' (LLM + regex fallback) or 'regex' (regex only)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "groq"],
        default=None,
        help="LLM backend (default: from LLM_BACKEND env var)",
    )
    parser.add_argument("--max-papers", type=int, default=None, help="Limit papers to enrich")
    parser.add_argument("--batch-size", type=int, default=50, help="Progress log interval")
    parser.add_argument("--show-stats", action="store_true", help="Show analytics after enrichment")

    args = parser.parse_args()
    config = get_config()

    db = SQLiteDB(config.sqlite_db_path)
    db.create_schema()

    paper_count = db.get_paper_count()
    if paper_count == 0:
        logger.error("No papers in database. Run ingestion first (scripts/ingest.py).")
        sys.exit(1)

    logger.info("Database has %d papers", paper_count)

    # Build pipeline (Groq primary → Ollama fallback)
    primary, fallback = get_llm_backends(args, config)
    pipeline = EnrichmentPipeline(
        db=db,
        llm_backend=primary,
        fallback_backend=fallback,
        use_regex_fallback=True,
    )

    # Run enrichment
    logger.info("=== Starting enrichment (mode=%s) ===", args.mode)
    stats = pipeline.enrich_all(
        batch_size=args.batch_size,
        max_papers=args.max_papers,
    )

    logger.info("=== Enrichment complete ===")
    logger.info(
        "Processed: %d | Skipped: %d | LLM: %d | Regex fallback: %d",
        stats.papers_processed,
        stats.papers_skipped,
        stats.llm_extractions,
        stats.regex_fallbacks,
    )
    logger.info(
        "Extracted: %d methods | %d datasets | %d tasks | %d topics",
        stats.total_methods,
        stats.total_datasets,
        stats.total_tasks,
        stats.total_topics,
    )

    if args.show_stats:
        show_stats(db)


if __name__ == "__main__":
    main()
