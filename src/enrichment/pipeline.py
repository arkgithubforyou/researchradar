"""Entity enrichment pipeline.

Orchestrates LLM extraction with regex fallback, and persists results
to SQLite. Processes papers in batches with progress logging.

Supports automatic Groq → Ollama failover when Groq rate limits are hit.
"""

import logging
from dataclasses import dataclass

from src.enrichment.llm_extractor import extract_entities_llm
from src.enrichment.regex_extractor import extract_all_regex
from src.generation.llm_backend_base import LLMBackend
from src.storage.sqlite_db import SQLiteDB

logger = logging.getLogger(__name__)

# Groq SDK raises these on rate-limit / quota exhaustion.
try:
    from groq import RateLimitError as GroqRateLimitError
except ImportError:
    GroqRateLimitError = None


@dataclass
class EnrichmentStats:
    """Statistics from an enrichment run."""

    papers_processed: int = 0
    papers_skipped: int = 0
    llm_extractions: int = 0
    regex_fallbacks: int = 0
    llm_fallback_switches: int = 0
    total_methods: int = 0
    total_datasets: int = 0
    total_tasks: int = 0
    total_topics: int = 0


class EnrichmentPipeline:
    """Orchestrates entity extraction and storage.

    Strategy:
        1. Try LLM extraction (if backend provided and available).
        2. On Groq rate-limit, automatically switch to fallback_backend.
        3. Fall back to regex extraction if LLM fails or is unavailable.
        4. Merge results (LLM takes priority; regex fills gaps).
        5. Store in SQLite methods/datasets/tasks/topics tables.

    Usage:
        pipeline = EnrichmentPipeline(db, llm_backend=groq, fallback_backend=ollama)
        stats = pipeline.enrich_all(batch_size=50)
    """

    def __init__(
        self,
        db: SQLiteDB,
        llm_backend: LLMBackend | None = None,
        fallback_backend: LLMBackend | None = None,
        use_regex_fallback: bool = True,
    ):
        self.db = db
        self.llm = llm_backend
        self.fallback_llm = fallback_backend
        self.use_regex_fallback = use_regex_fallback
        self._llm_available: bool | None = None
        self._fell_back: bool = False

    def _check_llm(self) -> bool:
        """Check if the LLM backend is available (cached)."""
        if self.llm is None:
            return False
        if self._llm_available is None:
            try:
                self._llm_available = self.llm.is_available()
            except Exception:
                self._llm_available = False
            if self._llm_available:
                logger.info("LLM backend %r is available", self.llm.backend_name)
            else:
                logger.warning("LLM backend not available, using regex only")
        return self._llm_available

    def _switch_to_fallback(self) -> bool:
        """Switch to the fallback LLM backend.

        Returns True if successfully switched, False otherwise.
        """
        if self.fallback_llm is None:
            return False
        try:
            if not self.fallback_llm.is_available():
                logger.warning("Fallback backend %r not available", self.fallback_llm.backend_name)
                return False
        except Exception:
            logger.warning("Fallback backend %r check failed", self.fallback_llm.backend_name)
            return False

        logger.info(
            "Switching from %r to fallback backend %r",
            self.llm.backend_name if self.llm else "none",
            self.fallback_llm.backend_name,
        )
        self.llm = self.fallback_llm
        self.fallback_llm = None
        self._llm_available = True
        self._fell_back = True
        return True

    def _merge_extractions(
        self, llm_result: dict | None, regex_result: dict
    ) -> dict:
        """Merge LLM and regex extractions.

        LLM results take priority. Regex fills in entities that the LLM missed.
        Deduplication is by lowercased name.
        """
        if llm_result is None:
            return regex_result

        merged = {"methods": [], "datasets": [], "tasks": [], "topics": []}

        # Methods: LLM first, then regex additions
        seen_methods = set()
        for m in llm_result.get("methods", []):
            key = m["method_name"].lower()
            if key not in seen_methods:
                seen_methods.add(key)
                merged["methods"].append(m)
        for m in regex_result.get("methods", []):
            key = m["method_name"].lower()
            if key not in seen_methods:
                seen_methods.add(key)
                merged["methods"].append(m)

        # Datasets: LLM first, then regex additions
        seen_datasets = set()
        for d in llm_result.get("datasets", []):
            key = d["dataset_name"].lower()
            if key not in seen_datasets:
                seen_datasets.add(key)
                merged["datasets"].append(d)
        for d in regex_result.get("datasets", []):
            key = d["dataset_name"].lower()
            if key not in seen_datasets:
                seen_datasets.add(key)
                merged["datasets"].append(d)

        # Tasks: LLM first, then regex additions
        seen_tasks = set()
        for t in llm_result.get("tasks", []):
            key = t.lower()
            if key not in seen_tasks:
                seen_tasks.add(key)
                merged["tasks"].append(t)
        for t in regex_result.get("tasks", []):
            key = t.lower()
            if key not in seen_tasks:
                seen_tasks.add(key)
                merged["tasks"].append(t)

        # Topics: LLM first, then regex additions
        seen_topics = set()
        for t in llm_result.get("topics", []):
            key = t.lower()
            if key not in seen_topics:
                seen_topics.add(key)
                merged["topics"].append(t)
        for t in regex_result.get("topics", []):
            key = t.lower()
            if key not in seen_topics:
                seen_topics.add(key)
                merged["topics"].append(t)

        return merged

    def _extract_text(self, paper: dict) -> str:
        """Get the best available text for extraction.

        Prefers abstract (highest signal-to-noise) but appends title
        for extra context.
        """
        parts = []
        if paper.get("title"):
            parts.append(paper["title"])
        if paper.get("abstract"):
            parts.append(paper["abstract"])
        return "\n".join(parts)

    def enrich_paper(self, paper: dict) -> dict:
        """Extract entities from a single paper dict.

        Args:
            paper: Dict with at least "id", "title", "abstract" keys.

        Returns:
            Merged extraction result dict.
        """
        text = self._extract_text(paper)
        title = paper.get("title", "")

        llm_result = None
        if self._check_llm():
            llm_result = extract_entities_llm(
                self.llm, title=title, text=text
            )

        regex_result = extract_all_regex(text) if self.use_regex_fallback else {
            "methods": [], "datasets": [], "tasks": [], "topics": []
        }

        return self._merge_extractions(llm_result, regex_result)

    def _get_unenriched_papers(self) -> list[dict]:
        """Fetch papers that have no methods, datasets, tasks, or topics yet."""
        conn = self.db.get_connection()
        try:
            rows = conn.execute(
                """SELECT p.id, p.title, p.abstract
                   FROM papers p
                   WHERE p.id NOT IN (SELECT DISTINCT paper_id FROM methods)
                     AND p.id NOT IN (SELECT DISTINCT paper_id FROM datasets)
                     AND p.id NOT IN (SELECT DISTINCT paper_id FROM tasks)
                     AND p.id NOT IN (SELECT DISTINCT paper_id FROM topics)
                   ORDER BY p.year DESC"""
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def enrich_all(
        self,
        batch_size: int = 50,
        max_papers: int | None = None,
    ) -> EnrichmentStats:
        """Enrich all unenriched papers in the database.

        Args:
            batch_size: Log progress every N papers.
            max_papers: Stop after this many papers (for testing).

        Returns:
            EnrichmentStats with counts.
        """
        papers = self._get_unenriched_papers()
        if max_papers is not None:
            papers = papers[:max_papers]

        total = len(papers)
        if total == 0:
            logger.info("No unenriched papers found")
            return EnrichmentStats()

        logger.info("Enriching %d papers", total)
        stats = EnrichmentStats()

        for i, paper in enumerate(papers):
            paper_id = paper["id"]
            text = self._extract_text(paper)

            if not text.strip():
                stats.papers_skipped += 1
                continue

            # LLM extraction (with Groq → fallback auto-switch)
            llm_result = None
            if self._check_llm():
                try:
                    llm_result = extract_entities_llm(
                        self.llm,
                        title=paper.get("title", ""),
                        text=text,
                    )
                except Exception as exc:
                    if GroqRateLimitError is not None and isinstance(exc, GroqRateLimitError):
                        logger.warning(
                            "Groq rate limit hit after %d papers. %s",
                            stats.llm_extractions,
                            exc,
                        )
                        if self._switch_to_fallback():
                            stats.llm_fallback_switches += 1
                            # Retry this paper with fallback backend
                            llm_result = extract_entities_llm(
                                self.llm,
                                title=paper.get("title", ""),
                                text=text,
                            )
                        else:
                            logger.warning("No fallback available, using regex for remaining papers")
                            self._llm_available = False
                    else:
                        logger.warning("LLM extraction failed for %s: %s", paper_id, exc)
                if llm_result is not None:
                    stats.llm_extractions += 1

            # Regex extraction
            regex_result = (
                extract_all_regex(text)
                if self.use_regex_fallback
                else {"methods": [], "datasets": [], "tasks": []}
            )
            if llm_result is None and self.use_regex_fallback:
                stats.regex_fallbacks += 1

            # Merge and store
            merged = self._merge_extractions(llm_result, regex_result)

            if merged["methods"]:
                self.db.insert_methods(paper_id, merged["methods"])
                stats.total_methods += len(merged["methods"])
            if merged["datasets"]:
                self.db.insert_datasets(paper_id, merged["datasets"])
                stats.total_datasets += len(merged["datasets"])
            if merged["tasks"]:
                self.db.insert_tasks(paper_id, merged["tasks"])
                stats.total_tasks += len(merged["tasks"])
            if merged["topics"]:
                self.db.insert_topics(paper_id, merged["topics"])
                stats.total_topics += len(merged["topics"])

            stats.papers_processed += 1

            if (i + 1) % batch_size == 0:
                logger.info(
                    "Progress: %d/%d papers enriched", i + 1, total
                )

        logger.info(
            "Enrichment complete: %d processed, %d skipped, "
            "%d LLM, %d regex fallback, "
            "%d methods, %d datasets, %d tasks, %d topics",
            stats.papers_processed,
            stats.papers_skipped,
            stats.llm_extractions,
            stats.regex_fallbacks,
            stats.total_methods,
            stats.total_datasets,
            stats.total_tasks,
            stats.total_topics,
        )
        return stats
