"""SQLite storage layer — hand-written SQL, no ORM.

Handles schema creation, paper/author/chunk ingestion, and analytical queries.
All queries are parameterized (no f-strings for SQL).
"""

import logging
import sqlite3
from pathlib import Path

from src.ingestion.base_loader import PaperRecord

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,              -- source::source_id (e.g., "hf_acl_ocl::P18-1001")
    source_id TEXT NOT NULL,          -- original ID from source
    source TEXT NOT NULL,             -- "hf_acl_ocl", "acl_anthology", "arxiv"
    title TEXT NOT NULL,
    abstract TEXT,
    full_text TEXT,                   -- NULL for abstract-only papers
    year INTEGER,
    venue TEXT,                       -- "acl", "emnlp", "naacl", etc.
    volume TEXT,                      -- "long", "short", "findings", etc.
    url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    name TEXT NOT NULL,
    position INTEGER NOT NULL         -- author order (0-indexed)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    chunk_text TEXT NOT NULL,
    chunk_type TEXT,                  -- "abstract", "introduction", "method", "full_text"
    chunk_index INTEGER,             -- order within paper
    token_count INTEGER
);

CREATE TABLE IF NOT EXISTS methods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    method_name TEXT NOT NULL,
    method_type TEXT                  -- "model", "technique", "framework"
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    dataset_name TEXT NOT NULL,
    task_type TEXT                    -- "classification", "QA", "generation", etc.
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    task_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    topic_name TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_venue ON papers(venue);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_authors_paper ON authors(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_methods_name ON methods(method_name);
CREATE INDEX IF NOT EXISTS idx_methods_paper ON methods(paper_id);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(dataset_name);
CREATE INDEX IF NOT EXISTS idx_datasets_paper ON datasets(paper_id);
CREATE INDEX IF NOT EXISTS idx_tasks_paper ON tasks(paper_id);
CREATE INDEX IF NOT EXISTS idx_topics_paper ON topics(paper_id);
CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(topic_name);
"""


class SQLiteDB:
    """SQLite database interface for ResearchRadar."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def create_schema(self):
        """Create all tables and indexes."""
        conn = self.get_connection()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            logger.info("Database schema created at %s", self.db_path)
        finally:
            conn.close()

    # ── Paper ingestion ──────────────────────────────────────────────

    def insert_papers(self, papers: list[PaperRecord], batch_size: int = 500):
        """Insert papers and their authors into the database.

        Uses INSERT OR IGNORE to skip duplicates (by paper_id).
        """
        conn = self.get_connection()
        try:
            inserted = 0
            for i in range(0, len(papers), batch_size):
                batch = papers[i : i + batch_size]

                paper_rows = [
                    (
                        p.paper_id(),
                        p.source_id,
                        p.source,
                        p.title,
                        p.abstract,
                        p.full_text,
                        p.year,
                        p.venue,
                        p.volume,
                        p.url,
                    )
                    for p in batch
                ]
                conn.executemany(
                    """INSERT OR IGNORE INTO papers
                       (id, source_id, source, title, abstract, full_text,
                        year, venue, volume, url)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    paper_rows,
                )

                author_rows = []
                for p in batch:
                    for pos, name in enumerate(p.authors):
                        author_rows.append((p.paper_id(), name, pos))

                if author_rows:
                    conn.executemany(
                        """INSERT OR IGNORE INTO authors (paper_id, name, position)
                           VALUES (?, ?, ?)""",
                        author_rows,
                    )

                inserted += len(batch)
                conn.commit()

            logger.info("Inserted %d papers into SQLite", inserted)
        finally:
            conn.close()

    # ── Chunk operations ─────────────────────────────────────────────

    def insert_chunks(
        self, chunks: list[dict], batch_size: int = 1000
    ):
        """Insert chunks into the database.

        Each chunk dict should have: paper_id, chunk_text, chunk_type,
        chunk_index, token_count.
        """
        conn = self.get_connection()
        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                conn.executemany(
                    """INSERT INTO chunks
                       (paper_id, chunk_text, chunk_type, chunk_index, token_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    [
                        (
                            c["paper_id"],
                            c["chunk_text"],
                            c["chunk_type"],
                            c["chunk_index"],
                            c["token_count"],
                        )
                        for c in batch
                    ],
                )
            conn.commit()
            logger.info("Inserted %d chunks into SQLite", len(chunks))
        finally:
            conn.close()

    def get_all_chunks(self) -> list[dict]:
        """Retrieve all chunks with their paper metadata."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT c.id, c.paper_id, c.chunk_text, c.chunk_type,
                          c.chunk_index, c.token_count,
                          p.title, p.year, p.venue
                   FROM chunks c
                   JOIN papers p ON c.paper_id = p.id
                   ORDER BY c.paper_id, c.chunk_index"""
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_chunk_texts_and_ids(self) -> tuple[list[str], list[int]]:
        """Get all chunk texts and their IDs for BM25 index building."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                "SELECT id, chunk_text FROM chunks ORDER BY id"
            ).fetchall()
            ids = [row["id"] for row in rows]
            texts = [row["chunk_text"] for row in rows]
            return texts, ids
        finally:
            conn.close()

    # ── Enrichment operations ────────────────────────────────────────

    def insert_methods(self, paper_id: str, methods: list[dict]):
        """Insert extracted methods for a paper.

        Each method dict: {"method_name": str, "method_type": str | None}
        """
        conn = self.get_connection()
        try:
            conn.executemany(
                """INSERT INTO methods (paper_id, method_name, method_type)
                   VALUES (?, ?, ?)""",
                [(paper_id, m["method_name"], m.get("method_type")) for m in methods],
            )
            conn.commit()
        finally:
            conn.close()

    def insert_datasets(self, paper_id: str, datasets: list[dict]):
        """Insert extracted datasets for a paper.

        Each dataset dict: {"dataset_name": str, "task_type": str | None}
        """
        conn = self.get_connection()
        try:
            conn.executemany(
                """INSERT INTO datasets (paper_id, dataset_name, task_type)
                   VALUES (?, ?, ?)""",
                [(paper_id, d["dataset_name"], d.get("task_type")) for d in datasets],
            )
            conn.commit()
        finally:
            conn.close()

    def insert_tasks(self, paper_id: str, task_names: list[str]):
        """Insert extracted tasks for a paper."""
        conn = self.get_connection()
        try:
            conn.executemany(
                "INSERT INTO tasks (paper_id, task_name) VALUES (?, ?)",
                [(paper_id, t) for t in task_names],
            )
            conn.commit()
        finally:
            conn.close()

    def insert_topics(self, paper_id: str, topic_names: list[str]):
        """Insert extracted topics for a paper."""
        conn = self.get_connection()
        try:
            conn.executemany(
                "INSERT INTO topics (paper_id, topic_name) VALUES (?, ?)",
                [(paper_id, t) for t in topic_names],
            )
            conn.commit()
        finally:
            conn.close()

    # ── Query operations ─────────────────────────────────────────────

    def get_paper_count(self) -> int:
        conn = self.get_connection()
        try:
            return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        finally:
            conn.close()

    def get_chunk_count(self) -> int:
        conn = self.get_connection()
        try:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()

    def get_paper_by_id(self, paper_id: str) -> dict | None:
        """Fetch a single paper with its authors."""
        conn = self.get_connection()
        try:
            paper = conn.execute(
                "SELECT * FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
            if paper is None:
                return None

            authors = conn.execute(
                "SELECT name FROM authors WHERE paper_id = ? ORDER BY position",
                (paper_id,),
            ).fetchall()

            methods = conn.execute(
                "SELECT method_name, method_type FROM methods WHERE paper_id = ?",
                (paper_id,),
            ).fetchall()

            datasets = conn.execute(
                "SELECT dataset_name, task_type FROM datasets WHERE paper_id = ?",
                (paper_id,),
            ).fetchall()

            result = dict(paper)
            result["authors"] = [row["name"] for row in authors]
            result["methods"] = [dict(row) for row in methods]
            result["datasets"] = [dict(row) for row in datasets]
            return result
        finally:
            conn.close()

    def _browse_conditions(
        self,
        venue: str | None = None,
        volume: str | None = None,
        year: int | None = None,
        method: str | None = None,
        dataset: str | None = None,
        author: str | None = None,
    ) -> tuple[str, list]:
        """Build WHERE clause and params for paper browsing/counting."""
        conditions: list[str] = []
        params: list = []

        if venue:
            conditions.append("p.venue = ?")
            params.append(venue)
        if volume:
            conditions.append("p.volume = ?")
            params.append(volume)
        if year:
            conditions.append("p.year = ?")
            params.append(year)
        if method:
            conditions.append(
                "p.id IN (SELECT paper_id FROM methods WHERE method_name LIKE ?)"
            )
            params.append(f"%{method}%")
        if dataset:
            conditions.append(
                "p.id IN (SELECT paper_id FROM datasets WHERE dataset_name LIKE ?)"
            )
            params.append(f"%{dataset}%")
        if author:
            conditions.append(
                "p.id IN (SELECT paper_id FROM authors WHERE name LIKE ?)"
            )
            params.append(f"%{author}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params

    def browse_papers(
        self,
        venue: str | None = None,
        volume: str | None = None,
        year: int | None = None,
        method: str | None = None,
        dataset: str | None = None,
        author: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """Browse/filter papers with optional filters."""
        where_clause, params = self._browse_conditions(
            venue=venue, volume=volume, year=year,
            method=method, dataset=dataset, author=author,
        )
        params.extend([limit, offset])

        conn = self.get_connection()
        try:
            rows = conn.execute(
                f"""SELECT p.id, p.title, p.abstract, p.year, p.venue, p.url
                    FROM papers p
                    WHERE {where_clause}
                    ORDER BY p.year DESC, p.title
                    LIMIT ? OFFSET ?""",
                params,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def count_papers(
        self,
        venue: str | None = None,
        volume: str | None = None,
        year: int | None = None,
        method: str | None = None,
        dataset: str | None = None,
        author: str | None = None,
    ) -> int:
        """Count papers matching the given filters."""
        where_clause, params = self._browse_conditions(
            venue=venue, volume=volume, year=year,
            method=method, dataset=dataset, author=author,
        )
        conn = self.get_connection()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM papers p WHERE {where_clause}",
                params,
            ).fetchone()
            return row[0]
        finally:
            conn.close()

    # ── Analytical queries (trend analytics) ─────────────────────────

    def papers_per_venue(self) -> list[dict]:
        """Paper count per venue, aggregated across all years."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT venue, COUNT(*) as paper_count
                   FROM papers
                   WHERE venue IS NOT NULL
                   GROUP BY venue
                   ORDER BY paper_count DESC"""
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def papers_per_venue_per_year(self) -> list[dict]:
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT venue, year, COUNT(*) as paper_count
                   FROM papers
                   GROUP BY venue, year
                   ORDER BY year, venue"""
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def top_methods_by_year(self, top_n: int = 10) -> list[dict]:
        """Most popular methods per year using window functions."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT method_name, year, count, rank FROM (
                       SELECT m.method_name, p.year, COUNT(*) as count,
                              RANK() OVER (PARTITION BY p.year ORDER BY COUNT(*) DESC) as rank
                       FROM methods m
                       JOIN papers p ON m.paper_id = p.id
                       GROUP BY m.method_name, p.year
                   )
                   WHERE rank <= ?
                   ORDER BY year, rank""",
                (top_n,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def top_datasets_by_year(self, top_n: int = 10) -> list[dict]:
        """Most used datasets per year."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT dataset_name, year, count, rank FROM (
                       SELECT d.dataset_name, p.year, COUNT(*) as count,
                              RANK() OVER (PARTITION BY p.year ORDER BY COUNT(*) DESC) as rank
                       FROM datasets d
                       JOIN papers p ON d.paper_id = p.id
                       GROUP BY d.dataset_name, p.year
                   )
                   WHERE rank <= ?
                   ORDER BY year, rank""",
                (top_n,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def method_trend(self, method_name: str) -> list[dict]:
        """Track a specific method's adoption over time."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT p.year, COUNT(*) as paper_count
                   FROM methods m
                   JOIN papers p ON m.paper_id = p.id
                   WHERE m.method_name LIKE ?
                   GROUP BY p.year
                   ORDER BY p.year""",
                (f"%{method_name}%",),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def dataset_trend(self, dataset_name: str) -> list[dict]:
        """Track a specific dataset's usage over time."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT p.year, COUNT(*) as paper_count
                   FROM datasets d
                   JOIN papers p ON d.paper_id = p.id
                   WHERE d.dataset_name LIKE ?
                   GROUP BY p.year
                   ORDER BY p.year""",
                (f"%{dataset_name}%",),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_enrichment_stats(self) -> dict:
        """Get counts of unique enriched entities (distinct names)."""
        conn = self.get_connection()
        try:
            paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            method_count = conn.execute(
                "SELECT COUNT(DISTINCT method_name) FROM methods"
            ).fetchone()[0]
            dataset_count = conn.execute(
                "SELECT COUNT(DISTINCT dataset_name) FROM datasets"
            ).fetchone()[0]
            task_count = conn.execute(
                "SELECT COUNT(DISTINCT task_name) FROM tasks"
            ).fetchone()[0]
            topic_count = conn.execute(
                "SELECT COUNT(DISTINCT topic_name) FROM topics"
            ).fetchone()[0]
            papers_with_methods = conn.execute(
                "SELECT COUNT(DISTINCT paper_id) FROM methods"
            ).fetchone()[0]
            return {
                "total_papers": paper_count,
                "total_methods": method_count,
                "total_datasets": dataset_count,
                "total_tasks": task_count,
                "total_topics": topic_count,
                "papers_with_methods": papers_with_methods,
            }
        finally:
            conn.close()

    def get_authors_for_papers(self, paper_ids: list[str]) -> dict[str, list[str]]:
        """Batch-fetch authors for multiple papers.

        Returns:
            Dict mapping paper_id → list of author names (ordered by position).
        """
        if not paper_ids:
            return {}
        conn = self.get_connection()
        try:
            placeholders = ",".join("?" * len(paper_ids))
            rows = conn.execute(
                f"SELECT paper_id, name FROM authors "
                f"WHERE paper_id IN ({placeholders}) "
                f"ORDER BY paper_id, position",
                paper_ids,
            ).fetchall()
            result: dict[str, list[str]] = {}
            for row in rows:
                result.setdefault(row["paper_id"], []).append(row["name"])
            return result
        finally:
            conn.close()

    _ENTITY_TABLE_MAP = {
        "methods": ("methods", "method_name"),
        "datasets": ("datasets", "dataset_name"),
        "tasks": ("tasks", "task_name"),
        "topics": ("topics", "topic_name"),
    }

    def get_entity_list(
        self, entity_type: str, limit: int = 500
    ) -> list[dict]:
        """Get all unique entity names with their paper counts.

        Args:
            entity_type: One of "methods", "datasets", "tasks", "topics".
            limit: Maximum entries to return.

        Returns:
            List of dicts with keys: name, count. Sorted by count descending.
        """
        if entity_type not in self._ENTITY_TABLE_MAP:
            raise ValueError(f"Unknown entity type: {entity_type}")
        table, col = self._ENTITY_TABLE_MAP[entity_type]
        conn = self.get_connection()
        try:
            rows = conn.execute(
                f"SELECT {col} AS name, COUNT(*) AS count "
                f"FROM {table} GROUP BY {col} ORDER BY count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # ── Co-occurrence analytics ───────────────────────────────────────

    def method_dataset_cooccurrence(self, top_n: int = 20) -> list[dict]:
        """Find which methods and datasets are used together most often.

        Returns rows of: method_name, dataset_name, co_count, ranked by frequency.
        """
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT m.method_name, d.dataset_name, COUNT(*) as co_count
                   FROM methods m
                   JOIN datasets d ON m.paper_id = d.paper_id
                   GROUP BY m.method_name, d.dataset_name
                   ORDER BY co_count DESC
                   LIMIT ?""",
                (top_n,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def method_task_cooccurrence(self, top_n: int = 20) -> list[dict]:
        """Find which methods are applied to which tasks most often.

        Returns rows of: method_name, task_name, co_count, ranked by frequency.
        """
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT m.method_name, t.task_name, COUNT(*) as co_count
                   FROM methods m
                   JOIN tasks t ON m.paper_id = t.paper_id
                   GROUP BY m.method_name, t.task_name
                   ORDER BY co_count DESC
                   LIMIT ?""",
                (top_n,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # ── Author analytics ──────────────────────────────────────────────

    def top_authors(self, top_n: int = 20) -> list[dict]:
        """Most prolific authors by paper count."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT a.name, COUNT(DISTINCT a.paper_id) as paper_count
                   FROM authors a
                   GROUP BY a.name
                   ORDER BY paper_count DESC
                   LIMIT ?""",
                (top_n,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def author_collaboration_pairs(self, top_n: int = 20) -> list[dict]:
        """Most frequent co-author pairs.

        Returns rows of: author_a, author_b, shared_papers.
        """
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT a1.name as author_a, a2.name as author_b,
                          COUNT(DISTINCT a1.paper_id) as shared_papers
                   FROM authors a1
                   JOIN authors a2 ON a1.paper_id = a2.paper_id
                        AND a1.name < a2.name
                   GROUP BY a1.name, a2.name
                   ORDER BY shared_papers DESC
                   LIMIT ?""",
                (top_n,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # ── Task analytics ────────────────────────────────────────────────

    def top_tasks_by_year(self, top_n: int = 10) -> list[dict]:
        """Most popular tasks per year using window functions."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT task_name, year, count, rank FROM (
                       SELECT t.task_name, p.year, COUNT(*) as count,
                              RANK() OVER (PARTITION BY p.year ORDER BY COUNT(*) DESC) as rank
                       FROM tasks t
                       JOIN papers p ON t.paper_id = p.id
                       GROUP BY t.task_name, p.year
                   )
                   WHERE rank <= ?
                   ORDER BY year, rank""",
                (top_n,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def task_trend(self, task_name: str) -> list[dict]:
        """Track a specific task's popularity over time."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT p.year, COUNT(*) as paper_count
                   FROM tasks t
                   JOIN papers p ON t.paper_id = p.id
                   WHERE t.task_name LIKE ?
                   GROUP BY p.year
                   ORDER BY p.year""",
                (f"%{task_name}%",),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # ── Topic analytics ─────────────────────────────────────────────

    def top_topics_by_year(self, top_n: int = 10) -> list[dict]:
        """Most popular topics per year using window functions."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT topic_name, year, count, rank FROM (
                       SELECT tp.topic_name, p.year, COUNT(*) as count,
                              RANK() OVER (PARTITION BY p.year ORDER BY COUNT(*) DESC) as rank
                       FROM topics tp
                       JOIN papers p ON tp.paper_id = p.id
                       GROUP BY tp.topic_name, p.year
                   )
                   WHERE rank <= ?
                   ORDER BY year, rank""",
                (top_n,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def topic_trend(self, topic_name: str) -> list[dict]:
        """Track a specific topic's popularity over time."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT p.year, COUNT(*) as paper_count
                   FROM topics tp
                   JOIN papers p ON tp.paper_id = p.id
                   WHERE tp.topic_name LIKE ?
                   GROUP BY p.year
                   ORDER BY p.year""",
                (f"%{topic_name}%",),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # ── Venue analytics ───────────────────────────────────────────────

    def venue_method_profile(self, venue: str, top_n: int = 10) -> list[dict]:
        """Top methods at a specific venue (across all years)."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT m.method_name, COUNT(*) as paper_count
                   FROM methods m
                   JOIN papers p ON m.paper_id = p.id
                   WHERE p.venue = ?
                   GROUP BY m.method_name
                   ORDER BY paper_count DESC
                   LIMIT ?""",
                (venue, top_n),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def year_over_year_growth(self) -> list[dict]:
        """Paper count per year with year-over-year growth rate."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """SELECT year, paper_count,
                          LAG(paper_count) OVER (ORDER BY year) as prev_count,
                          CASE
                              WHEN LAG(paper_count) OVER (ORDER BY year) > 0
                              THEN ROUND(
                                  100.0 * (paper_count - LAG(paper_count) OVER (ORDER BY year))
                                  / LAG(paper_count) OVER (ORDER BY year), 1)
                              ELSE NULL
                          END as growth_pct
                   FROM (
                       SELECT year, COUNT(*) as paper_count
                       FROM papers
                       GROUP BY year
                   )
                   ORDER BY year"""
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
