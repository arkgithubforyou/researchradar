"""Base data layer: canonical PaperRecord schema and DataLoader interface.

All data sources (HuggingFace, ACL Anthology, arXiv, etc.) normalize their
raw data into PaperRecord instances. Everything downstream — SQLite ingestion,
chunking, embedding — only sees PaperRecord.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaperRecord:
    """Canonical schema for a research paper, source-agnostic.

    Fields that a source doesn't provide should be left as None/empty.
    The `source` field tracks provenance so we can trace data lineage.
    """

    source_id: str  # original ID from source (acl_id, arxiv_id, etc.)
    source: str  # "hf_acl_ocl", "acl_anthology", "arxiv"
    title: str
    abstract: str
    authors: list[str]  # ["First Last", "First Last", ...]
    year: int
    venue: str | None = None  # "acl", "emnlp", "naacl", etc.
    volume: str | None = None  # "long", "short", "findings", etc.
    full_text: str | None = None  # only available from some sources
    url: str | None = None  # PDF or landing page URL
    metadata: dict[str, Any] = field(default_factory=dict)  # source-specific extras

    def has_full_text(self) -> bool:
        """Check if full text is available and non-empty."""
        return bool(self.full_text and self.full_text.strip())

    def paper_id(self) -> str:
        """Generate a unique paper ID across sources.

        Format: {source}::{source_id}
        Examples:
            hf_acl_ocl::P18-1001
            acl_anthology::2024.acl-long.1
            arxiv::2301.12345
        """
        return f"{self.source}::{self.source_id}"


class DataLoader(ABC):
    """Abstract interface for loading papers from any data source.

    Each source implements its own loader that:
    1. Fetches/reads raw data from the source
    2. Normalizes it into list[PaperRecord]
    3. Applies optional filtering (year range, venues, sample size)

    Usage:
        loader = HFDataLoader()
        papers = loader.load(year_from=2018, year_to=2022, venues=["acl", "emnlp"])
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source identifier (e.g., 'hf_acl_ocl')."""
        ...

    @abstractmethod
    def load(
        self,
        year_from: int | None = None,
        year_to: int | None = None,
        venues: list[str] | None = None,
        max_papers: int | None = None,
    ) -> list[PaperRecord]:
        """Load papers from this source, applying optional filters.

        Args:
            year_from: Minimum publication year (inclusive).
            year_to: Maximum publication year (inclusive).
            venues: List of venue names to include (lowercase). None = all.
            max_papers: Maximum number of papers to return. None = all.

        Returns:
            List of PaperRecord instances normalized to the canonical schema.
        """
        ...

    @abstractmethod
    def validate_source(self) -> bool:
        """Check if the data source is accessible.

        Returns True if the source data can be reached (file exists,
        API is reachable, package is installed, etc.).
        """
        ...
