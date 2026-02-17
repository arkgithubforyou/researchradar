"""ACL Anthology official package data loader.

Loads papers from the official `acl-anthology` Python package (v0.5+).
This provides access to ~120K papers up to the present, but with
abstracts only (no full text).

Usage:
    loader = ACLAnthologyLoader()
    papers = loader.load(year_from=2023, year_to=2025, venues=["acl", "emnlp"])
"""

import logging

from tqdm import tqdm

from src.ingestion.base_loader import DataLoader, PaperRecord

logger = logging.getLogger(__name__)

# Map internal ACL Anthology venue IDs to our normalized venue names
VENUE_ID_MAP = {
    "acl": "acl",
    "emnlp": "emnlp",
    "naacl": "naacl",
    "coling": "coling",
    "eacl": "eacl",
    "findings": "findings",
    "tacl": "tacl",
    "cl": "cl",
    "semeval": "semeval",
    "conll": "conll",
}


def _normalize_venue_ids(venue_ids: list[str]) -> str | None:
    """Map ACL Anthology venue_ids to a single normalized venue name.

    Papers can be associated with multiple venue IDs. We pick the most
    specific one that matches our known venues.
    """
    for vid in venue_ids:
        vid_lower = vid.lower()
        if vid_lower in VENUE_ID_MAP:
            return VENUE_ID_MAP[vid_lower]
    # If no exact match, return the first one lowercased
    if venue_ids:
        return venue_ids[0].lower()
    return None


def _extract_volume_from_id(full_id: str) -> str | None:
    """Extract volume type from a full anthology ID.

    E.g., '2024.acl-long.1' -> 'long', '2024.findings-emnlp.50' -> 'emnlp'
    """
    parts = full_id.split(".")
    if len(parts) >= 2 and "-" in parts[1]:
        return parts[1].split("-", 1)[1]
    return None


class ACLAnthologyLoader(DataLoader):
    """Load papers from the official ACL Anthology Python package.

    Requires: pip install acl-anthology>=0.5.0
    On first use, calls Anthology.from_repo() which clones ~120 MB of
    metadata from the official GitHub repo (requires git installed).
    """

    def __init__(self):
        self._anthology = None

    @property
    def source_name(self) -> str:
        return "acl_anthology"

    def validate_source(self) -> bool:
        try:
            import acl_anthology  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_anthology(self):
        """Lazy-load the Anthology object (expensive first call)."""
        if self._anthology is None:
            from acl_anthology import Anthology

            logger.info("Loading ACL Anthology from repo (this may take a minute)...")
            self._anthology = Anthology.from_repo()
            logger.info("ACL Anthology loaded successfully")
        return self._anthology

    def load(
        self,
        year_from: int | None = None,
        year_to: int | None = None,
        venues: list[str] | None = None,
        max_papers: int | None = None,
    ) -> list[PaperRecord]:
        if not self.validate_source():
            raise ImportError(
                "acl-anthology package not installed. "
                "Run: pip install acl-anthology>=0.5.0"
            )

        anthology = self._get_anthology()
        venues_lower = [v.lower() for v in venues] if venues else None

        papers = []
        skipped = 0

        for paper in tqdm(anthology.papers(), desc="Loading ACL Anthology papers"):
            # Year filter
            try:
                year = int(paper.year)
            except (ValueError, TypeError):
                skipped += 1
                continue

            if year_from is not None and year < year_from:
                continue
            if year_to is not None and year > year_to:
                continue

            # Must have abstract
            abstract = paper.abstract
            if abstract is None:
                skipped += 1
                continue
            abstract_str = str(abstract).strip()
            if not abstract_str:
                skipped += 1
                continue

            # Venue filter
            paper_venue_ids = list(paper.venue_ids) if paper.venue_ids else []
            venue = _normalize_venue_ids(paper_venue_ids)

            if venues_lower and venue not in venues_lower:
                continue

            # Extract authors
            authors = []
            if paper.authors:
                for namespec in paper.authors:
                    name = namespec.name
                    full_name = f"{name.first} {name.last}".strip()
                    if full_name:
                        authors.append(full_name)

            # Build PDF URL
            pdf_url = None
            if paper.pdf is not None:
                pdf_url = paper.pdf.url

            full_id = str(paper.full_id)

            record = PaperRecord(
                source_id=full_id,
                source=self.source_name,
                title=str(paper.title).strip(),
                abstract=abstract_str,
                authors=authors,
                year=year,
                venue=venue,
                volume=_extract_volume_from_id(full_id),
                full_text=None,  # ACL Anthology does not provide full text
                url=pdf_url or paper.web_url,
                metadata={
                    "bibkey": paper.bibkey,
                    "doi": paper.doi,
                    "venue_ids": paper_venue_ids,
                },
            )
            papers.append(record)

            if max_papers is not None and len(papers) >= max_papers:
                logger.info("Reached max_papers limit (%d)", max_papers)
                break

        logger.info(
            "Produced %d PaperRecord objects from ACL Anthology (skipped %d)",
            len(papers),
            skipped,
        )
        return papers
