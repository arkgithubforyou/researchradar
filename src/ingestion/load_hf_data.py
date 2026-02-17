"""HuggingFace ACL-OCL parquet data loader.

Loads papers from the WINGNUS/ACL-OCL dataset on HuggingFace.
This dataset contains ~74K papers through September 2022, with full text
extracted via GROBID.

Usage:
    loader = HFDataLoader(parquet_path="data/raw/acl-publication-info.74k.parquet")
    papers = loader.load(year_from=2018, year_to=2022, venues=["acl", "emnlp"])
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ingestion.base_loader import DataLoader, PaperRecord

logger = logging.getLogger(__name__)

# Major ACL venues — used for filtering by venue from the booktitle field
MAJOR_VENUES = {
    "acl": ["acl", "annual meeting of the association for computational linguistics"],
    "emnlp": ["emnlp", "empirical methods in natural language processing"],
    "naacl": ["naacl", "north american chapter"],
    "coling": ["coling", "international conference on computational linguistics"],
    "eacl": ["eacl", "european chapter"],
    "findings": ["findings"],
    "tacl": ["tacl", "transactions of the association"],
    "cl": ["computational linguistics"],
}


def _extract_venue(booktitle: str | None, acl_id: str | None) -> str | None:
    """Extract a normalized venue name from booktitle or acl_id.

    Tries acl_id first (e.g., "P18-1001" -> "acl", "D19-1234" -> "emnlp"),
    then falls back to booktitle keyword matching.
    """
    # ACL ID prefix mapping (legacy format)
    id_prefix_map = {
        "P": "acl",
        "D": "emnlp",
        "N": "naacl",
        "C": "coling",
        "E": "eacl",
        "Q": "tacl",
        "J": "cl",
        "W": "workshop",
        "S": "semeval",
    }

    if acl_id and len(acl_id) > 0:
        prefix = acl_id[0].upper()
        if prefix in id_prefix_map:
            return id_prefix_map[prefix]

        # New-style IDs: "2022.acl-long.220"
        if "." in acl_id:
            parts = acl_id.split(".")
            if len(parts) >= 2:
                venue_part = parts[1].split("-")[0].lower()
                return venue_part

    if booktitle:
        booktitle_lower = booktitle.lower()
        for venue, keywords in MAJOR_VENUES.items():
            if any(kw in booktitle_lower for kw in keywords):
                return venue

    return None


def _extract_volume(acl_id: str | None) -> str | None:
    """Extract volume type from acl_id (e.g., '2022.acl-long.220' -> 'long')."""
    if acl_id and "." in acl_id:
        parts = acl_id.split(".")
        if len(parts) >= 2 and "-" in parts[1]:
            return parts[1].split("-", 1)[1]
    return None


def _parse_authors(author_field) -> list[str]:
    """Parse the author field into a flat list of name strings.

    The HF dataset stores authors in various formats — this handles
    lists of dicts, lists of strings, and raw strings.
    """
    if author_field is None:
        return []

    if isinstance(author_field, str):
        # "First Last and First Last" or "First Last, First Last"
        if " and " in author_field:
            return [a.strip() for a in author_field.split(" and ") if a.strip()]
        return [a.strip() for a in author_field.split(",") if a.strip()]

    if isinstance(author_field, list):
        result = []
        for item in author_field:
            if isinstance(item, dict):
                first = item.get("first", "")
                last = item.get("last", "")
                name = f"{first} {last}".strip()
                if name:
                    result.append(name)
            elif isinstance(item, str):
                result.append(item.strip())
        return result

    return []


class HFDataLoader(DataLoader):
    """Load papers from the HuggingFace ACL-OCL parquet dataset.

    This dataset provides ~74K papers through 2022 with full text.
    Download from: https://huggingface.co/datasets/WINGNUS/ACL-OCL
    """

    def __init__(self, parquet_path: str | Path):
        self.parquet_path = Path(parquet_path)

    @property
    def source_name(self) -> str:
        return "hf_acl_ocl"

    def validate_source(self) -> bool:
        return self.parquet_path.exists()

    def load(
        self,
        year_from: int | None = None,
        year_to: int | None = None,
        venues: list[str] | None = None,
        max_papers: int | None = None,
    ) -> list[PaperRecord]:
        if not self.validate_source():
            raise FileNotFoundError(
                f"Parquet file not found: {self.parquet_path}\n"
                "Download from: https://huggingface.co/datasets/WINGNUS/ACL-OCL"
            )

        logger.info("Loading parquet file: %s", self.parquet_path)
        df = pd.read_parquet(self.parquet_path)
        logger.info("Loaded %d raw records", len(df))

        # Filter by year
        if year_from is not None:
            df = df[df["year"] >= year_from]
        if year_to is not None:
            df = df[df["year"] <= year_to]

        # Must have at least a title and abstract
        df = df.dropna(subset=["title", "abstract"])
        df = df[df["abstract"].str.strip().str.len() > 0]

        logger.info("%d records after year/null filtering", len(df))

        # Extract venue and filter
        df["_venue"] = df.apply(
            lambda row: _extract_venue(
                row.get("booktitle"), row.get("acl_id")
            ),
            axis=1,
        )

        if venues:
            venues_lower = [v.lower() for v in venues]
            df = df[df["_venue"].isin(venues_lower)]
            logger.info("%d records after venue filtering (%s)", len(df), venues_lower)

        # Apply max_papers limit
        if max_papers is not None and len(df) > max_papers:
            df = df.sample(n=max_papers, random_state=42)
            logger.info("Sampled %d papers", max_papers)

        # Convert to PaperRecord
        papers = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing papers"):
            full_text = row.get("full_text")
            if isinstance(full_text, str) and not full_text.strip():
                full_text = None

            paper = PaperRecord(
                source_id=str(row.get("acl_id", "")),
                source=self.source_name,
                title=str(row["title"]).strip(),
                abstract=str(row["abstract"]).strip(),
                authors=_parse_authors(row.get("author")),
                year=int(row["year"]),
                venue=row.get("_venue"),
                volume=_extract_volume(row.get("acl_id")),
                full_text=full_text,
                url=row.get("url"),
                metadata={
                    "corpus_paper_id": row.get("corpus_paper_id"),
                    "pdf_hash": row.get("pdf_hash"),
                    "booktitle": row.get("booktitle"),
                    "numcitedby": row.get("numcitedby"),
                },
            )
            papers.append(paper)

        logger.info("Produced %d PaperRecord objects from HF dataset", len(papers))
        return papers
