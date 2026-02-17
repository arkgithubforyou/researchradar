"""Chunking strategies for research papers.

Three approaches implemented for ablation study:
1. Abstract-only — one chunk per paper (baseline, works for all papers)
2. Fixed-size — 512 tokens, 50-token overlap (for full-text papers)
3. Section-aware — split on section boundaries (for full-text papers)
"""

import logging
import re

from src.ingestion.base_loader import PaperRecord

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (~1.3 words per token for English)."""
    return int(len(text.split()) * 1.3)


def chunk_abstract_only(paper: PaperRecord) -> list[dict]:
    """Strategy 1: One chunk per paper = the abstract.

    Highest signal-to-noise ratio. Works for all papers.
    """
    if not paper.abstract or not paper.abstract.strip():
        return []

    return [
        {
            "paper_id": paper.paper_id(),
            "chunk_text": paper.abstract.strip(),
            "chunk_type": "abstract",
            "chunk_index": 0,
            "token_count": _estimate_tokens(paper.abstract),
        }
    ]


def chunk_fixed_size(
    paper: PaperRecord,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict]:
    """Strategy 2: Fixed-size chunks with overlap.

    Splits full text (or abstract if no full text) into chunks of
    approximately `chunk_size` tokens with `overlap` token overlap.
    """
    text = paper.full_text if paper.has_full_text() else paper.abstract
    if not text or not text.strip():
        return []

    words = text.split()
    # Convert token counts to approximate word counts
    words_per_chunk = int(chunk_size / 1.3)
    words_overlap = int(overlap / 1.3)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append(
            {
                "paper_id": paper.paper_id(),
                "chunk_text": chunk_text,
                "chunk_type": "full_text",
                "chunk_index": chunk_index,
                "token_count": _estimate_tokens(chunk_text),
            }
        )

        if end >= len(words):
            break

        start = end - words_overlap
        chunk_index += 1

    return chunks


# Common section headers in GROBID-extracted text
SECTION_PATTERNS = [
    r"(?:^|\n)(?:\d+\.?\s*)?(?:introduction)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:related\s+work)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:background)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:method(?:ology|s)?)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:approach)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:model)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:experiment(?:s|al\s+setup)?)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:result(?:s)?(?:\s+and\s+discussion)?)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:discussion)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:analysis)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:conclusion(?:s)?)\s*\n",
    r"(?:^|\n)(?:\d+\.?\s*)?(?:abstract)\s*\n",
]

SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE)


def chunk_section_aware(paper: PaperRecord, min_chunk_size: int = 50) -> list[dict]:
    """Strategy 3: Split on section boundaries.

    Uses regex to detect section headers in GROBID-extracted text.
    Falls back to abstract-only if no full text or no sections found.
    """
    if not paper.has_full_text():
        return chunk_abstract_only(paper)

    text = paper.full_text
    splits = list(SECTION_REGEX.finditer(text))

    if not splits:
        # No section headers found — fall back to fixed-size
        return chunk_fixed_size(paper)

    chunks = []
    chunk_index = 0

    # Text before the first section header
    pre_text = text[: splits[0].start()].strip()
    if pre_text and len(pre_text.split()) >= min_chunk_size:
        chunks.append(
            {
                "paper_id": paper.paper_id(),
                "chunk_text": pre_text,
                "chunk_type": "preamble",
                "chunk_index": chunk_index,
                "token_count": _estimate_tokens(pre_text),
            }
        )
        chunk_index += 1

    # Each section
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        section_text = text[start:end].strip()

        if len(section_text.split()) < min_chunk_size:
            continue

        # Infer section type from the header
        header = match.group().strip().lower()
        section_type = "section"
        for keyword in ["introduction", "method", "approach", "model", "experiment",
                        "result", "discussion", "analysis", "conclusion",
                        "related work", "background", "abstract"]:
            if keyword in header:
                section_type = keyword.replace(" ", "_")
                break

        chunks.append(
            {
                "paper_id": paper.paper_id(),
                "chunk_text": section_text,
                "chunk_type": section_type,
                "chunk_index": chunk_index,
                "token_count": _estimate_tokens(section_text),
            }
        )
        chunk_index += 1

    if not chunks:
        return chunk_abstract_only(paper)

    return chunks


# Strategy registry for easy selection
CHUNKING_STRATEGIES = {
    "abstract": chunk_abstract_only,
    "fixed": chunk_fixed_size,
    "section": chunk_section_aware,
}


def chunk_paper(paper: PaperRecord, strategy: str = "abstract") -> list[dict]:
    """Chunk a paper using the specified strategy.

    Args:
        paper: The paper to chunk.
        strategy: One of "abstract", "fixed", "section".

    Returns:
        List of chunk dicts ready for SQLite insertion.
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Use: {list(CHUNKING_STRATEGIES)}")
    return CHUNKING_STRATEGIES[strategy](paper)


def chunk_papers(
    papers: list[PaperRecord], strategy: str = "abstract"
) -> list[dict]:
    """Chunk multiple papers using the specified strategy."""
    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunk_paper(paper, strategy))
    logger.info(
        "Chunked %d papers → %d chunks (strategy=%s)",
        len(papers), len(all_chunks), strategy,
    )
    return all_chunks
