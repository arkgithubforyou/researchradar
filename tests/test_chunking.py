"""Tests for chunking strategies."""

from src.ingestion.chunking import (
    chunk_abstract_only,
    chunk_fixed_size,
    chunk_paper,
    chunk_section_aware,
)


def test_abstract_only_basic(sample_papers):
    paper = sample_papers[0]  # no full text
    chunks = chunk_abstract_only(paper)
    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "abstract"
    assert chunks[0]["paper_id"] == paper.paper_id()
    assert chunks[0]["chunk_index"] == 0
    assert "transformer" in chunks[0]["chunk_text"]


def test_abstract_only_empty():
    from src.ingestion.base_loader import PaperRecord

    paper = PaperRecord(
        source_id="test", source="test", title="Test",
        abstract="", authors=[], year=2020,
    )
    assert chunk_abstract_only(paper) == []


def test_fixed_size_with_full_text(sample_papers):
    paper = sample_papers[1]  # has full_text
    chunks = chunk_fixed_size(paper, chunk_size=20, overlap=5)
    assert len(chunks) >= 1
    assert all(c["chunk_type"] == "full_text" for c in chunks)
    # Chunks should be ordered
    indices = [c["chunk_index"] for c in chunks]
    assert indices == sorted(indices)


def test_fixed_size_falls_back_to_abstract(sample_papers):
    paper = sample_papers[0]  # no full text
    chunks = chunk_fixed_size(paper)
    assert len(chunks) >= 1


def test_section_aware_with_full_text(sample_papers):
    paper = sample_papers[1]  # has full_text with section headers
    chunks = chunk_section_aware(paper, min_chunk_size=3)
    assert len(chunks) >= 1


def test_section_aware_falls_back_without_full_text(sample_papers):
    paper = sample_papers[0]  # no full text
    chunks = chunk_section_aware(paper)
    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "abstract"


def test_chunk_paper_strategy_dispatch(sample_papers):
    paper = sample_papers[0]
    abstract_chunks = chunk_paper(paper, strategy="abstract")
    fixed_chunks = chunk_paper(paper, strategy="fixed")
    assert len(abstract_chunks) >= 1
    assert len(fixed_chunks) >= 1


def test_chunk_paper_invalid_strategy(sample_papers):
    import pytest
    with pytest.raises(ValueError, match="Unknown strategy"):
        chunk_paper(sample_papers[0], strategy="invalid")
