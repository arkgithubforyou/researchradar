"""Shared pytest fixtures for ResearchRadar tests."""

import tempfile
from pathlib import Path

import pytest

from src.ingestion.base_loader import PaperRecord
from src.storage.sqlite_db import SQLiteDB


@pytest.fixture
def sample_papers() -> list[PaperRecord]:
    """A small set of fake papers for testing."""
    return [
        PaperRecord(
            source_id="P18-1001",
            source="hf_acl_ocl",
            title="Attention Is All You Need (Not Really)",
            abstract="We propose a novel transformer architecture for machine translation.",
            authors=["Alice Smith", "Bob Jones"],
            year=2018,
            venue="acl",
            volume="long",
            full_text=None,
            url="https://example.com/paper1",
        ),
        PaperRecord(
            source_id="D19-1234",
            source="hf_acl_ocl",
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            abstract="We propose LoRA, a parameter-efficient fine-tuning method using low-rank matrices.",
            authors=["Carol Chen"],
            year=2019,
            venue="emnlp",
            volume="long",
            full_text="Introduction\nWe study parameter-efficient methods.\n\nMethod\nLoRA decomposes weight updates.",
            url="https://example.com/paper2",
        ),
        PaperRecord(
            source_id="2022.acl-long.100",
            source="acl_anthology",
            title="Contrastive Learning for NLP: A Survey",
            abstract="This paper surveys contrastive learning approaches applied to NLP tasks.",
            authors=["Dave Wilson", "Eve Taylor"],
            year=2022,
            venue="acl",
            volume="long",
        ),
    ]


@pytest.fixture
def tmp_db(tmp_path) -> SQLiteDB:
    """A temporary SQLite database for testing."""
    db = SQLiteDB(tmp_path / "test.db")
    db.create_schema()
    return db
