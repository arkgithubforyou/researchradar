"""Tests for ingestion base layer: PaperRecord and DataLoader."""

import pytest

from src.ingestion.base_loader import DataLoader, PaperRecord


class TestPaperRecord:
    def test_paper_id_format(self):
        paper = PaperRecord(
            source_id="P18-1001", source="hf_acl_ocl", title="Test",
            abstract="Abstract", authors=["Alice"], year=2018,
        )
        assert paper.paper_id() == "hf_acl_ocl::P18-1001"

    def test_paper_id_different_sources(self):
        hf = PaperRecord(source_id="P18-1001", source="hf_acl_ocl", title="T", abstract="A", authors=[], year=2018)
        acl = PaperRecord(source_id="2024.acl-long.1", source="acl_anthology", title="T", abstract="A", authors=[], year=2024)
        assert hf.paper_id() != acl.paper_id()
        assert "hf_acl_ocl::" in hf.paper_id()
        assert "acl_anthology::" in acl.paper_id()

    def test_has_full_text_true(self):
        paper = PaperRecord(
            source_id="x", source="test", title="T", abstract="A",
            authors=[], year=2020, full_text="Some full text content",
        )
        assert paper.has_full_text() is True

    def test_has_full_text_false_none(self):
        paper = PaperRecord(
            source_id="x", source="test", title="T", abstract="A",
            authors=[], year=2020, full_text=None,
        )
        assert paper.has_full_text() is False

    def test_has_full_text_false_empty(self):
        paper = PaperRecord(
            source_id="x", source="test", title="T", abstract="A",
            authors=[], year=2020, full_text="   ",
        )
        assert paper.has_full_text() is False

    def test_defaults(self):
        paper = PaperRecord(
            source_id="x", source="test", title="T", abstract="A",
            authors=[], year=2020,
        )
        assert paper.venue is None
        assert paper.volume is None
        assert paper.full_text is None
        assert paper.url is None
        assert paper.metadata == {}

    def test_metadata_storage(self):
        paper = PaperRecord(
            source_id="x", source="test", title="T", abstract="A",
            authors=[], year=2020, metadata={"numcitedby": 42},
        )
        assert paper.metadata["numcitedby"] == 42


class TestDataLoaderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            DataLoader()

    def test_concrete_subclass_works(self):
        class DummyLoader(DataLoader):
            @property
            def source_name(self) -> str:
                return "dummy"

            def load(self, **kwargs):
                return []

            def validate_source(self) -> bool:
                return True

        loader = DummyLoader()
        assert loader.source_name == "dummy"
        assert loader.validate_source() is True
        assert loader.load() == []
