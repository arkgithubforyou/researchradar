"""Tests for HuggingFace data loader (unit tests, no real data needed)."""

import pytest

from src.ingestion.load_hf_data import (
    _extract_venue,
    _extract_volume,
    _parse_authors,
    HFDataLoader,
)


class TestExtractVenue:
    def test_legacy_acl_id(self):
        assert _extract_venue(None, "P18-1001") == "acl"

    def test_legacy_emnlp_id(self):
        assert _extract_venue(None, "D19-1234") == "emnlp"

    def test_legacy_naacl_id(self):
        assert _extract_venue(None, "N19-1001") == "naacl"

    def test_legacy_coling_id(self):
        assert _extract_venue(None, "C18-1001") == "coling"

    def test_new_style_id(self):
        assert _extract_venue(None, "2022.acl-long.220") == "acl"

    def test_new_style_emnlp(self):
        assert _extract_venue(None, "2022.emnlp-main.50") == "emnlp"

    def test_booktitle_fallback(self):
        assert _extract_venue("Proceedings of the Annual Meeting of the Association for Computational Linguistics", None) == "acl"

    def test_booktitle_emnlp(self):
        assert _extract_venue("Empirical Methods in Natural Language Processing", None) == "emnlp"

    def test_unknown_returns_none(self):
        assert _extract_venue(None, None) is None

    def test_unknown_booktitle(self):
        assert _extract_venue("Some Random Workshop", None) is None


class TestExtractVolume:
    def test_long(self):
        assert _extract_volume("2022.acl-long.220") == "long"

    def test_short(self):
        assert _extract_volume("2022.acl-short.10") == "short"

    def test_findings(self):
        assert _extract_volume("2022.findings-emnlp.50") == "emnlp"

    def test_no_volume(self):
        assert _extract_volume("P18-1001") is None

    def test_none_input(self):
        assert _extract_volume(None) is None


class TestParseAuthors:
    def test_list_of_dicts(self):
        authors = [{"first": "Alice", "last": "Smith"}, {"first": "Bob", "last": "Jones"}]
        result = _parse_authors(authors)
        assert result == ["Alice Smith", "Bob Jones"]

    def test_list_of_strings(self):
        result = _parse_authors(["Alice Smith", "Bob Jones"])
        assert result == ["Alice Smith", "Bob Jones"]

    def test_and_separated_string(self):
        result = _parse_authors("Alice Smith and Bob Jones")
        assert result == ["Alice Smith", "Bob Jones"]

    def test_comma_separated_string(self):
        result = _parse_authors("Alice Smith, Bob Jones")
        assert result == ["Alice Smith", "Bob Jones"]

    def test_none_input(self):
        assert _parse_authors(None) == []

    def test_empty_list(self):
        assert _parse_authors([]) == []


class TestHFDataLoaderValidation:
    def test_validate_source_missing_file(self, tmp_path):
        loader = HFDataLoader(tmp_path / "nonexistent.parquet")
        assert loader.validate_source() is False

    def test_source_name(self):
        loader = HFDataLoader("dummy.parquet")
        assert loader.source_name == "hf_acl_ocl"

    def test_load_missing_file_raises(self, tmp_path):
        loader = HFDataLoader(tmp_path / "nonexistent.parquet")
        with pytest.raises(FileNotFoundError, match="Parquet file not found"):
            loader.load()
