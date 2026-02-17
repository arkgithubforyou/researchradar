"""Tests for config module."""

from pathlib import Path

from src.config import Config, get_config, PROJECT_ROOT


def test_config_defaults():
    config = Config()
    assert config.llm_backend == "ollama"
    assert config.embedding_model == "BAAI/bge-base-en-v1.5"
    assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.aws_s3_bucket == "researchradar-data"


def test_config_paths_are_path_objects():
    config = Config()
    assert isinstance(config.sqlite_db_path, Path)
    assert isinstance(config.chroma_db_path, Path)
    assert isinstance(config.data_dir, Path)


def test_config_default_paths_under_project():
    config = Config()
    assert "researchradar" in str(config.data_dir)


def test_get_config_creates_dirs(tmp_path):
    config = Config(data_dir=tmp_path / "testdata")
    config.ensure_dirs()
    assert (tmp_path / "testdata").exists()
    assert (tmp_path / "testdata" / "raw").exists()


def test_project_root_is_valid():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()
