"""Tests for S3 store (unit tests — no real AWS calls)."""

import pytest

from src.storage.s3_store import S3Store, SQLITE_KEY, CHROMA_KEY, SNAPSHOT_PREFIX


def test_upload_file_missing_raises(tmp_path):
    store = S3Store(bucket="test-bucket")
    with pytest.raises(FileNotFoundError):
        store.upload_file(tmp_path / "nonexistent.txt", "key.txt")


def test_s3store_init():
    store = S3Store(
        bucket="my-bucket",
        aws_access_key_id="fake_key",
        aws_secret_access_key="fake_secret",
    )
    assert store.bucket == "my-bucket"


def test_snapshot_keys_use_prefix():
    assert SQLITE_KEY.startswith(SNAPSHOT_PREFIX)
    assert CHROMA_KEY.startswith(SNAPSHOT_PREFIX)


def test_push_snapshot_missing_sqlite_raises(tmp_path):
    store = S3Store(bucket="test-bucket")
    with pytest.raises(FileNotFoundError):
        store.push_snapshot(
            tmp_path / "nonexistent.db",
            tmp_path / "chroma",
        )
