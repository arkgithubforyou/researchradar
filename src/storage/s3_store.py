"""AWS S3 storage for database snapshots.

Handles upload/download of SQLite and ChromaDB snapshots so deployments
can pull pre-built data from S3 instead of re-running ingestion.

Usage:
    store = S3Store(bucket="researchradar-data", ...)
    store.push_snapshot(sqlite_path, chroma_path)   # after ingestion
    store.pull_snapshot(sqlite_path, chroma_path)    # on deployment startup
"""

import logging
import shutil
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SNAPSHOT_PREFIX = "snapshots/"
SQLITE_KEY = f"{SNAPSHOT_PREFIX}researchradar.db"
CHROMA_KEY = f"{SNAPSHOT_PREFIX}chroma_db.zip"


class S3Store:
    """S3 interface for database snapshot storage."""

    def __init__(
        self,
        bucket: str,
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
        region: str = "us-east-1",
    ):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id or None,
            aws_secret_access_key=aws_secret_access_key or None,
            region_name=region,
        )

    # ── Low-level operations ─────────────────────────────────────────

    def upload_file(self, local_path: str | Path, s3_key: str):
        """Upload a local file to S3."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        self.client.upload_file(str(local_path), self.bucket, s3_key)
        logger.info("Uploaded %s → s3://%s/%s", local_path, self.bucket, s3_key)

    def download_file(self, s3_key: str, local_path: str | Path):
        """Download a file from S3 to local path."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, s3_key, str(local_path))
        logger.info("Downloaded s3://%s/%s → %s", self.bucket, s3_key, local_path)

    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def list_files(self, prefix: str = "") -> list[str]:
        """List files in the bucket with optional prefix."""
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    # ── Snapshot operations ──────────────────────────────────────────

    def push_snapshot(self, sqlite_path: str | Path, chroma_path: str | Path):
        """Upload SQLite DB and ChromaDB directory to S3.

        Call this after ingestion to make data available for deployment.
        ChromaDB directory is zipped before upload.
        """
        sqlite_path = Path(sqlite_path)
        chroma_path = Path(chroma_path)

        # Upload SQLite file directly
        self.upload_file(sqlite_path, SQLITE_KEY)

        # Zip and upload ChromaDB directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_base = Path(tmp_dir) / "chroma_db"
            zip_path = shutil.make_archive(str(zip_base), "zip", str(chroma_path))
            self.upload_file(zip_path, CHROMA_KEY)

        logger.info("Snapshot pushed to s3://%s/%s", self.bucket, SNAPSHOT_PREFIX)

    def pull_snapshot(self, sqlite_path: str | Path, chroma_path: str | Path):
        """Download SQLite DB and ChromaDB from S3.

        Call this on deployment startup to hydrate local data without
        re-running ingestion.
        """
        sqlite_path = Path(sqlite_path)
        chroma_path = Path(chroma_path)

        # Download SQLite
        self.download_file(SQLITE_KEY, sqlite_path)

        # Download and unzip ChromaDB
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "chroma_db.zip"
            self.download_file(CHROMA_KEY, zip_path)

            # Clear existing chroma dir and extract
            if chroma_path.exists():
                shutil.rmtree(chroma_path)
            chroma_path.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(zip_path), str(chroma_path))

        logger.info("Snapshot pulled from s3://%s/%s", self.bucket, SNAPSHOT_PREFIX)

    def snapshot_exists(self) -> bool:
        """Check if a snapshot is available on S3."""
        return self.file_exists(SQLITE_KEY) and self.file_exists(CHROMA_KEY)
