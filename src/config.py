"""Central configuration for ResearchRadar."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root is the researchradar/ directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # LLM
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    llm_backend: str = field(
        default_factory=lambda: os.getenv("LLM_BACKEND", "ollama")
    )

    # AWS S3
    aws_access_key_id: str = field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", "")
    )
    aws_secret_access_key: str = field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", "")
    )
    aws_s3_bucket: str = field(
        default_factory=lambda: os.getenv("AWS_S3_BUCKET", "researchradar-data")
    )

    # Models
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    # Storage paths
    sqlite_db_path: Path = field(default=None)
    chroma_db_path: Path = field(default=None)
    bm25_index_path: Path = field(default=None)

    # Data
    data_dir: Path = field(default=None)

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = PROJECT_ROOT / "data"
        if self.sqlite_db_path is None:
            self.sqlite_db_path = Path(
                os.getenv("SQLITE_DB_PATH", str(self.data_dir / "researchradar.db"))
            )
        if self.chroma_db_path is None:
            self.chroma_db_path = Path(
                os.getenv("CHROMA_DB_PATH", str(self.data_dir / "chroma_db"))
            )
        if self.bm25_index_path is None:
            self.bm25_index_path = Path(
                os.getenv("BM25_INDEX_PATH", str(self.data_dir / "bm25_index.pkl"))
            )

    def ensure_dirs(self):
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get a Config instance. Call this instead of constructing directly."""
    config = Config()
    config.ensure_dirs()
    return config
