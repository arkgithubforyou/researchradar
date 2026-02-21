"""FastAPI dependency injection — shared component singletons.

Lazily initializes heavy components (embedding model, reranker, LLM backend,
retrieval pipeline, RAG engine) once, then provides them via FastAPI Depends().
"""

import logging

from src.config import Config, get_config
from src.generation.groq_backend import GroqBackend
from src.generation.llm_backend_base import LLMBackend
from src.generation.ollama_backend import OllamaBackend
from src.generation.rag_engine import RAGEngine
from src.ingestion.embeddings import EmbeddingGenerator
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.reranker import CrossEncoderReranker
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_db import SQLiteDB

logger = logging.getLogger(__name__)

# ── Singletons (populated by init_components) ───────────────────────

_config: Config | None = None
_db: SQLiteDB | None = None
_chroma: ChromaStore | None = None
_rag_engine: RAGEngine | None = None


def init_components(config: Config | None = None) -> None:
    """Initialize all heavy components. Call once at startup."""
    global _config, _db, _chroma, _rag_engine

    _config = config or get_config()
    _db = SQLiteDB(_config.sqlite_db_path)
    _db.create_schema()  # Ensure tables exist (idempotent)
    _chroma = ChromaStore(_config.chroma_db_path)

    # Embedding generator
    embed_gen = EmbeddingGenerator(_config.embedding_model)

    # Reranker
    reranker = CrossEncoderReranker(_config.reranker_model)

    # Retrieval pipeline
    pipeline = RetrievalPipeline(
        db=_db,
        chroma_store=_chroma,
        embedding_generator=embed_gen,
        reranker=reranker,
    )
    pipeline.build_index()

    # LLM backend
    llm: LLMBackend
    if _config.llm_backend == "groq" and _config.groq_api_key:
        llm = GroqBackend(api_key=_config.groq_api_key)
        logger.info("LLM backend: Groq (model=%s)", llm.model)
    else:
        llm = OllamaBackend(host=_config.ollama_host)
        if _config.llm_backend == "groq":
            logger.warning(
                "LLM_BACKEND=groq but GROQ_API_KEY is empty — "
                "falling back to Ollama at %s",
                _config.ollama_host,
            )
        else:
            logger.info("LLM backend: Ollama at %s", _config.ollama_host)

    _rag_engine = RAGEngine(pipeline, llm)
    logger.info("All components initialized")


def is_initialized() -> bool:
    """Check if components have been initialized (or mocked for testing)."""
    return _db is not None and _rag_engine is not None


def get_db() -> SQLiteDB:
    assert _db is not None, "Components not initialized — call init_components()"
    return _db


def get_rag_engine() -> RAGEngine:
    assert _rag_engine is not None, "Components not initialized — call init_components()"
    return _rag_engine
