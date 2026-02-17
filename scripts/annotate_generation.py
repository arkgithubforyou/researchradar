"""CLI: Generation quality annotation tool.

For each annotated question in eval_set.json, runs the full RAG pipeline
and presents the generated answer for human scoring.

Scores:
    - Faithfulness: y/n (does the answer match the retrieved context?)
    - Relevance: 1-5 (how well does it address the question?)
    - Citation accuracy: y/n (do [1], [2] markers support the claims?)

Usage:
    python scripts/annotate_generation.py
    python scripts/annotate_generation.py --backend groq
    python scripts/annotate_generation.py --force   # re-score already scored
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROJECT_ROOT, get_config
from src.generation.llm_backend_base import LLMBackend
from src.generation.rag_engine import RAGEngine
from src.ingestion.embeddings import EmbeddingGenerator
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.reranker import CrossEncoderReranker
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval_set.json"


def load_eval_set() -> list[dict]:
    if EVAL_SET_PATH.exists():
        with open(EVAL_SET_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_eval_set(eval_set: list[dict]) -> None:
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)


def make_llm_backend(backend_name: str, config) -> LLMBackend:
    if backend_name == "groq":
        from src.generation.groq_backend import GroqBackend

        if not config.groq_api_key:
            logger.error("GROQ_API_KEY not set")
            sys.exit(1)
        return GroqBackend(api_key=config.groq_api_key)
    else:
        from src.generation.ollama_backend import OllamaBackend

        return OllamaBackend(host=config.ollama_host)


def init_rag_engine(config, backend_name: str) -> RAGEngine:
    db = SQLiteDB(config.sqlite_db_path)
    chroma = ChromaStore(config.chroma_db_path)
    embed_gen = EmbeddingGenerator(config.embedding_model)
    reranker = CrossEncoderReranker(config.reranker_model)

    pipeline = RetrievalPipeline(
        db=db, chroma_store=chroma,
        embedding_generator=embed_gen, reranker=reranker,
    )
    pipeline.build_index()

    llm = make_llm_backend(backend_name, config)
    return RAGEngine(pipeline, llm)


def display_answer(entry: dict, answer: str, sources: list[dict]):
    """Display the generated answer with context for scoring."""
    print(f"\n{'='*60}")
    print(f"  Question [{entry['id']}]: {entry['question']}")
    print(f"  Type: {entry.get('type', '?')}")
    kw = entry.get("expected_keywords", [])
    if kw:
        print(f"  Expected keywords: {', '.join(kw)}")
    print(f"{'─'*60}")
    print(f"  Relevant chunks: {len(entry.get('relevant_chunk_ids', []))}")
    print(f"{'─'*60}")

    print(f"\n  === Generated Answer ===\n")
    print(answer)

    if sources:
        print(f"\n  === Sources ===")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s.get('title', '?')} ({s.get('venue', '?')}, {s.get('year', '?')})")

    print(f"\n{'─'*60}")


def prompt_yn(label: str) -> bool | None:
    """Prompt for y/n, return None on quit."""
    while True:
        val = input(f"  {label} (y/n/q): ").strip().lower()
        if val == "y":
            return True
        if val == "n":
            return False
        if val == "q":
            return None
        print("  Invalid. Use y/n/q.")


def prompt_score(label: str, min_val: int = 1, max_val: int = 5) -> int | None:
    """Prompt for a numeric score, return None on quit."""
    while True:
        val = input(f"  {label} ({min_val}-{max_val}/q): ").strip().lower()
        if val == "q":
            return None
        try:
            num = int(val)
            if min_val <= num <= max_val:
                return num
            print(f"  Must be between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid. Enter a number or 'q'.")


def annotate_entry(entry: dict, engine: RAGEngine) -> bool:
    """Score generation quality for one entry. Returns False if user quit."""
    question = entry["question"]

    print(f"\nGenerating answer for [{entry['id']}]...")
    response = engine.query(question=question, top_k=5)

    display_answer(entry, response.answer, response.sources)

    # Faithfulness
    faithfulness = prompt_yn("Faithfulness — does the answer match the context?")
    if faithfulness is None:
        return False

    # Relevance
    relevance = prompt_score("Relevance — how well does it address the question?")
    if relevance is None:
        return False

    # Citation accuracy
    citation = prompt_yn("Citation accuracy — do [1], [2] markers support the claims?")
    if citation is None:
        return False

    entry["generation_scores"] = {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "citation_accuracy": citation,
        "model": response.model,
        "answer": response.answer,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"  -> Scored: faithful={faithfulness}, relevance={relevance}, citations={citation}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Annotate RAG generation quality"
    )
    parser.add_argument(
        "--backend", choices=["ollama", "groq"], default=None,
        help="LLM backend (default: from LLM_BACKEND env var)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score entries that already have generation_scores",
    )
    args = parser.parse_args()

    config = get_config()
    backend_name = args.backend or config.llm_backend

    eval_set = load_eval_set()
    if not eval_set:
        print(f"No annotations found at {EVAL_SET_PATH}")
        print("Run: python scripts/annotate.py first")
        sys.exit(1)

    # Filter to entries that have retrieval annotations
    annotated = [e for e in eval_set if e.get("relevant_chunk_ids")]
    if not annotated:
        print("No entries with retrieval annotations. Run scripts/annotate.py first.")
        sys.exit(1)

    print(f"\n=== Generation Quality Annotation ===")
    print(f"Entries with retrieval annotations: {len(annotated)}")
    print(f"Backend: {backend_name}\n")

    engine = init_rag_engine(config, backend_name)

    for entry in annotated:
        if entry.get("generation_scores") and not args.force:
            print(f"  [{entry['id']}] already scored — skipping (use --force to redo)")
            continue

        if not annotate_entry(entry, engine):
            save_eval_set(eval_set)
            print("\nAnnotation paused. Progress saved.")
            return

        save_eval_set(eval_set)

    print(f"\nDone. All generation scores saved to {EVAL_SET_PATH}")


if __name__ == "__main__":
    main()
