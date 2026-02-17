"""CLI: Pooled retrieval annotation tool (TREC-style).

Runs BM25, vector, and hybrid retrieval on each question, pools unique
chunks, and presents them for human relevance judgment.

Usage:
    python scripts/annotate.py
    python scripts/annotate.py --question "What is LoRA?"
    python scripts/annotate.py --top-k 10
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROJECT_ROOT, get_config
from src.ingestion.embeddings import EmbeddingGenerator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval_set.json"


# ── Retrieval helpers ────────────────────────────────────────────────


def init_retrievers(config):
    """Initialize DB, BM25, vector, and hybrid retrievers."""
    db = SQLiteDB(config.sqlite_db_path)
    chroma = ChromaStore(config.chroma_db_path)
    embed_gen = EmbeddingGenerator(config.embedding_model)

    bm25 = BM25Retriever(db)
    bm25.build_index()

    vector = VectorRetriever(chroma, embed_gen)
    hybrid = HybridRetriever(bm25, vector)

    return db, bm25, vector, hybrid


def run_pooled_retrieval(
    question: str,
    bm25: BM25Retriever,
    vector: VectorRetriever,
    hybrid: HybridRetriever,
    top_k: int = 10,
) -> tuple[list[dict], dict[str, list]]:
    """Run all three methods and pool unique chunks.

    Returns:
        (pooled_chunks, method_pools) where pooled_chunks has unique chunk_ids
        and method_pools maps method name → list of chunk_ids retrieved.
    """
    bm25_results = bm25.search(question, top_k=top_k)
    vector_results = vector.search(question, top_k=top_k)
    hybrid_results = hybrid.search(question, top_k=top_k)

    bm25_ids = [r["chunk_id"] for r in bm25_results]
    vector_ids = [r["chunk_id"] for r in vector_results]
    hybrid_ids = [r["chunk_id"] for r in hybrid_results]

    method_pools = {
        "bm25_top10": bm25_ids,
        "vector_top10": vector_ids,
        "hybrid_top10": hybrid_ids,
    }

    # Deduplicate, preserving first-seen order
    seen = set()
    pooled_ids = []
    for cid in bm25_ids + vector_ids + hybrid_ids:
        if cid not in seen:
            seen.add(cid)
            pooled_ids.append(cid)

    return pooled_ids, method_pools


def resolve_chunks(db: SQLiteDB, chunk_ids: list) -> dict:
    """Look up chunk + paper metadata for each chunk ID."""
    all_chunks = db.get_all_chunks()
    chunk_map = {c["id"]: c for c in all_chunks}

    resolved = {}
    for cid in chunk_ids:
        chunk = chunk_map.get(cid)
        if chunk is not None:
            resolved[cid] = chunk
    return resolved


def chunk_methods(chunk_id, method_pools: dict) -> list[str]:
    """Which methods retrieved this chunk."""
    methods = []
    for method, ids in method_pools.items():
        if chunk_id in ids:
            methods.append(method.split("_")[0])  # "bm25", "vector", "hybrid"
    return methods


# ── Annotation state ─────────────────────────────────────────────────


def load_eval_set() -> list[dict]:
    if EVAL_SET_PATH.exists():
        with open(EVAL_SET_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_eval_set(eval_set: list[dict]) -> None:
    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)


def find_entry(eval_set: list[dict], question_id: str) -> dict | None:
    for entry in eval_set:
        if entry.get("id") == question_id:
            return entry
    return None


# ── Display helpers ──────────────────────────────────────────────────


def truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def display_chunk(
    idx: int,
    total: int,
    chunk: dict,
    methods: list[str],
    show_full: bool = False,
):
    """Print a single chunk for annotation."""
    print(f"\n{'='*60}")
    print(f"  [{idx}/{total}]  Methods: {', '.join(methods)}")
    print(f"  Title: {chunk.get('title', '?')}")
    print(f"  Venue: {chunk.get('venue', '?')} | Year: {chunk.get('year', '?')}")
    print(f"  Paper ID: {chunk.get('paper_id', '?')}")
    print(f"  Chunk ID: {chunk.get('id', '?')} | Type: {chunk.get('chunk_type', '?')}")
    print(f"{'─'*60}")

    text = chunk.get("chunk_text", "")
    if show_full or len(text) <= 500:
        print(text)
    else:
        print(truncate(text, 500))
        print(f"  [{len(text)} chars total — press 'f' to show full]")

    print(f"{'─'*60}")
    print("  (y) relevant  (n) not relevant  (s) skip  (f) full text  (q) quit")


# ── Annotation loop ──────────────────────────────────────────────────


def annotate_question(
    question: dict,
    db: SQLiteDB,
    bm25: BM25Retriever,
    vector: VectorRetriever,
    hybrid: HybridRetriever,
    eval_set: list[dict],
    top_k: int = 10,
) -> bool:
    """Annotate one question. Returns False if user quit."""
    qid = question["id"]
    qtext = question["question"]

    print(f"\n{'#'*60}")
    print(f"  Question [{qid}]: {qtext}")
    print(f"  Type: {question.get('type', '?')}")
    kw = question.get("expected_keywords", [])
    if kw:
        print(f"  Expected keywords: {', '.join(kw)}")
    print(f"{'#'*60}")

    print("\nRunning retrieval (BM25 + vector + hybrid)...")
    pooled_ids, method_pools = run_pooled_retrieval(
        qtext, bm25, vector, hybrid, top_k=top_k,
    )

    resolved = resolve_chunks(db, pooled_ids)
    ordered_ids = [cid for cid in pooled_ids if cid in resolved]
    total = len(ordered_ids)

    if total == 0:
        print("  No chunks retrieved. Skipping.")
        return True

    print(f"\nPooled {total} unique chunks from {top_k * 3} candidates.\n")

    relevant_chunk_ids = []
    irrelevant_chunk_ids = []
    skipped_chunk_ids = []

    i = 0
    while i < total:
        cid = ordered_ids[i]
        chunk = resolved[cid]
        methods = chunk_methods(cid, method_pools)
        display_chunk(i + 1, total, chunk, methods, show_full=False)

        action = input("  > ").strip().lower()
        if action == "y":
            relevant_chunk_ids.append(cid)
            i += 1
        elif action == "n":
            irrelevant_chunk_ids.append(cid)
            i += 1
        elif action == "s":
            skipped_chunk_ids.append(cid)
            i += 1
        elif action == "f":
            display_chunk(i + 1, total, chunk, methods, show_full=True)
            # Don't advance — let user judge after seeing full text
        elif action == "q":
            # Save partial progress before quitting
            _save_annotation(
                eval_set, question, method_pools,
                relevant_chunk_ids, irrelevant_chunk_ids, skipped_chunk_ids,
                resolved,
            )
            return False
        else:
            print("  Invalid. Use y/n/s/f/q.")

    # Save completed annotation
    _save_annotation(
        eval_set, question, method_pools,
        relevant_chunk_ids, irrelevant_chunk_ids, skipped_chunk_ids,
        resolved,
    )
    return True


def _save_annotation(
    eval_set: list[dict],
    question: dict,
    method_pools: dict,
    relevant_chunk_ids: list,
    irrelevant_chunk_ids: list,
    skipped_chunk_ids: list,
    resolved: dict,
):
    """Build and save the annotation entry."""
    # Derive relevant paper IDs from relevant chunks
    relevant_paper_ids = list(dict.fromkeys(
        resolved[cid]["paper_id"]
        for cid in relevant_chunk_ids
        if cid in resolved
    ))

    # Stringify chunk IDs for JSON
    def to_str_ids(ids):
        return [str(x) for x in ids]

    entry = {
        "id": question["id"],
        "question": question["question"],
        "type": question.get("type", ""),
        "expected_keywords": question.get("expected_keywords", []),
        "relevant_chunk_ids": to_str_ids(relevant_chunk_ids),
        "irrelevant_chunk_ids": to_str_ids(irrelevant_chunk_ids),
        "skipped_chunk_ids": to_str_ids(skipped_chunk_ids),
        "relevant_paper_ids": relevant_paper_ids,
        "pooled_from": {
            k: to_str_ids(v) for k, v in method_pools.items()
        },
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Replace or append
    existing = find_entry(eval_set, question["id"])
    if existing is not None:
        idx = eval_set.index(existing)
        eval_set[idx] = entry
    else:
        eval_set.append(entry)

    save_eval_set(eval_set)
    n_rel = len(relevant_chunk_ids)
    n_irr = len(irrelevant_chunk_ids)
    n_skip = len(skipped_chunk_ids)
    print(f"\n  Saved: {n_rel} relevant, {n_irr} irrelevant, {n_skip} skipped")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Pooled retrieval annotation (TREC-style)"
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Annotate a single ad-hoc question (bypasses questions.json)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Results per method")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-annotate questions that already have judgments",
    )
    args = parser.parse_args()

    config = get_config()
    db, bm25, vector, hybrid = init_retrievers(config)
    eval_set = load_eval_set()

    if args.question:
        # Ad-hoc single question
        q = {
            "id": f"adhoc_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "question": args.question,
            "type": "factual",
            "expected_keywords": [],
        }
        annotate_question(q, db, bm25, vector, hybrid, eval_set, top_k=args.top_k)
        return

    # Load questions from file
    if not QUESTIONS_PATH.exists():
        print(f"No questions file found at {QUESTIONS_PATH}")
        print("Run: python scripts/write_questions.py")
        sys.exit(1)

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    if not questions:
        print("Questions file is empty. Add questions first.")
        sys.exit(1)

    annotated_ids = {e["id"] for e in eval_set}

    for q in questions:
        qid = q["id"]
        if qid in annotated_ids and not args.force:
            print(f"\n  [{qid}] already annotated — skipping (use --force to redo)")
            continue

        if not annotate_question(q, db, bm25, vector, hybrid, eval_set, top_k=args.top_k):
            print("\nAnnotation paused. Progress saved.")
            break

    print(f"\nDone. {len(eval_set)} entries in {EVAL_SET_PATH}")


if __name__ == "__main__":
    main()
