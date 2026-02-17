"""Retrieval evaluation using annotated ground truth from eval_set.json.

Loads the annotation file, computes Hit Rate@k and MRR for each retrieval
method separately using the pooled_from data and relevant_chunk_ids, and
produces a comparison table.

Usage:
    python scripts/retrieval_eval.py
    python scripts/retrieval_eval.py --k 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROJECT_ROOT
from src.evaluation.metrics import mrr, precision_at_k, recall_at_k, ndcg_at_k

EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval_set.json"


def load_eval_set() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        print(f"Eval set not found: {EVAL_SET_PATH}")
        print("Run: python scripts/write_questions.py && python scripts/annotate.py")
        sys.exit(1)

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    annotated = [e for e in data if e.get("relevant_chunk_ids")]
    if not annotated:
        print("No annotated entries found. Run scripts/annotate.py first.")
        sys.exit(1)

    return annotated


def evaluate_method(
    entries: list[dict],
    method_key: str,
    k_values: list[int],
) -> dict:
    """Compute retrieval metrics for one method across all entries."""
    all_metrics: dict[str, list[float]] = {}

    for entry in entries:
        pooled = entry.get("pooled_from", {})
        retrieved_ids = [str(x) for x in pooled.get(method_key, [])]
        relevant_ids = set(str(x) for x in entry.get("relevant_chunk_ids", []))

        if not relevant_ids:
            continue

        entry_mrr = mrr(retrieved_ids, relevant_ids)
        all_metrics.setdefault("mrr", []).append(entry_mrr)

        for k in k_values:
            hit = 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0
            all_metrics.setdefault(f"hit_rate@{k}", []).append(hit)
            all_metrics.setdefault(f"precision@{k}", []).append(
                precision_at_k(retrieved_ids, relevant_ids, k)
            )
            all_metrics.setdefault(f"recall@{k}", []).append(
                recall_at_k(retrieved_ids, relevant_ids, k)
            )
            all_metrics.setdefault(f"ndcg@{k}", []).append(
                ndcg_at_k(retrieved_ids, relevant_ids, k)
            )

    # Average
    return {
        key: sum(vals) / len(vals) if vals else 0.0
        for key, vals in all_metrics.items()
    }


def print_table(results: dict[str, dict], k_values: list[int]) -> None:
    """Print a comparison table."""
    methods = list(results.keys())

    # Collect metric columns in order
    columns = ["mrr"]
    for k in k_values:
        columns.extend([f"hit_rate@{k}", f"precision@{k}", f"recall@{k}", f"ndcg@{k}"])

    # Header
    header = f"{'Method':<15}"
    for col in columns:
        header += f"  {col:>12}"
    print(header)
    print("-" * len(header))

    # Rows
    for method in methods:
        row = f"{method:<15}"
        for col in columns:
            val = results[method].get(col, 0.0)
            row += f"  {val:>12.3f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation from annotations")
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1, 3, 5, 10],
        help="k values for metrics (default: 1 3 5 10)",
    )
    args = parser.parse_args()

    entries = load_eval_set()
    print(f"\n=== Retrieval Evaluation ({len(entries)} annotated questions) ===\n")

    method_keys = ["bm25_top10", "vector_top10", "hybrid_top10"]
    results = {}
    for mk in method_keys:
        label = mk.replace("_top10", "")
        results[label] = evaluate_method(entries, mk, args.k)

    print_table(results, args.k)
    print()


if __name__ == "__main__":
    main()
