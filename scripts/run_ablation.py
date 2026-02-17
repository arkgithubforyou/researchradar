"""Run the full ablation study using annotated ground truth.

Checks that eval_set.json has annotated entries, runs retrieval eval,
and optionally summarizes generation scores.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --k 1 3 5 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROJECT_ROOT

EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval_set.json"


def load_eval_set() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        print(f"No eval set found at {EVAL_SET_PATH}")
        print()
        print("To create ground truth annotations:")
        print("  1. python scripts/write_questions.py     # author questions")
        print("  2. python scripts/annotate.py             # annotate retrieval relevance")
        print("  3. python scripts/annotate_generation.py  # score generation quality")
        sys.exit(1)

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def summarize_generation_scores(entries: list[dict]) -> None:
    """Print aggregate generation quality scores."""
    scored = [e for e in entries if e.get("generation_scores")]
    if not scored:
        print("\nNo generation scores yet. Run: python scripts/annotate_generation.py")
        return

    faithfulness = [e["generation_scores"]["faithfulness"] for e in scored]
    relevance = [e["generation_scores"]["relevance"] for e in scored]
    citations = [e["generation_scores"]["citation_accuracy"] for e in scored]

    n = len(scored)
    print(f"\n=== Generation Quality ({n} scored) ===\n")
    print(f"  Faithfulness:       {sum(faithfulness) / n:.1%} "
          f"({sum(faithfulness)}/{n} faithful)")
    print(f"  Avg relevance:      {sum(relevance) / n:.2f} / 5.0")
    print(f"  Citation accuracy:  {sum(citations) / n:.1%} "
          f"({sum(citations)}/{n} accurate)")

    # Per-entry breakdown
    print(f"\n  {'ID':<10} {'Faith':>6} {'Relev':>6} {'Cite':>6}")
    print(f"  {'-'*30}")
    for e in scored:
        gs = e["generation_scores"]
        f_str = "Y" if gs["faithfulness"] else "N"
        c_str = "Y" if gs["citation_accuracy"] else "N"
        print(f"  {e['id']:<10} {f_str:>6} {gs['relevance']:>6} {c_str:>6}")


def main():
    parser = argparse.ArgumentParser(description="Full ablation study from annotations")
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1, 3, 5, 10],
        help="k values for retrieval metrics",
    )
    args = parser.parse_args()

    eval_set = load_eval_set()
    annotated = [e for e in eval_set if e.get("relevant_chunk_ids")]

    if not annotated:
        print("No retrieval annotations found in eval_set.json")
        print("Run: python scripts/annotate.py")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ResearchRadar Ablation Study")
    print(f"  {len(annotated)} annotated questions")
    print(f"{'='*60}")

    # Import and run retrieval evaluation
    from scripts.retrieval_eval import evaluate_method, print_table

    print(f"\n=== Retrieval Ablation ===\n")
    method_keys = ["bm25_top10", "vector_top10", "hybrid_top10"]
    results = {}
    for mk in method_keys:
        label = mk.replace("_top10", "")
        results[label] = evaluate_method(annotated, mk, args.k)

    print_table(results, args.k)

    # Generation scores
    summarize_generation_scores(eval_set)

    print(f"\n{'='*60}")
    print(f"  Study complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
