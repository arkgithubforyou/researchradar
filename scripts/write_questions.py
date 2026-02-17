"""CLI: Author evaluation questions for the annotation pipeline.

Prompts the user to write questions one at a time, asks for type and
expected keywords, and appends to data/questions.json.

Usage:
    python scripts/write_questions.py
    python scripts/write_questions.py --list          # review existing
    python scripts/write_questions.py --edit q001     # re-edit a question
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"
VALID_TYPES = ("factual", "comparison", "analytical", "unanswerable")


def _load_questions() -> list[dict]:
    if QUESTIONS_PATH.exists():
        with open(QUESTIONS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_questions(questions: list[dict]) -> None:
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d questions to %s", len(questions), QUESTIONS_PATH)


def _next_id(questions: list[dict]) -> str:
    existing = [q["id"] for q in questions if q.get("id", "").startswith("q")]
    nums = []
    for qid in existing:
        try:
            nums.append(int(qid[1:]))
        except ValueError:
            continue
    next_num = max(nums, default=0) + 1
    return f"q{next_num:03d}"


def _prompt_question_type() -> str:
    print(f"\nQuestion types: {', '.join(VALID_TYPES)}")
    while True:
        qtype = input("Type: ").strip().lower()
        if qtype in VALID_TYPES:
            return qtype
        print(f"  Invalid. Choose from: {', '.join(VALID_TYPES)}")


def _prompt_keywords() -> list[str]:
    raw = input("Expected keywords (comma-separated, or Enter to skip): ").strip()
    if not raw:
        return []
    return [kw.strip() for kw in raw.split(",") if kw.strip()]


def author_loop(questions: list[dict]) -> None:
    """Interactive loop for writing new questions."""
    print(f"\n=== ResearchRadar Question Authoring ===")
    print(f"Existing questions: {len(questions)}")
    print("Type 'q' to quit.\n")

    while True:
        text = input("Question: ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue

        qtype = _prompt_question_type()
        keywords = _prompt_keywords()

        qid = _next_id(questions)
        entry = {
            "id": qid,
            "question": text,
            "type": qtype,
            "expected_keywords": keywords,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        questions.append(entry)
        _save_questions(questions)
        print(f"  -> Saved as {qid}\n")


def list_questions(questions: list[dict]) -> None:
    """Print all existing questions."""
    if not questions:
        print("No questions yet. Run without --list to create some.")
        return

    print(f"\n=== {len(questions)} Questions ===\n")
    for q in questions:
        kw = ", ".join(q.get("expected_keywords", []))
        kw_str = f"  keywords: {kw}" if kw else ""
        print(f"[{q['id']}] ({q.get('type', '?')}) {q['question']}{kw_str}")
    print()


def edit_question(questions: list[dict], qid: str) -> None:
    """Re-edit an existing question by ID."""
    for q in questions:
        if q["id"] == qid:
            print(f"\nEditing {qid}: {q['question']}")
            text = input(f"New text (Enter to keep): ").strip()
            if text:
                q["question"] = text

            print(f"Current type: {q.get('type', '?')}")
            new_type = input("New type (Enter to keep): ").strip().lower()
            if new_type and new_type in VALID_TYPES:
                q["type"] = new_type

            print(f"Current keywords: {', '.join(q.get('expected_keywords', []))}")
            kw = _prompt_keywords()
            if kw:
                q["expected_keywords"] = kw

            _save_questions(questions)
            print(f"  -> Updated {qid}")
            return

    print(f"Question {qid} not found.")


def main():
    parser = argparse.ArgumentParser(description="Author evaluation questions")
    parser.add_argument("--list", action="store_true", help="List existing questions")
    parser.add_argument("--edit", type=str, metavar="ID", help="Edit a question by ID")
    args = parser.parse_args()

    questions = _load_questions()

    if args.list:
        list_questions(questions)
    elif args.edit:
        edit_question(questions, args.edit)
    else:
        author_loop(questions)


if __name__ == "__main__":
    main()
