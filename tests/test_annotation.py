"""Tests for the annotation infrastructure.

Covers: write_questions helpers, annotate.py pooling/resolution logic,
retrieval_eval metrics computation, and eval_set schema handling.
All tests use mocks and temp files — no model loading required.
"""

import json
from unittest.mock import MagicMock

import pytest


# ── We test the helper functions by importing from scripts via sys.path
# manipulation, but since scripts use sys.path.insert, we import the
# underlying modules they depend on instead and test the logic directly.


# ══════════════════════════════════════════════════════════════════════
# write_questions.py helpers
# ══════════════════════════════════════════════════════════════════════


class TestWriteQuestions:
    def test_load_empty(self, tmp_path):
        """Loading a nonexistent file returns empty list."""
        import scripts.write_questions as wq

        orig = wq.QUESTIONS_PATH
        wq.QUESTIONS_PATH = tmp_path / "questions.json"
        try:
            assert wq._load_questions() == []
        finally:
            wq.QUESTIONS_PATH = orig

    def test_save_and_load_roundtrip(self, tmp_path):
        import scripts.write_questions as wq

        orig = wq.QUESTIONS_PATH
        wq.QUESTIONS_PATH = tmp_path / "questions.json"
        try:
            questions = [
                {"id": "q001", "question": "What is BERT?", "type": "factual",
                 "expected_keywords": ["BERT", "bidirectional"]},
            ]
            wq._save_questions(questions)
            loaded = wq._load_questions()
            assert len(loaded) == 1
            assert loaded[0]["id"] == "q001"
            assert loaded[0]["expected_keywords"] == ["BERT", "bidirectional"]
        finally:
            wq.QUESTIONS_PATH = orig

    def test_next_id_empty(self):
        import scripts.write_questions as wq

        assert wq._next_id([]) == "q001"

    def test_next_id_increments(self):
        import scripts.write_questions as wq

        questions = [{"id": "q001"}, {"id": "q002"}, {"id": "q005"}]
        assert wq._next_id(questions) == "q006"

    def test_next_id_handles_non_numeric(self):
        import scripts.write_questions as wq

        questions = [{"id": "q001"}, {"id": "adhoc_20260101"}]
        assert wq._next_id(questions) == "q002"


# ══════════════════════════════════════════════════════════════════════
# annotate.py pooling and resolution logic
# ══════════════════════════════════════════════════════════════════════


class TestAnnotatePooling:
    def test_run_pooled_retrieval_deduplicates(self):
        """Pooled results should deduplicate across methods."""
        import scripts.annotate as ann

        bm25 = MagicMock()
        vector = MagicMock()
        hybrid = MagicMock()

        bm25.search.return_value = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 3}]
        vector.search.return_value = [{"chunk_id": 2}, {"chunk_id": 4}, {"chunk_id": 5}]
        hybrid.search.return_value = [{"chunk_id": 1}, {"chunk_id": 4}, {"chunk_id": 6}]

        pooled_ids, method_pools = ann.run_pooled_retrieval(
            "test query", bm25, vector, hybrid, top_k=3,
        )

        # Should have 6 unique IDs
        assert len(pooled_ids) == 6
        assert set(pooled_ids) == {1, 2, 3, 4, 5, 6}

        # Method pools should reflect each method's results
        assert method_pools["bm25_top10"] == [1, 2, 3]
        assert method_pools["vector_top10"] == [2, 4, 5]
        assert method_pools["hybrid_top10"] == [1, 4, 6]

    def test_run_pooled_preserves_order(self):
        """First-seen order should be preserved (BM25 first, then vector, then hybrid)."""
        import scripts.annotate as ann

        bm25 = MagicMock()
        vector = MagicMock()
        hybrid = MagicMock()

        bm25.search.return_value = [{"chunk_id": 10}, {"chunk_id": 20}]
        vector.search.return_value = [{"chunk_id": 30}, {"chunk_id": 10}]
        hybrid.search.return_value = [{"chunk_id": 40}]

        pooled_ids, _ = ann.run_pooled_retrieval("q", bm25, vector, hybrid, top_k=2)
        assert pooled_ids == [10, 20, 30, 40]

    def test_chunk_methods(self):
        """chunk_methods should return which methods found a given chunk."""
        import scripts.annotate as ann

        pools = {
            "bm25_top10": [1, 2, 3],
            "vector_top10": [2, 4, 5],
            "hybrid_top10": [1, 4, 6],
        }

        assert set(ann.chunk_methods(1, pools)) == {"bm25", "hybrid"}
        assert ann.chunk_methods(2, pools) == ["bm25", "vector"]
        assert ann.chunk_methods(5, pools) == ["vector"]
        assert ann.chunk_methods(99, pools) == []


class TestAnnotateState:
    def test_load_empty_eval_set(self, tmp_path):
        import scripts.annotate as ann

        orig = ann.EVAL_SET_PATH
        ann.EVAL_SET_PATH = tmp_path / "eval_set.json"
        try:
            assert ann.load_eval_set() == []
        finally:
            ann.EVAL_SET_PATH = orig

    def test_save_and_load_eval_set(self, tmp_path):
        import scripts.annotate as ann

        orig = ann.EVAL_SET_PATH
        ann.EVAL_SET_PATH = tmp_path / "eval_set.json"
        try:
            data = [{"id": "q001", "question": "test?", "relevant_chunk_ids": ["1"]}]
            ann.save_eval_set(data)
            loaded = ann.load_eval_set()
            assert len(loaded) == 1
            assert loaded[0]["id"] == "q001"
        finally:
            ann.EVAL_SET_PATH = orig

    def test_find_entry(self):
        import scripts.annotate as ann

        eval_set = [
            {"id": "q001", "question": "A?"},
            {"id": "q002", "question": "B?"},
        ]
        assert ann.find_entry(eval_set, "q001")["question"] == "A?"
        assert ann.find_entry(eval_set, "q999") is None

    def test_truncate(self):
        import scripts.annotate as ann

        short = "Hello world"
        assert ann.truncate(short, 500) == short

        long = "x" * 600
        truncated = ann.truncate(long, 500)
        assert len(truncated) == 503  # 500 + "..."
        assert truncated.endswith("...")


# ══════════════════════════════════════════════════════════════════════
# retrieval_eval.py metric computation
# ══════════════════════════════════════════════════════════════════════


class TestRetrievalEval:
    def test_evaluate_method_perfect(self):
        """All relevant chunks in the pool should yield perfect scores."""
        from scripts.retrieval_eval import evaluate_method

        entries = [{
            "pooled_from": {"bm25_top10": ["1", "2", "3"]},
            "relevant_chunk_ids": ["1", "2"],
        }]
        result = evaluate_method(entries, "bm25_top10", k_values=[3])
        assert result["mrr"] == 1.0
        assert result["hit_rate@3"] == 1.0
        assert result["recall@3"] == 1.0

    def test_evaluate_method_miss(self):
        """No relevant chunks in pool yields zero scores."""
        from scripts.retrieval_eval import evaluate_method

        entries = [{
            "pooled_from": {"bm25_top10": ["1", "2"]},
            "relevant_chunk_ids": ["99"],
        }]
        result = evaluate_method(entries, "bm25_top10", k_values=[2])
        assert result["mrr"] == 0.0
        assert result["hit_rate@2"] == 0.0

    def test_evaluate_method_partial(self):
        """One relevant at position 2 out of 3."""
        from scripts.retrieval_eval import evaluate_method

        entries = [{
            "pooled_from": {"vector_top10": ["10", "20", "30"]},
            "relevant_chunk_ids": ["20"],
        }]
        result = evaluate_method(entries, "vector_top10", k_values=[3])
        assert result["mrr"] == 0.5
        assert result["hit_rate@3"] == 1.0
        assert result["precision@3"] == pytest.approx(1 / 3)

    def test_evaluate_method_multiple_entries(self):
        """Average across multiple questions."""
        from scripts.retrieval_eval import evaluate_method

        entries = [
            {
                "pooled_from": {"bm25_top10": ["1"]},
                "relevant_chunk_ids": ["1"],
            },
            {
                "pooled_from": {"bm25_top10": ["2"]},
                "relevant_chunk_ids": ["99"],  # miss
            },
        ]
        result = evaluate_method(entries, "bm25_top10", k_values=[1])
        assert result["mrr"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert result["hit_rate@1"] == pytest.approx(0.5)

    def test_evaluate_method_missing_pool(self):
        """Entry without the method in pooled_from should still work."""
        from scripts.retrieval_eval import evaluate_method

        entries = [{
            "pooled_from": {"bm25_top10": ["1"]},
            "relevant_chunk_ids": ["1"],
        }]
        result = evaluate_method(entries, "vector_top10", k_values=[1])
        assert result["mrr"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# eval_set.json schema tests
# ══════════════════════════════════════════════════════════════════════


class TestEvalSetSchema:
    def test_full_schema_roundtrip(self, tmp_path):
        """Verify the full annotated schema can be saved and loaded."""
        import scripts.annotate as ann

        orig = ann.EVAL_SET_PATH
        ann.EVAL_SET_PATH = tmp_path / "eval_set.json"
        try:
            entry = {
                "id": "q001",
                "question": "What methods are used for fine-tuning?",
                "type": "factual",
                "expected_keywords": ["LoRA", "adapter"],
                "relevant_chunk_ids": ["123", "456"],
                "irrelevant_chunk_ids": ["012"],
                "skipped_chunk_ids": [],
                "relevant_paper_ids": ["acl::2022.acl-long.220"],
                "pooled_from": {
                    "bm25_top10": ["123", "012", "789"],
                    "vector_top10": ["456", "345"],
                    "hybrid_top10": ["123", "456"],
                },
                "annotated_at": "2026-02-15T14:30:00Z",
            }
            ann.save_eval_set([entry])
            loaded = ann.load_eval_set()

            assert len(loaded) == 1
            e = loaded[0]
            assert e["id"] == "q001"
            assert e["relevant_chunk_ids"] == ["123", "456"]
            assert e["relevant_paper_ids"] == ["acl::2022.acl-long.220"]
            assert "bm25_top10" in e["pooled_from"]
        finally:
            ann.EVAL_SET_PATH = orig

    def test_generation_scores_schema(self, tmp_path):
        """Verify generation_scores field can be added and persisted."""
        path = tmp_path / "eval_set.json"
        entry = {
            "id": "q001",
            "question": "test?",
            "relevant_chunk_ids": ["1"],
            "generation_scores": {
                "faithfulness": True,
                "relevance": 4,
                "citation_accuracy": True,
                "model": "test-model",
                "answer": "The answer is...",
                "scored_at": "2026-02-15T15:00:00Z",
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump([entry], f)

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)

        gs = loaded[0]["generation_scores"]
        assert gs["faithfulness"] is True
        assert gs["relevance"] == 4
        assert gs["citation_accuracy"] is True
        assert gs["model"] == "test-model"
