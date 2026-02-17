"""Tests for the Phase 5 evaluation framework.

Covers metrics (retrieval + generation), dataset loading, eval runner,
and ablation study. All tests use mocks — no model loading required.
"""

import json
import math
from unittest.mock import MagicMock

import pytest

from src.evaluation.ablation import (
    AblationConfig,
    AblationReport,
    AblationRunner,
    default_ablation_configs,
)
from src.evaluation.dataset import (
    EvalDataset,
    EvalExample,
    create_sample_dataset,
    load_eval_dataset,
)
from src.evaluation.metrics import (
    answer_coverage,
    evaluate_generation,
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_l_f1,
)
from src.evaluation.runner import EvalReport, EvalRunner, ExampleResult
from src.generation.rag_engine import RAGEngine, RAGResponse
from src.retrieval.pipeline import RetrievalConfig, RetrievalPipeline, RetrievalResult


# ══════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════════════


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y", "z"], {"a", "b"}, 3) == 0.0

    def test_partial_precision(self):
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)

    def test_k_larger_than_results(self):
        assert precision_at_k(["a"], {"a", "b"}, 5) == pytest.approx(1 / 5)

    def test_k_zero(self):
        assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, 3) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, 2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_recall_at_1(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 1) == pytest.approx(1 / 3)


class TestMRR:
    def test_first_relevant(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_relevant(self):
        assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_third_relevant(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], {"a"}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_zero_ndcg(self):
        assert ndcg_at_k(["x", "y"], {"a", "b"}, 2) == 0.0

    def test_imperfect_ranking(self):
        # relevant = {a, b}, retrieved = [x, a, b] at k=3
        # DCG = 0 + 1/log2(3) + 1/log2(4)
        # IDCG = 1/log2(2) + 1/log2(3)
        score = ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3)
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert score == pytest.approx(dcg / idcg)

    def test_k_zero(self):
        assert ndcg_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), 2) == 0.0


class TestEvaluateRetrieval:
    def test_returns_all_metrics(self):
        results = evaluate_retrieval(["a", "b", "c"], {"a", "c"})
        assert "mrr" in results
        assert "precision@1" in results
        assert "recall@5" in results
        assert "ndcg@10" in results

    def test_custom_k_values(self):
        results = evaluate_retrieval(["a"], {"a"}, k_values=[1, 2])
        assert "precision@1" in results
        assert "precision@2" in results
        assert "precision@5" not in results


# ══════════════════════════════════════════════════════════════════════
# GENERATION METRICS
# ══════════════════════════════════════════════════════════════════════


class TestRougeL:
    def test_identical(self):
        assert rouge_l_f1("the cat sat on the mat", "the cat sat on the mat") == 1.0

    def test_no_overlap(self):
        assert rouge_l_f1("hello world", "foo bar baz") == 0.0

    def test_partial_overlap(self):
        score = rouge_l_f1("the cat sat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_empty_prediction(self):
        assert rouge_l_f1("", "some reference") == 0.0

    def test_empty_reference(self):
        assert rouge_l_f1("some prediction", "") == 0.0

    def test_symmetry_approximately(self):
        # ROUGE-L F1 is symmetric for same-length inputs
        s1 = rouge_l_f1("a b c d", "a b x d")
        s2 = rouge_l_f1("a b x d", "a b c d")
        assert s1 == pytest.approx(s2)


class TestAnswerCoverage:
    def test_full_coverage(self):
        assert answer_coverage("BERT uses attention", ["BERT", "attention"]) == 1.0

    def test_zero_coverage(self):
        assert answer_coverage("hello world", ["BERT", "attention"]) == 0.0

    def test_partial_coverage(self):
        assert answer_coverage("BERT is great", ["BERT", "attention"]) == 0.5

    def test_case_insensitive(self):
        assert answer_coverage("bert uses ATTENTION", ["BERT", "attention"]) == 1.0

    def test_empty_keywords(self):
        assert answer_coverage("some text", []) == 0.0


class TestEvaluateGeneration:
    def test_returns_rouge(self):
        results = evaluate_generation("pred", "ref")
        assert "rouge_l_f1" in results
        assert "answer_coverage" not in results

    def test_with_keywords(self):
        results = evaluate_generation("pred has word", "ref", keywords=["word"])
        assert "answer_coverage" in results
        assert results["answer_coverage"] == 1.0


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════


class TestEvalExample:
    def test_basic(self):
        ex = EvalExample(
            query="What is BERT?",
            relevant_paper_ids=["test::1"],
            reference_answer="BERT is...",
        )
        assert ex.query == "What is BERT?"
        assert ex.keywords == []
        assert ex.metadata == {}


class TestEvalDataset:
    def test_len(self):
        ds = EvalDataset(name="test", examples=[
            EvalExample(query="q1", relevant_paper_ids=[], reference_answer="a1"),
            EvalExample(query="q2", relevant_paper_ids=[], reference_answer="a2"),
        ])
        assert len(ds) == 2

    def test_iter(self):
        examples = [
            EvalExample(query="q1", relevant_paper_ids=[], reference_answer="a1"),
        ]
        ds = EvalDataset(name="test", examples=examples)
        assert list(ds) == examples


class TestLoadEvalDataset:
    def test_load_from_file(self, tmp_path):
        data = {
            "name": "test-ds",
            "description": "A test dataset",
            "examples": [
                {
                    "query": "What is attention?",
                    "relevant_paper_ids": ["test::1"],
                    "reference_answer": "Attention is...",
                    "keywords": ["attention", "mechanism"],
                    "metadata": {"category": "methods"},
                },
                {
                    "query": "What is LoRA?",
                    "relevant_paper_ids": ["test::2"],
                    "reference_answer": "LoRA is...",
                },
            ],
        }
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        ds = load_eval_dataset(path)
        assert ds.name == "test-ds"
        assert len(ds) == 2
        assert ds.examples[0].keywords == ["attention", "mechanism"]
        assert ds.examples[1].keywords == []

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_eval_dataset(tmp_path / "nonexistent.json")


class TestCreateSampleDataset:
    def test_returns_dataset(self):
        ds = create_sample_dataset()
        assert isinstance(ds, EvalDataset)
        assert len(ds) == 3
        assert ds.name == "smoke-test"

    def test_examples_have_required_fields(self):
        ds = create_sample_dataset()
        for ex in ds:
            assert ex.query
            assert ex.relevant_paper_ids
            assert ex.reference_answer


# ══════════════════════════════════════════════════════════════════════
# EVAL RUNNER
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_rag_engine_for_eval():
    """RAG engine mock for evaluation tests."""
    engine = MagicMock(spec=RAGEngine)
    engine.query.return_value = RAGResponse(
        answer="Transformers use self-attention mechanisms for machine translation.",
        sources=[
            {
                "paper_id": "hf_acl_ocl::P18-1001",
                "title": "Attention Is All You Need",
                "year": 2018,
                "venue": "acl",
                "chunk_type": "abstract",
            }
        ],
        model="mock",
        usage={},
    )
    return engine


class TestEvalRunner:
    def test_evaluate_example(self, mock_rag_engine_for_eval):
        runner = EvalRunner(mock_rag_engine_for_eval)
        example = EvalExample(
            query="What is the transformer?",
            relevant_paper_ids=["hf_acl_ocl::P18-1001"],
            reference_answer="Transformers use attention for translation.",
            keywords=["transformer", "attention"],
        )
        result = runner.evaluate_example(example)

        assert isinstance(result, ExampleResult)
        assert result.query == "What is the transformer?"
        assert "mrr" in result.retrieval_metrics
        assert result.retrieval_metrics["mrr"] == 1.0  # first result is relevant
        assert "rouge_l_f1" in result.generation_metrics
        assert "answer_coverage" in result.generation_metrics

    def test_run_full_dataset(self, mock_rag_engine_for_eval):
        runner = EvalRunner(mock_rag_engine_for_eval, top_k=3)
        dataset = EvalDataset(
            name="test",
            examples=[
                EvalExample(
                    query="q1",
                    relevant_paper_ids=["hf_acl_ocl::P18-1001"],
                    reference_answer="ref1",
                    keywords=["attention"],
                ),
                EvalExample(
                    query="q2",
                    relevant_paper_ids=["nonexistent::999"],
                    reference_answer="ref2",
                ),
            ],
        )
        report = runner.run(dataset)

        assert isinstance(report, EvalReport)
        assert report.num_examples == 2
        assert len(report.example_results) == 2
        assert "mrr" in report.aggregate_retrieval
        assert "rouge_l_f1" in report.aggregate_generation

    def test_aggregate_metrics_are_averages(self, mock_rag_engine_for_eval):
        runner = EvalRunner(mock_rag_engine_for_eval)
        dataset = EvalDataset(
            name="test",
            examples=[
                EvalExample(
                    query="q1",
                    relevant_paper_ids=["hf_acl_ocl::P18-1001"],
                    reference_answer="ref",
                ),
                EvalExample(
                    query="q2",
                    relevant_paper_ids=["hf_acl_ocl::P18-1001"],
                    reference_answer="ref",
                ),
            ],
        )
        report = runner.run(dataset)
        # Since both examples have same setup, aggregates should equal individual
        ind = report.example_results[0].retrieval_metrics
        agg = report.aggregate_retrieval
        for key in ind:
            assert agg[key] == pytest.approx(ind[key])


# ══════════════════════════════════════════════════════════════════════
# ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_pipeline_for_ablation():
    """A mock retrieval pipeline for ablation tests."""
    pipeline = MagicMock(spec=RetrievalPipeline)
    pipeline.config = RetrievalConfig()
    pipeline.reranker = MagicMock()
    pipeline.search.return_value = [
        RetrievalResult(
            chunk_id=1,
            paper_id="hf_acl_ocl::P18-1001",
            chunk_text="Transformer architecture for machine translation.",
            chunk_type="abstract",
            title="Attention Is All You Need",
            year=2018,
            venue="acl",
            rrf_score=0.05,
            ce_score=0.9,
        ),
    ]
    return pipeline


class TestDefaultAblationConfigs:
    def test_returns_list(self):
        configs = default_ablation_configs()
        assert isinstance(configs, list)
        assert len(configs) >= 4

    def test_configs_have_names(self):
        for cfg in default_ablation_configs():
            assert cfg.name
            assert isinstance(cfg.retrieval_config, RetrievalConfig)

    def test_includes_full_pipeline(self):
        names = {c.name for c in default_ablation_configs()}
        assert "full_pipeline" in names

    def test_includes_no_reranker(self):
        configs = {c.name: c for c in default_ablation_configs()}
        assert not configs["no_reranker"].retrieval_config.use_reranker


class TestAblationRunner:
    def test_run_single_config(self, mock_pipeline_for_ablation):
        runner = AblationRunner(mock_pipeline_for_ablation)
        dataset = EvalDataset(
            name="test",
            examples=[
                EvalExample(
                    query="What is attention?",
                    relevant_paper_ids=["hf_acl_ocl::P18-1001"],
                    reference_answer="Attention is...",
                ),
            ],
        )
        config = AblationConfig(
            name="test_config",
            retrieval_config=RetrievalConfig(use_reranker=False),
        )
        result = runner.run_config(config, dataset)

        assert result.config_name == "test_config"
        assert result.num_examples == 1
        assert "mrr" in result.aggregate_metrics

    def test_run_full_study(self, mock_pipeline_for_ablation):
        runner = AblationRunner(mock_pipeline_for_ablation)
        dataset = EvalDataset(
            name="test",
            examples=[
                EvalExample(
                    query="q1",
                    relevant_paper_ids=["hf_acl_ocl::P18-1001"],
                    reference_answer="ref",
                ),
            ],
        )
        report = runner.run_study(dataset)

        assert isinstance(report, AblationReport)
        assert len(report.results) >= 4  # default configs
        assert report.dataset_name == "test"

    def test_summary_table(self, mock_pipeline_for_ablation):
        runner = AblationRunner(mock_pipeline_for_ablation)
        dataset = EvalDataset(
            name="test",
            examples=[
                EvalExample(
                    query="q1",
                    relevant_paper_ids=["hf_acl_ocl::P18-1001"],
                    reference_answer="ref",
                ),
            ],
        )
        report = runner.run_study(dataset)
        table = report.summary_table()

        assert isinstance(table, list)
        assert len(table) >= 4
        for row in table:
            assert "config" in row
            assert "mrr" in row

    def test_restores_pipeline_config(self, mock_pipeline_for_ablation):
        """Ablation runner should restore original config after each experiment."""
        original_config = mock_pipeline_for_ablation.config
        original_reranker = mock_pipeline_for_ablation.reranker

        runner = AblationRunner(mock_pipeline_for_ablation)
        dataset = EvalDataset(
            name="test",
            examples=[
                EvalExample(
                    query="q", relevant_paper_ids=[], reference_answer="r",
                ),
            ],
        )
        config = AblationConfig(
            name="no_reranker",
            retrieval_config=RetrievalConfig(use_reranker=False),
        )
        runner.run_config(config, dataset)

        assert mock_pipeline_for_ablation.config is original_config
        assert mock_pipeline_for_ablation.reranker is original_reranker
