"""Evaluation runner — runs the RAG pipeline on an eval dataset and scores it.

Produces per-example and aggregate metrics for both retrieval and generation.
"""

import logging
from dataclasses import dataclass, field

from src.evaluation.dataset import EvalDataset, EvalExample
from src.evaluation.metrics import evaluate_generation, evaluate_retrieval
from src.generation.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


@dataclass
class ExampleResult:
    """Evaluation results for a single example."""

    query: str
    retrieval_metrics: dict
    generation_metrics: dict
    retrieved_paper_ids: list[str]
    answer: str


@dataclass
class EvalReport:
    """Aggregate evaluation report."""

    dataset_name: str
    num_examples: int
    example_results: list[ExampleResult]
    aggregate_retrieval: dict = field(default_factory=dict)
    aggregate_generation: dict = field(default_factory=dict)


class EvalRunner:
    """Runs evaluation on a dataset using the RAG engine."""

    def __init__(self, rag_engine: RAGEngine, top_k: int = 5):
        self.engine = rag_engine
        self.top_k = top_k

    def evaluate_example(self, example: EvalExample) -> ExampleResult:
        """Evaluate a single example."""
        response = self.engine.query(question=example.query, top_k=self.top_k)

        retrieved_paper_ids = [s["paper_id"] for s in response.sources]
        relevant_ids = set(example.relevant_paper_ids)

        retrieval_metrics = evaluate_retrieval(
            retrieved_ids=retrieved_paper_ids,
            relevant_ids=relevant_ids,
        )

        generation_metrics = evaluate_generation(
            prediction=response.answer,
            reference=example.reference_answer,
            keywords=example.keywords if example.keywords else None,
        )

        return ExampleResult(
            query=example.query,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            retrieved_paper_ids=retrieved_paper_ids,
            answer=response.answer,
        )

    def run(self, dataset: EvalDataset) -> EvalReport:
        """Run evaluation on the full dataset."""
        logger.info("Running evaluation on %r (%d examples)", dataset.name, len(dataset))

        results = []
        for i, example in enumerate(dataset):
            logger.info("Evaluating %d/%d: %r", i + 1, len(dataset), example.query)
            result = self.evaluate_example(example)
            results.append(result)

        report = EvalReport(
            dataset_name=dataset.name,
            num_examples=len(results),
            example_results=results,
        )

        # Aggregate retrieval metrics
        if results:
            all_ret_keys = results[0].retrieval_metrics.keys()
            report.aggregate_retrieval = {
                key: sum(r.retrieval_metrics[key] for r in results) / len(results)
                for key in all_ret_keys
            }

            # Collect union of all generation metric keys (some examples may
            # have keywords and others may not, so keys can differ).
            all_gen_keys: set[str] = set()
            for r in results:
                all_gen_keys.update(r.generation_metrics.keys())
            report.aggregate_generation = {
                key: sum(r.generation_metrics.get(key, 0.0) for r in results) / len(results)
                for key in sorted(all_gen_keys)
            }

        logger.info(
            "Evaluation complete — retrieval: %s, generation: %s",
            {k: round(v, 3) for k, v in report.aggregate_retrieval.items()},
            {k: round(v, 3) for k, v in report.aggregate_generation.items()},
        )
        return report
