"""Ablation study framework.

Systematically varies retrieval pipeline knobs (BM25 weight, vector weight,
reranker on/off, top_k values) and measures the impact on retrieval quality.
Each configuration is evaluated on the same dataset to produce a comparison table.
"""

import logging
from dataclasses import dataclass, field

from src.evaluation.dataset import EvalDataset, EvalExample
from src.evaluation.metrics import evaluate_retrieval
from src.retrieval.pipeline import RetrievalConfig, RetrievalPipeline

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """A single ablation experiment configuration."""

    name: str
    retrieval_config: RetrievalConfig
    description: str = ""


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""

    config_name: str
    description: str
    num_examples: int
    aggregate_metrics: dict = field(default_factory=dict)
    per_example: list[dict] = field(default_factory=list)


@dataclass
class AblationReport:
    """Full ablation study report across all configurations."""

    results: list[AblationResult]
    dataset_name: str
    num_examples: int

    def summary_table(self) -> list[dict]:
        """Produce a flat summary table for easy comparison."""
        rows = []
        for r in self.results:
            row = {"config": r.config_name, "description": r.description}
            row.update(r.aggregate_metrics)
            rows.append(row)
        return rows


def default_ablation_configs() -> list[AblationConfig]:
    """Standard ablation configurations for studying retrieval component contributions."""
    return [
        AblationConfig(
            name="full_pipeline",
            description="BM25 + Vector + Reranker (default)",
            retrieval_config=RetrievalConfig(
                bm25_weight=1, vector_weight=1, use_reranker=True,
            ),
        ),
        AblationConfig(
            name="no_reranker",
            description="BM25 + Vector, no cross-encoder reranking",
            retrieval_config=RetrievalConfig(
                bm25_weight=1, vector_weight=1, use_reranker=False,
            ),
        ),
        AblationConfig(
            name="bm25_only",
            description="BM25 only (vector weight=0), with reranker",
            retrieval_config=RetrievalConfig(
                bm25_weight=1, vector_weight=0, use_reranker=True,
            ),
        ),
        AblationConfig(
            name="vector_only",
            description="Vector only (BM25 weight=0), with reranker",
            retrieval_config=RetrievalConfig(
                bm25_weight=0, vector_weight=1, use_reranker=True,
            ),
        ),
        AblationConfig(
            name="bm25_heavy",
            description="BM25-heavy hybrid (3:1) + reranker",
            retrieval_config=RetrievalConfig(
                bm25_weight=3, vector_weight=1, use_reranker=True,
            ),
        ),
        AblationConfig(
            name="vector_heavy",
            description="Vector-heavy hybrid (1:3) + reranker",
            retrieval_config=RetrievalConfig(
                bm25_weight=1, vector_weight=3, use_reranker=True,
            ),
        ),
    ]


class AblationRunner:
    """Runs ablation experiments over a retrieval pipeline."""

    def __init__(self, pipeline: RetrievalPipeline, top_k: int = 5):
        self.pipeline = pipeline
        self.top_k = top_k

    def run_config(
        self,
        config: AblationConfig,
        dataset: EvalDataset,
    ) -> AblationResult:
        """Run one ablation configuration against the eval dataset."""
        logger.info("Ablation: running config %r", config.name)

        # Swap pipeline config
        original_config = self.pipeline.config
        self.pipeline.config = config.retrieval_config

        # Also toggle reranker
        original_reranker = self.pipeline.reranker
        if not config.retrieval_config.use_reranker:
            self.pipeline.reranker = None

        per_example = []
        for example in dataset:
            results = self.pipeline.search(
                query=example.query, top_k=self.top_k,
            )
            retrieved_ids = [r.paper_id for r in results]
            relevant_ids = set(example.relevant_paper_ids)
            metrics = evaluate_retrieval(retrieved_ids, relevant_ids)
            per_example.append({"query": example.query, "metrics": metrics})

        # Restore original config
        self.pipeline.config = original_config
        self.pipeline.reranker = original_reranker

        # Aggregate
        aggregate: dict = {}
        if per_example:
            all_keys = per_example[0]["metrics"].keys()
            aggregate = {
                key: sum(ex["metrics"][key] for ex in per_example) / len(per_example)
                for key in all_keys
            }

        return AblationResult(
            config_name=config.name,
            description=config.description,
            num_examples=len(per_example),
            aggregate_metrics=aggregate,
            per_example=per_example,
        )

    def run_study(
        self,
        dataset: EvalDataset,
        configs: list[AblationConfig] | None = None,
    ) -> AblationReport:
        """Run the full ablation study across all configurations."""
        if configs is None:
            configs = default_ablation_configs()

        logger.info(
            "Starting ablation study: %d configs x %d examples",
            len(configs), len(dataset),
        )

        results = [self.run_config(cfg, dataset) for cfg in configs]

        report = AblationReport(
            results=results,
            dataset_name=dataset.name,
            num_examples=len(dataset),
        )

        # Log summary
        for row in report.summary_table():
            logger.info(
                "  %-20s  MRR=%.3f  P@5=%.3f  R@5=%.3f  NDCG@5=%.3f",
                row["config"],
                row.get("mrr", 0),
                row.get("precision@5", 0),
                row.get("recall@5", 0),
                row.get("ndcg@5", 0),
            )

        return report
