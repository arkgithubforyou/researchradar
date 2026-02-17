"""Evaluation dataset loader and schema.

Loads evaluation question-answer pairs from a JSON file. Each example
contains a query, expected relevant paper IDs, a reference answer, and
optional keywords for coverage scoring.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """A single evaluation example."""

    query: str
    relevant_paper_ids: list[str]
    reference_answer: str
    keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalDataset:
    """Collection of evaluation examples with metadata."""

    name: str
    examples: list[EvalExample]
    description: str = ""

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


def load_eval_dataset(path: str | Path) -> EvalDataset:
    """Load an evaluation dataset from a JSON file.

    Expected JSON format:
    {
        "name": "researchradar-eval-v1",
        "description": "...",
        "examples": [
            {
                "query": "What is LoRA?",
                "relevant_paper_ids": ["hf_acl_ocl::D19-1234"],
                "reference_answer": "LoRA is a parameter-efficient...",
                "keywords": ["low-rank", "parameter-efficient", "adaptation"],
                "metadata": {"category": "methods"}
            }
        ]
    }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    examples = [
        EvalExample(
            query=ex["query"],
            relevant_paper_ids=ex["relevant_paper_ids"],
            reference_answer=ex["reference_answer"],
            keywords=ex.get("keywords", []),
            metadata=ex.get("metadata", {}),
        )
        for ex in data["examples"]
    ]

    ds = EvalDataset(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        examples=examples,
    )
    logger.info("Loaded eval dataset %r: %d examples", ds.name, len(ds))
    return ds


def create_sample_dataset() -> EvalDataset:
    """Create a small built-in eval dataset for smoke testing.

    These examples use the fixture papers from conftest.py so they
    can run without a real corpus.
    """
    return EvalDataset(
        name="smoke-test",
        description="Minimal eval set for testing the evaluation pipeline.",
        examples=[
            EvalExample(
                query="What is the transformer architecture used for?",
                relevant_paper_ids=["hf_acl_ocl::P18-1001"],
                reference_answer=(
                    "The transformer architecture is used for machine translation, "
                    "relying on self-attention mechanisms instead of recurrence."
                ),
                keywords=["transformer", "machine translation", "attention"],
                metadata={"category": "methods"},
            ),
            EvalExample(
                query="How does LoRA reduce the number of trainable parameters?",
                relevant_paper_ids=["hf_acl_ocl::D19-1234"],
                reference_answer=(
                    "LoRA decomposes weight update matrices into low-rank factors, "
                    "drastically reducing the number of trainable parameters "
                    "during fine-tuning."
                ),
                keywords=["LoRA", "low-rank", "parameter-efficient", "fine-tuning"],
                metadata={"category": "methods"},
            ),
            EvalExample(
                query="What are the recent approaches to contrastive learning in NLP?",
                relevant_paper_ids=["acl_anthology::2022.acl-long.100"],
                reference_answer=(
                    "Contrastive learning has been applied to various NLP tasks "
                    "including sentence embedding, text classification, and relation extraction."
                ),
                keywords=["contrastive learning", "NLP", "sentence embedding"],
                metadata={"category": "survey"},
            ),
        ],
    )
