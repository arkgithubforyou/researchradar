"""Tests for the enrichment module: regex extractor, LLM extractor, pipeline."""

from unittest.mock import MagicMock

import pytest

from src.enrichment.llm_extractor import (
    _parse_llm_json,
    _validate_extraction,
    extract_entities_llm,
)
from src.enrichment.pipeline import EnrichmentPipeline, EnrichmentStats
from src.enrichment.regex_extractor import (
    TOPIC_NAMES,
    extract_all_regex,
    extract_datasets_regex,
    extract_methods_regex,
    extract_tasks_regex,
    extract_topics_regex,
)
from src.generation.llm_backend_base import GenerationResult


# ── Regex extractor tests ────────────────────────────────────────────


class TestRegexMethods:
    def test_extracts_bert(self):
        text = "We fine-tune BERT on downstream tasks."
        results = extract_methods_regex(text)
        names = [m["method_name"] for m in results]
        assert "BERT" in names

    def test_extracts_multiple_methods(self):
        text = "We compare BERT, GPT-2, and LoRA for text classification."
        results = extract_methods_regex(text)
        names = [m["method_name"] for m in results]
        assert "BERT" in names
        assert "GPT-2" in names
        assert "LoRA" in names

    def test_case_insensitive(self):
        text = "We apply lora to the bert model."
        results = extract_methods_regex(text)
        names = [m["method_name"] for m in results]
        assert "LoRA" in names
        assert "BERT" in names

    def test_no_methods(self):
        text = "This paper studies linguistic phenomena."
        results = extract_methods_regex(text)
        assert results == []

    def test_deduplication(self):
        text = "BERT is great. We use BERT extensively. BERT outperforms others."
        results = extract_methods_regex(text)
        names = [m["method_name"] for m in results]
        assert names.count("BERT") == 1

    def test_word_boundary(self):
        text = "We use ALBERT for feature extraction."
        results = extract_methods_regex(text)
        names = [m["method_name"] for m in results]
        assert "ALBERT" in names

    def test_method_type_is_none(self):
        text = "We use Transformer models."
        results = extract_methods_regex(text)
        for m in results:
            assert m["method_type"] is None


class TestRegexDatasets:
    def test_extracts_squad(self):
        text = "We evaluate on SQuAD and MNLI benchmarks."
        results = extract_datasets_regex(text)
        names = [d["dataset_name"] for d in results]
        assert "SQuAD" in names
        assert "MNLI" in names

    def test_extracts_glue(self):
        text = "Results on the GLUE benchmark show improvements."
        results = extract_datasets_regex(text)
        names = [d["dataset_name"] for d in results]
        assert "GLUE" in names

    def test_no_datasets(self):
        text = "We propose a new architecture."
        results = extract_datasets_regex(text)
        assert results == []

    def test_task_type_is_none(self):
        text = "We evaluate on SQuAD."
        results = extract_datasets_regex(text)
        for d in results:
            assert d["task_type"] is None


class TestRegexTasks:
    def test_extracts_tasks(self):
        text = "We address machine translation and sentiment analysis."
        results = extract_tasks_regex(text)
        assert "machine translation" in results
        assert "sentiment analysis" in results

    def test_extracts_ner(self):
        text = "Our model improves named entity recognition."
        results = extract_tasks_regex(text)
        assert "named entity recognition" in results

    def test_no_tasks(self):
        text = "This paper presents a novel architecture."
        results = extract_tasks_regex(text)
        assert results == []


class TestRegexTopics:
    def test_extracts_multimodal(self):
        text = "We study multimodal learning for vision and language tasks."
        results = extract_topics_regex(text)
        assert "multimodal" in results

    def test_extracts_multiple_topics(self):
        text = "We address fairness and explainability in low-resource settings."
        results = extract_topics_regex(text)
        assert "fairness" in results
        assert "explainability" in results
        assert "low-resource" in results

    def test_no_topics(self):
        text = "This paper presents results."
        results = extract_topics_regex(text)
        assert results == []

    def test_no_overlap_with_other_lists(self):
        """TOPIC_NAMES should have zero overlap with methods, datasets, tasks."""
        from src.enrichment.regex_extractor import (
            DATASET_NAMES,
            METHOD_NAMES,
            TASK_NAMES,
        )
        other = {n.lower() for n in METHOD_NAMES + DATASET_NAMES + TASK_NAMES}
        for topic in TOPIC_NAMES:
            assert topic.lower() not in other, f"{topic!r} overlaps with another list"


class TestRegexAll:
    def test_extract_all(self):
        text = (
            "We fine-tune BERT on SQuAD for question answering "
            "and evaluate on GLUE for text classification."
        )
        result = extract_all_regex(text)
        assert "BERT" in [m["method_name"] for m in result["methods"]]
        assert "SQuAD" in [d["dataset_name"] for d in result["datasets"]]
        assert "question answering" in result["tasks"]

    def test_extract_all_includes_topics(self):
        text = "We study multimodal fairness in biomedical NLP."
        result = extract_all_regex(text)
        assert "topics" in result
        assert "multimodal" in result["topics"]


# ── LLM extractor tests ─────────────────────────────────────────────


class TestParseJson:
    def test_clean_json(self):
        raw = '{"methods": [], "datasets": [], "tasks": []}'
        assert _parse_llm_json(raw) == {"methods": [], "datasets": [], "tasks": []}

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"methods": [], "datasets": [], "tasks": []}\n```'
        assert _parse_llm_json(raw) == {"methods": [], "datasets": [], "tasks": []}

    def test_json_with_trailing_text(self):
        raw = '{"methods": [], "datasets": [], "tasks": []}\nHere are the results.'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result["methods"] == []

    def test_invalid_json(self):
        assert _parse_llm_json("not json at all") is None

    def test_empty_string(self):
        assert _parse_llm_json("") is None


class TestValidateExtraction:
    def test_valid_input(self):
        data = {
            "methods": [{"method_name": "BERT", "method_type": "model"}],
            "datasets": [{"dataset_name": "SQuAD", "task_type": "QA"}],
            "tasks": ["question answering"],
        }
        result = _validate_extraction(data)
        assert len(result["methods"]) == 1
        assert result["methods"][0]["method_name"] == "BERT"
        assert result["methods"][0]["method_type"] == "model"
        assert len(result["datasets"]) == 1
        assert len(result["tasks"]) == 1

    def test_strips_whitespace(self):
        data = {
            "methods": [{"method_name": "  BERT  ", "method_type": "model"}],
            "datasets": [],
            "tasks": ["  NER  "],
        }
        result = _validate_extraction(data)
        assert result["methods"][0]["method_name"] == "BERT"
        assert result["tasks"][0] == "NER"

    def test_drops_empty_names(self):
        data = {
            "methods": [{"method_name": "", "method_type": "model"}],
            "datasets": [{"dataset_name": "  "}],
            "tasks": [""],
        }
        result = _validate_extraction(data)
        assert result["methods"] == []
        assert result["datasets"] == []
        assert result["tasks"] == []

    def test_invalid_method_type_set_to_none(self):
        data = {
            "methods": [{"method_name": "BERT", "method_type": "unknown"}],
            "datasets": [],
            "tasks": [],
        }
        result = _validate_extraction(data)
        assert result["methods"][0]["method_type"] is None

    def test_validates_topics(self):
        data = {
            "methods": [],
            "datasets": [],
            "tasks": [],
            "topics": ["multimodal", 123, "  fairness  ", ""],
        }
        result = _validate_extraction(data)
        assert result["topics"] == ["multimodal", "fairness"]

    def test_missing_keys_returns_empty(self):
        result = _validate_extraction({})
        assert result == {"methods": [], "datasets": [], "tasks": [], "topics": []}

    def test_malformed_entries_skipped(self):
        data = {
            "methods": ["just a string", {"method_name": "BERT"}],
            "datasets": [42, {"dataset_name": "SQuAD"}],
            "tasks": [123, "valid task"],
        }
        result = _validate_extraction(data)
        assert len(result["methods"]) == 1
        assert result["methods"][0]["method_name"] == "BERT"
        assert len(result["datasets"]) == 1
        assert len(result["tasks"]) == 1


class TestLLMExtraction:
    def test_successful_extraction(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = GenerationResult(
            answer='{"methods": [{"method_name": "BERT", "method_type": "model"}], '
            '"datasets": [{"dataset_name": "SQuAD", "task_type": "QA"}], '
            '"tasks": ["question answering"], '
            '"topics": ["multimodal"]}',
            model="test",
        )
        result = extract_entities_llm(mock_llm, "Test Paper", "We fine-tune BERT on SQuAD.")
        assert result is not None
        assert len(result["methods"]) == 1
        assert result["methods"][0]["method_name"] == "BERT"
        assert "multimodal" in result["topics"]

    def test_llm_returns_invalid_json(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = GenerationResult(
            answer="I couldn't extract anything meaningful.", model="test"
        )
        result = extract_entities_llm(mock_llm, "Test Paper", "Some text.")
        assert result is None

    def test_llm_exception_returns_none(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("API error")
        result = extract_entities_llm(mock_llm, "Test Paper", "Some text.")
        assert result is None


# ── Pipeline tests ───────────────────────────────────────────────────


class TestEnrichmentPipeline:
    def test_regex_only_pipeline(self, tmp_db, sample_papers):
        """Pipeline with no LLM should use regex extraction."""
        tmp_db.insert_papers(sample_papers)
        pipeline = EnrichmentPipeline(db=tmp_db, llm_backend=None, use_regex_fallback=True)

        stats = pipeline.enrich_all()
        assert stats.papers_processed > 0
        assert stats.llm_extractions == 0
        assert stats.regex_fallbacks == stats.papers_processed

    def test_enrichment_populates_db(self, tmp_db, sample_papers):
        """Verify entities are stored in the database after enrichment."""
        tmp_db.insert_papers(sample_papers)
        pipeline = EnrichmentPipeline(db=tmp_db, llm_backend=None)

        pipeline.enrich_all()

        paper = tmp_db.get_paper_by_id("hf_acl_ocl::P18-1001")
        methods = paper["methods"]
        method_names = [m["method_name"].lower() for m in methods]
        assert "transformer" in method_names

    def test_enrich_paper_single(self, tmp_db, sample_papers):
        """Test single paper enrichment."""
        tmp_db.insert_papers(sample_papers)
        pipeline = EnrichmentPipeline(db=tmp_db, llm_backend=None)

        paper = {"id": "hf_acl_ocl::D19-1234", "title": "LoRA: Low-Rank Adaptation", "abstract": "We propose LoRA."}
        result = pipeline.enrich_paper(paper)
        assert "LoRA" in [m["method_name"] for m in result["methods"]]

    def test_skips_already_enriched(self, tmp_db, sample_papers):
        """Papers already enriched should be skipped on second run."""
        tmp_db.insert_papers(sample_papers)
        pipeline = EnrichmentPipeline(db=tmp_db, llm_backend=None)

        stats1 = pipeline.enrich_all()
        stats2 = pipeline.enrich_all()

        assert stats1.papers_processed > 0
        assert stats2.papers_processed == 0

    def test_max_papers_limit(self, tmp_db, sample_papers):
        """max_papers should limit the number processed."""
        tmp_db.insert_papers(sample_papers)
        pipeline = EnrichmentPipeline(db=tmp_db, llm_backend=None)

        stats = pipeline.enrich_all(max_papers=1)
        assert stats.papers_processed == 1

    def test_llm_with_regex_fallback(self, tmp_db, sample_papers):
        """When LLM fails, regex should fill in."""
        tmp_db.insert_papers(sample_papers)

        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.backend_name = "mock"
        mock_llm.generate.return_value = GenerationResult(
            answer="not valid json", model="mock"
        )

        pipeline = EnrichmentPipeline(db=tmp_db, llm_backend=mock_llm)
        stats = pipeline.enrich_all()

        assert stats.papers_processed > 0
        assert stats.regex_fallbacks > 0

    def test_llm_unavailable_uses_regex(self, tmp_db, sample_papers):
        """When LLM is not available, pipeline should use regex only."""
        tmp_db.insert_papers(sample_papers)

        mock_llm = MagicMock()
        mock_llm.is_available.return_value = False

        pipeline = EnrichmentPipeline(db=tmp_db, llm_backend=mock_llm)
        stats = pipeline.enrich_all()

        assert stats.papers_processed > 0
        assert stats.llm_extractions == 0
        assert stats.regex_fallbacks == stats.papers_processed

    def test_merge_llm_and_regex(self, tmp_db):
        """LLM results should be merged with regex results."""
        pipeline = EnrichmentPipeline(db=tmp_db, llm_backend=None)

        llm_result = {
            "methods": [{"method_name": "CustomModel", "method_type": "model"}],
            "datasets": [],
            "tasks": ["text classification"],
            "topics": ["fairness"],
        }
        regex_result = {
            "methods": [
                {"method_name": "BERT", "method_type": None},
                {"method_name": "custommodel", "method_type": None},
            ],
            "datasets": [{"dataset_name": "GLUE", "task_type": None}],
            "tasks": ["text classification", "NER"],
            "topics": ["fairness", "multimodal"],
        }

        merged = pipeline._merge_extractions(llm_result, regex_result)

        method_names = [m["method_name"] for m in merged["methods"]]
        assert "CustomModel" in method_names
        assert "BERT" in method_names
        assert len([n for n in method_names if n.lower() == "custommodel"]) == 1

        assert len(merged["datasets"]) == 1
        assert merged["datasets"][0]["dataset_name"] == "GLUE"

        assert "text classification" in merged["tasks"]
        assert "NER" in merged["tasks"]
        assert merged["tasks"].count("text classification") == 1

        assert "fairness" in merged["topics"]
        assert "multimodal" in merged["topics"]
        assert merged["topics"].count("fairness") == 1

    def test_merge_none_llm_returns_regex(self, tmp_db):
        """If LLM result is None, merge should return regex result."""
        pipeline = EnrichmentPipeline(db=tmp_db, llm_backend=None)
        regex_result = {"methods": [{"method_name": "BERT", "method_type": None}], "datasets": [], "tasks": [], "topics": []}
        merged = pipeline._merge_extractions(None, regex_result)
        assert merged == regex_result
