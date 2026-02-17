"""Tests for the Phase 3 generation module.

Covers LLM backend abstraction, Ollama/Groq backends, context formatting,
prompt building, and the full RAG engine. All tests use mocks — no
model downloads, API calls, or running servers required.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.generation.groq_backend import GroqBackend
from src.generation.llm_backend_base import GenerationConfig, GenerationResult, LLMBackend
from src.generation.ollama_backend import OllamaBackend
from src.generation.rag_engine import (
    RAGEngine,
    RAGResponse,
    build_prompt,
    format_context,
)
from src.retrieval.pipeline import RetrievalResult


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_retrieval_results() -> list[RetrievalResult]:
    """A small set of retrieval results for testing context formatting."""
    return [
        RetrievalResult(
            chunk_id=1,
            paper_id="hf_acl_ocl::P18-1001",
            chunk_text="We propose a novel transformer architecture for machine translation.",
            chunk_type="abstract",
            title="Attention Is All You Need",
            year=2018,
            venue="acl",
            rrf_score=0.05,
            ce_score=0.95,
        ),
        RetrievalResult(
            chunk_id=2,
            paper_id="hf_acl_ocl::D19-1234",
            chunk_text="LoRA decomposes weight updates into low-rank matrices.",
            chunk_type="abstract",
            title="LoRA: Low-Rank Adaptation",
            year=2019,
            venue="emnlp",
            rrf_score=0.03,
            ce_score=0.80,
        ),
        RetrievalResult(
            chunk_id=3,
            paper_id="hf_acl_ocl::D19-1234",
            chunk_text="Low-rank matrices reduce the number of trainable parameters.",
            chunk_type="full_text",
            title="LoRA: Low-Rank Adaptation",
            year=2019,
            venue="emnlp",
            rrf_score=0.02,
            ce_score=0.70,
        ),
    ]


@pytest.fixture
def mock_llm_backend():
    """A mock LLM backend that returns a canned response."""
    backend = MagicMock(spec=LLMBackend)
    backend.backend_name = "mock"
    backend.generate.return_value = GenerationResult(
        answer="Transformers use self-attention [1]. LoRA reduces parameters [2].",
        model="mock-model",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )
    backend.is_available.return_value = True
    return backend


@pytest.fixture
def mock_pipeline():
    """A mock retrieval pipeline that returns fixed results."""
    pipeline = MagicMock()
    pipeline.search.return_value = [
        RetrievalResult(
            chunk_id=1,
            paper_id="hf_acl_ocl::P18-1001",
            chunk_text="We propose a novel transformer architecture.",
            chunk_type="abstract",
            title="Attention Is All You Need",
            year=2018,
            venue="acl",
            rrf_score=0.05,
            ce_score=0.95,
        ),
        RetrievalResult(
            chunk_id=2,
            paper_id="hf_acl_ocl::D19-1234",
            chunk_text="LoRA decomposes weight updates into low-rank matrices.",
            chunk_type="abstract",
            title="LoRA: Low-Rank Adaptation",
            year=2019,
            venue="emnlp",
            rrf_score=0.03,
            ce_score=0.80,
        ),
    ]
    return pipeline


# ── GenerationConfig ─────────────────────────────────────────────────


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.model == ""
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.1
        assert "research assistant" in cfg.system_prompt

    def test_custom_values(self):
        cfg = GenerationConfig(model="qwen2.5:14b", max_tokens=2048, temperature=0.5)
        assert cfg.model == "qwen2.5:14b"
        assert cfg.max_tokens == 2048
        assert cfg.temperature == 0.5


# ── GenerationResult ─────────────────────────────────────────────────


class TestGenerationResult:
    def test_basic(self):
        result = GenerationResult(answer="Hello", model="test")
        assert result.answer == "Hello"
        assert result.model == "test"
        assert result.usage == {}

    def test_with_usage(self):
        result = GenerationResult(
            answer="Hi",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert result.usage["prompt_tokens"] == 10


# ── Context Formatting ───────────────────────────────────────────────


class TestFormatContext:
    def test_empty_results(self):
        context = format_context([])
        assert context == "No relevant papers found."

    def test_single_result(self, sample_retrieval_results):
        context = format_context(sample_retrieval_results[:1])
        assert '[1] "Attention Is All You Need"' in context
        assert "(acl, 2018)" in context
        assert "novel transformer architecture" in context

    def test_multiple_results(self, sample_retrieval_results):
        context = format_context(sample_retrieval_results)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        assert "---" in context  # separator between blocks

    def test_unknown_venue(self):
        result = RetrievalResult(
            chunk_id=1,
            paper_id="test::1",
            chunk_text="Some text.",
            chunk_type="abstract",
            title="Test Paper",
            year=2020,
            venue="",
            rrf_score=0.1,
        )
        context = format_context([result])
        assert "(unknown venue, 2020)" in context


# ── Prompt Building ──────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_context_and_question(self):
        prompt = build_prompt("What is LoRA?", "Context here")
        assert "Context here" in prompt
        assert "What is LoRA?" in prompt

    def test_contains_citation_instruction(self):
        prompt = build_prompt("query", "context")
        assert "Cite papers" in prompt

    def test_contains_excerpts_label(self):
        prompt = build_prompt("query", "context")
        assert "excerpts" in prompt.lower()


# ── OllamaBackend ────────────────────────────────────────────────────


class TestOllamaBackend:
    def test_backend_name(self):
        backend = OllamaBackend.__new__(OllamaBackend)
        backend.host = "http://localhost:11434"
        backend.model = "qwen2.5:14b"
        assert backend.backend_name == "ollama"

    def test_host_trailing_slash_stripped(self):
        backend = OllamaBackend.__new__(OllamaBackend)
        backend.host = "http://localhost:11434/"
        backend.model = "test"
        # __init__ strips trailing slash, but we set it directly here
        b = OllamaBackend(host="http://localhost:11434/")
        assert b.host == "http://localhost:11434"

    @patch("src.generation.ollama_backend.requests.get")
    def test_is_available_success(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        backend = OllamaBackend()
        assert backend.is_available() is True
        mock_get.assert_called_once()

    @patch("src.generation.ollama_backend.requests.get")
    def test_is_available_connection_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.ConnectionError("refused")
        backend = OllamaBackend()
        assert backend.is_available() is False

    @patch("src.generation.ollama_backend.requests.post")
    def test_generate_basic(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Generated answer"},
            "eval_count": 20,
            "prompt_eval_count": 50,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        backend = OllamaBackend(model="test-model")
        result = backend.generate("What is attention?", system_prompt="Be helpful.")

        assert result.answer == "Generated answer"
        assert result.model == "test-model"
        assert result.usage["completion_tokens"] == 20
        assert result.usage["prompt_tokens"] == 50

        # Verify the request payload
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        assert payload["model"] == "test-model"
        assert len(payload["messages"]) == 2  # system + user
        assert payload["messages"][0]["role"] == "system"
        assert payload["stream"] is False

    @patch("src.generation.ollama_backend.requests.post")
    def test_generate_no_system_prompt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Answer"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        backend = OllamaBackend()
        result = backend.generate("question")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        assert len(payload["messages"]) == 1  # user only
        assert result.answer == "Answer"

    @patch("src.generation.ollama_backend.requests.post")
    def test_generate_custom_params(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        backend = OllamaBackend()
        backend.generate("q", max_tokens=512, temperature=0.7)

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        assert payload["options"]["num_predict"] == 512
        assert payload["options"]["temperature"] == 0.7


# ── GroqBackend ──────────────────────────────────────────────────────


class TestGroqBackend:
    def test_backend_name(self):
        with patch("src.generation.groq_backend.Groq"):
            backend = GroqBackend(api_key="test-key")
            assert backend.backend_name == "groq"

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            GroqBackend(api_key="")

    @patch("src.generation.groq_backend.Groq")
    def test_is_available_success(self, MockGroq):
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        backend = GroqBackend(api_key="test-key")
        assert backend.is_available() is True
        mock_client.models.list.assert_called_once()

    @patch("src.generation.groq_backend.Groq")
    def test_is_available_failure(self, MockGroq):
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("auth error")
        MockGroq.return_value = mock_client
        backend = GroqBackend(api_key="bad-key")
        assert backend.is_available() is False

    @patch("src.generation.groq_backend.Groq")
    def test_generate_basic(self, MockGroq):
        mock_client = MagicMock()
        MockGroq.return_value = mock_client

        # Set up mock response
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 80
        mock_usage.completion_tokens = 30
        mock_usage.total_tokens = 110

        mock_message = MagicMock()
        mock_message.content = "Groq answer"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        backend = GroqBackend(api_key="test-key", model="llama-3.3-70b-versatile")
        result = backend.generate("What is NLP?", system_prompt="Be concise.")

        assert result.answer == "Groq answer"
        assert result.model == "llama-3.3-70b-versatile"
        assert result.usage["prompt_tokens"] == 80
        assert result.usage["completion_tokens"] == 30
        assert result.usage["total_tokens"] == 110

        # Verify API call
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"
        assert len(call_kwargs["messages"]) == 2

    @patch("src.generation.groq_backend.Groq")
    def test_generate_no_system_prompt(self, MockGroq):
        mock_client = MagicMock()
        MockGroq.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "Answer"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        backend = GroqBackend(api_key="key")
        result = backend.generate("question")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert len(call_kwargs["messages"]) == 1
        assert result.usage == {}


# ── RAGEngine ────────────────────────────────────────────────────────


class TestRAGEngine:
    def test_query_returns_rag_response(self, mock_pipeline, mock_llm_backend):
        engine = RAGEngine(mock_pipeline, mock_llm_backend)
        response = engine.query("What is attention?")

        assert isinstance(response, RAGResponse)
        assert response.answer != ""
        assert response.model == "mock-model"
        assert len(response.sources) > 0

    def test_query_passes_top_k_to_pipeline(self, mock_pipeline, mock_llm_backend):
        engine = RAGEngine(mock_pipeline, mock_llm_backend)
        engine.query("query", top_k=3)

        mock_pipeline.search.assert_called_once_with(
            query="query", top_k=3, where=None
        )

    def test_query_passes_where_filter(self, mock_pipeline, mock_llm_backend):
        engine = RAGEngine(mock_pipeline, mock_llm_backend)
        engine.query("query", where={"year": {"$gte": 2020}})

        mock_pipeline.search.assert_called_once_with(
            query="query", top_k=5, where={"year": {"$gte": 2020}}
        )

    def test_query_uses_generation_config(self, mock_pipeline, mock_llm_backend):
        config = GenerationConfig(
            max_tokens=2048,
            temperature=0.5,
            system_prompt="Custom prompt.",
        )
        engine = RAGEngine(mock_pipeline, mock_llm_backend, config=config)
        engine.query("query")

        call_kwargs = mock_llm_backend.generate.call_args.kwargs
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["system_prompt"] == "Custom prompt."

    def test_query_deduplicates_sources(self, mock_pipeline, mock_llm_backend):
        """If multiple chunks come from the same paper, sources should be deduplicated."""
        mock_pipeline.search.return_value = [
            RetrievalResult(
                chunk_id=1, paper_id="test::1", chunk_text="chunk 1",
                chunk_type="abstract", title="Paper A", year=2020,
                venue="acl", rrf_score=0.1, ce_score=0.9,
            ),
            RetrievalResult(
                chunk_id=2, paper_id="test::1", chunk_text="chunk 2",
                chunk_type="full_text", title="Paper A", year=2020,
                venue="acl", rrf_score=0.08, ce_score=0.7,
            ),
            RetrievalResult(
                chunk_id=3, paper_id="test::2", chunk_text="chunk 3",
                chunk_type="abstract", title="Paper B", year=2021,
                venue="emnlp", rrf_score=0.05, ce_score=0.6,
            ),
        ]

        engine = RAGEngine(mock_pipeline, mock_llm_backend)
        response = engine.query("query")

        # 3 chunks but only 2 unique papers
        assert len(response.sources) == 2
        paper_ids = {s["paper_id"] for s in response.sources}
        assert paper_ids == {"test::1", "test::2"}

    def test_query_empty_retrieval(self, mock_pipeline, mock_llm_backend):
        """When retrieval returns nothing, the LLM should still be called."""
        mock_pipeline.search.return_value = []

        engine = RAGEngine(mock_pipeline, mock_llm_backend)
        response = engine.query("obscure question")

        assert response.sources == []
        mock_llm_backend.generate.assert_called_once()
        # The context should say no relevant papers found
        call_kwargs = mock_llm_backend.generate.call_args.kwargs
        assert "No relevant papers found" in call_kwargs["prompt"]

    def test_query_includes_context_in_prompt(self, mock_pipeline, mock_llm_backend):
        engine = RAGEngine(mock_pipeline, mock_llm_backend)
        engine.query("What is attention?")

        call_kwargs = mock_llm_backend.generate.call_args.kwargs
        prompt = call_kwargs["prompt"]
        assert "Attention Is All You Need" in prompt
        assert "What is attention?" in prompt

    def test_source_metadata_fields(self, mock_pipeline, mock_llm_backend):
        engine = RAGEngine(mock_pipeline, mock_llm_backend)
        response = engine.query("query")

        source = response.sources[0]
        assert "paper_id" in source
        assert "title" in source
        assert "year" in source
        assert "venue" in source
        assert "chunk_type" in source


# ── RAGResponse ──────────────────────────────────────────────────────


class TestRAGResponse:
    def test_basic(self):
        resp = RAGResponse(answer="test", sources=[], model="m")
        assert resp.answer == "test"
        assert resp.sources == []
        assert resp.usage == {}

    def test_with_sources(self):
        sources = [{"paper_id": "test::1", "title": "Paper"}]
        resp = RAGResponse(answer="answer", sources=sources, model="m")
        assert len(resp.sources) == 1
