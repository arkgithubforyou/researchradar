"""Smoke test for Phase 3: LLM backend + RAG engine.

Usage:
    python scripts/smoke_test_llm.py                          # Ollama (default)
    python scripts/smoke_test_llm.py --backend groq            # Groq cloud
    python scripts/smoke_test_llm.py --query "Explain BERT"    # Custom query
    python scripts/smoke_test_llm.py --backend-only            # Skip RAG test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.generation.llm_backend_base import GenerationConfig, LLMBackend
from src.generation.groq_backend import GroqBackend
from src.generation.ollama_backend import OllamaBackend
from src.generation.rag_engine import build_prompt, format_context
from src.retrieval.pipeline import RetrievalResult


FAKE_RESULTS = [
    RetrievalResult(
        chunk_id=1,
        paper_id="hf_acl_ocl::P18-1001",
        chunk_text="We propose a novel transformer architecture that relies "
        "entirely on attention mechanisms, dispensing with recurrence "
        "and convolutions entirely.",
        chunk_type="abstract",
        title="Attention Is All You Need",
        year=2017,
        venue="neurips",
        rrf_score=0.05,
        ce_score=0.95,
    ),
    RetrievalResult(
        chunk_id=2,
        paper_id="hf_acl_ocl::D21-1234",
        chunk_text="LoRA freezes the pre-trained model weights and injects "
        "trainable rank decomposition matrices into each layer, "
        "greatly reducing trainable parameters for downstream tasks.",
        chunk_type="abstract",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        year=2021,
        venue="iclr",
        rrf_score=0.03,
        ce_score=0.80,
    ),
]


def create_backend(backend_name: str) -> LLMBackend:
    """Create the appropriate LLM backend."""
    if backend_name == "groq":
        config = get_config()
        if not config.groq_api_key:
            print("ERROR: GROQ_API_KEY not set in .env")
            sys.exit(1)
        return GroqBackend(api_key=config.groq_api_key)
    return OllamaBackend(model="qwen2.5:14b")


def test_backend(backend: LLMBackend) -> bool:
    """Test the backend with a simple query."""
    print("=" * 60)
    print(f"1. BACKEND TEST — {backend.backend_name}")
    print("=" * 60)

    print(f"Available: {backend.is_available()}")
    if not backend.is_available():
        print(f"ERROR: {backend.backend_name} backend not reachable.")
        return False

    result = backend.generate(
        "What is LoRA in one sentence?",
        system_prompt="Be concise.",
        max_tokens=100,
    )
    print(f"Model:  {result.model}")
    print(f"Answer: {result.answer}")
    print(f"Usage:  {result.usage}")
    return True


def test_context_formatting():
    """Test context formatting with fake retrieval results."""
    print("\n" + "=" * 60)
    print("2. CONTEXT FORMATTING TEST")
    print("=" * 60)

    context = format_context(FAKE_RESULTS)
    prompt = build_prompt("How does LoRA relate to transformers?", context)
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Preview:\n{prompt[:500]}...")


def test_rag_with_fake_context(backend: LLMBackend, query: str, max_tokens: int = 1024):
    """Test the full generate step with fake retrieval results."""
    print("\n" + "=" * 60)
    print(f"3. RAG GENERATION TEST ({backend.backend_name}, fake retrieval)")
    print("=" * 60)

    context = format_context(FAKE_RESULTS)
    prompt = build_prompt(query, context)

    config = GenerationConfig(max_tokens=max_tokens, temperature=0.1)
    result = backend.generate(
        prompt=prompt,
        system_prompt=config.system_prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    print(f"Question: {query}")
    print(f"Answer:\n{result.answer}")
    print(f"\nUsage: {result.usage}")


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Phase 3")
    parser.add_argument("--backend", choices=["ollama", "groq"], default="ollama", help="LLM backend")
    parser.add_argument("--backend-only", action="store_true", help="Only test the LLM backend")
    parser.add_argument("--query", type=str, help="Custom query")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens (default: 1024)")
    args = parser.parse_args()

    backend = create_backend(args.backend)

    if not test_backend(backend):
        sys.exit(1)

    if args.backend_only:
        return

    query = args.query or "How does LoRA relate to transformers?"
    test_context_formatting()
    test_rag_with_fake_context(backend, query, max_tokens=args.max_tokens)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
