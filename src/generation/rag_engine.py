"""End-to-end RAG engine.

Orchestrates retrieval → context formatting → LLM generation.
This is the main entry point for answering questions about research papers.
"""

import logging
from dataclasses import dataclass, field

from src.generation.llm_backend_base import GenerationConfig, GenerationResult, LLMBackend
from src.retrieval.pipeline import RetrievalPipeline, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Full RAG response with answer, sources, and metadata."""

    answer: str
    sources: list[dict]
    model: str
    usage: dict = field(default_factory=dict)


def format_context(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a context block for the LLM prompt.

    Each chunk is wrapped with its paper metadata so the LLM can cite sources.
    """
    if not results:
        return "No relevant papers found."

    blocks = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] \"{r.title}\" ({r.venue or 'unknown venue'}, {r.year})"
        blocks.append(f"{header}\n{r.chunk_text}")

    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    """Build the user prompt with context and question."""
    return (
        f"Below are excerpts from relevant research papers.\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Based on the above excerpts, answer the following question. "
        f"Cite papers by their number (e.g., [1], [2]) when referencing specific findings.\n\n"
        f"Question: {query}"
    )


class RAGEngine:
    """Orchestrates retrieval and generation for end-to-end RAG.

    Usage:
        engine = RAGEngine(pipeline, llm_backend)
        response = engine.query("What is LoRA?")
        print(response.answer)
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        llm_backend: LLMBackend,
        config: GenerationConfig | None = None,
    ):
        self.pipeline = retrieval_pipeline
        self.llm = llm_backend
        self.config = config or GenerationConfig()

    def query(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> RAGResponse:
        """Answer a question using retrieval-augmented generation.

        Args:
            question: The user's natural-language question.
            top_k: Number of chunks to retrieve as context.
            where: Optional metadata filter for retrieval (e.g., year, venue).

        Returns:
            RAGResponse with the answer, source papers, and metadata.
        """
        logger.info("RAG query: %r (top_k=%d)", question, top_k)

        # Step 1: Retrieve relevant chunks
        results = self.pipeline.search(query=question, top_k=top_k, where=where)
        logger.info("Retrieved %d chunks", len(results))

        # Step 2: Format context
        context = format_context(results)

        # Step 3: Build prompt and generate
        prompt = build_prompt(question, context)
        gen_result: GenerationResult = self.llm.generate(
            prompt=prompt,
            system_prompt=self.config.system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Step 4: Build source list (deduplicated by paper_id)
        seen_papers: set[str] = set()
        sources = []
        for r in results:
            if r.paper_id not in seen_papers:
                seen_papers.add(r.paper_id)
                sources.append({
                    "paper_id": r.paper_id,
                    "title": r.title,
                    "year": r.year,
                    "venue": r.venue,
                    "chunk_type": r.chunk_type,
                })

        return RAGResponse(
            answer=gen_result.answer,
            sources=sources,
            model=gen_result.model,
            usage=gen_result.usage,
        )
