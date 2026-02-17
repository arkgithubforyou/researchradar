"""Abstract LLM backend interface.

All LLM providers (Ollama, Groq, etc.) implement this interface so
the rest of the system never sees provider-specific details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Knobs for LLM generation."""

    model: str = ""
    max_tokens: int = 1024
    temperature: float = 0.1
    system_prompt: str = (
        "You are a research assistant specializing in NLP and machine learning. "
        "Answer questions based on the provided research paper excerpts. "
        "Cite specific papers when possible. If the provided context does not "
        "contain enough information to answer, say so clearly."
    )


@dataclass
class GenerationResult:
    """LLM response with metadata."""

    answer: str
    model: str
    usage: dict = field(default_factory=dict)


class LLMBackend(ABC):
    """Abstract interface for LLM generation backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> GenerationResult:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt (including injected context).
            system_prompt: Optional system-level instruction.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            GenerationResult with answer text and metadata.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is reachable and ready."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend identifier (e.g., 'ollama', 'groq')."""
        ...
