"""Groq cloud LLM backend.

Uses the Groq API for fast inference with models like Llama, Mixtral, etc.
Requires a GROQ_API_KEY.
"""

import logging

from groq import Groq

from src.generation.llm_backend_base import GenerationResult, LLMBackend

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqBackend(LLMBackend):
    """LLM backend using the Groq cloud API."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("Groq API key is required")
        self.model = model
        self._client = Groq(api_key=api_key)

    @property
    def backend_name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        """Check if the Groq API is reachable with valid credentials."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> GenerationResult:
        """Generate via the Groq chat completions API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info("Groq request: model=%s, tokens=%d", self.model, max_tokens)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        answer = choice.message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.info("Groq response: %d chars, usage=%s", len(answer), usage)

        return GenerationResult(answer=answer, model=self.model, usage=usage)
