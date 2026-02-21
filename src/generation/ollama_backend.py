"""Ollama LLM backend.

Calls the local Ollama REST API for generation using models like
Qwen2.5-14B. Requires a running Ollama server.
"""

import logging

import requests

from src.generation.llm_backend_base import GenerationResult, LLMBackend

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5:14b"


class OllamaBackend(LLMBackend):
    """LLM backend that talks to a local Ollama server."""

    def __init__(self, host: str = "http://localhost:11434", model: str = DEFAULT_MODEL):
        self.host = host.rstrip("/")
        self.model = model

    @property
    def backend_name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Check if the Ollama server is running."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> GenerationResult:
        """Generate via Ollama /api/chat endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        logger.info("Ollama request: model=%s, tokens=%d", self.model, max_tokens)

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            logger.error("Ollama server unreachable at %s: %s", self.host, exc)
            raise RuntimeError(
                f"Ollama server unreachable at {self.host}"
            ) from exc
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        data = resp.json()

        answer = data.get("message", {}).get("content", "")
        usage = {}
        if "eval_count" in data:
            usage["completion_tokens"] = data["eval_count"]
        if "prompt_eval_count" in data:
            usage["prompt_tokens"] = data["prompt_eval_count"]

        logger.info("Ollama response: %d chars, usage=%s", len(answer), usage)

        return GenerationResult(answer=answer, model=self.model, usage=usage)
