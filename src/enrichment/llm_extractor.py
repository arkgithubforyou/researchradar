"""LLM-based entity extraction.

Uses an LLM backend (Ollama/Groq) to extract structured entities
(methods, datasets, tasks, topics) from paper abstracts via JSON-mode prompting.
"""

import json
import logging
import re

from src.generation.llm_backend_base import LLMBackend

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = (
    "You are an NLP research paper entity extractor. "
    "Extract structured entities from paper text. "
    "Always respond with valid JSON only, no markdown fences, no commentary."
)

EXTRACTION_USER_PROMPT = """\
Extract entities from the following research paper text.

Title: {title}
Text: {text}

Return a JSON object with exactly these keys:
{{
  "methods": [
    {{"method_name": "...", "method_type": "model|technique|framework|null"}}
  ],
  "datasets": [
    {{"dataset_name": "...", "task_type": "classification|QA|generation|translation|NER|summarization|NLI|other|null"}}
  ],
  "tasks": ["task name 1", "task name 2"],
  "topics": ["topic 1", "topic 2"]
}}

Rules:
- "methods" includes models (BERT, GPT), techniques (LoRA, contrastive learning), and frameworks.
- "datasets" includes benchmark datasets (SQuAD, GLUE) with their task type if identifiable.
- "tasks" includes specific NLP/ML tasks (e.g., "machine translation", "sentiment analysis").
- "topics" includes high-level research themes/areas (e.g., "multimodal", "agentic", "fairness", "low-resource", "biomedical NLP", "explainability"). Topics are broader than tasks.
- Use canonical names (e.g., "BERT" not "bert", "SQuAD" not "squad").
- Only extract entities explicitly mentioned in the text. Do not infer.
- Return empty lists if no entities are found for a category.
- Return ONLY the JSON object, nothing else."""


def _parse_llm_json(raw: str) -> dict | None:
    """Parse JSON from LLM output, handling common formatting issues.

    LLMs sometimes wrap JSON in markdown fences or add trailing text.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find the first JSON object in the text
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _validate_extraction(data: dict) -> dict:
    """Validate and normalize extracted entities.

    Ensures the output has the expected structure, dropping malformed entries.
    """
    result = {"methods": [], "datasets": [], "tasks": [], "topics": []}

    # Validate methods
    for m in data.get("methods", []):
        if isinstance(m, dict) and "method_name" in m:
            name = str(m["method_name"]).strip()
            if name:
                method_type = m.get("method_type")
                if method_type not in ("model", "technique", "framework", None):
                    method_type = None
                result["methods"].append({
                    "method_name": name,
                    "method_type": method_type,
                })

    # Validate datasets
    for d in data.get("datasets", []):
        if isinstance(d, dict) and "dataset_name" in d:
            name = str(d["dataset_name"]).strip()
            if name:
                result["datasets"].append({
                    "dataset_name": name,
                    "task_type": d.get("task_type"),
                })

    # Validate tasks
    for t in data.get("tasks", []):
        if isinstance(t, str):
            name = t.strip()
            if name:
                result["tasks"].append(name)

    # Validate topics
    for t in data.get("topics", []):
        if isinstance(t, str):
            name = t.strip()
            if name:
                result["topics"].append(name)

    return result


def extract_entities_llm(
    llm: LLMBackend,
    title: str,
    text: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict | None:
    """Extract entities from a single paper using an LLM.

    Args:
        llm: An LLM backend (Ollama, Groq, etc.).
        title: The paper title.
        text: The paper abstract or full text excerpt.
        max_tokens: Max tokens for the LLM response.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        Validated extraction dict, or None if extraction fails.
        {
            "methods": [{"method_name": str, "method_type": str | None}, ...],
            "datasets": [{"dataset_name": str, "task_type": str | None}, ...],
            "tasks": [str, ...],
            "topics": [str, ...],
        }
    """
    prompt = EXTRACTION_USER_PROMPT.format(title=title, text=text)

    try:
        result = llm.generate(
            prompt=prompt,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception:
        logger.exception("LLM extraction failed for %r", title)
        return None

    parsed = _parse_llm_json(result.answer)
    if parsed is None:
        logger.warning(
            "Failed to parse LLM JSON for %r: %s", title, result.answer[:200]
        )
        return None

    return _validate_extraction(parsed)
