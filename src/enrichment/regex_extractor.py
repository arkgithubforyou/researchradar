"""Regex-based entity extraction — fast fallback when LLM is unavailable.

Extracts methods, datasets, tasks, and topics from paper abstracts/text
using curated pattern lists. Not as accurate as LLM extraction, but
zero-cost and fully deterministic.
"""

import logging
import re

logger = logging.getLogger(__name__)


# ── Method patterns ──────────────────────────────────────────────────
# Models, architectures, and techniques commonly found in NLP/ML papers.
# Each pattern is compiled as case-insensitive word-boundary match.

METHOD_NAMES: list[str] = [
    # Transformers & variants
    "BERT", "RoBERTa", "ALBERT", "DistilBERT", "DeBERTa", "ELECTRA",
    "ModernBERT", "ColBERT",
    "GPT", "GPT-2", "GPT-3", "GPT-3.5", "GPT-4", "GPT-4o", "o1", "ChatGPT",
    "Claude", "Gemini", "Gemini Pro", "Gemini Ultra",
    "T5", "FLAN-T5", "mT5", "BART", "mBART", "PEGASUS", "ProphetNet",
    "XLNet", "XLM", "XLM-R", "XLM-RoBERTa",
    "LLaMA", "Llama", "Llama 2", "Llama 3", "Llama 3.1",
    "Mistral", "Mixtral", "Falcon", "Gemma", "Gemma 2", "Phi", "Phi-3",
    "Qwen", "Qwen2", "Qwen2.5", "Yi", "DeepSeek", "DeepSeek-V3", "DeepSeek-R1",
    "Mamba", "RWKV", "Hyena",
    "Transformer", "Vision Transformer", "ViT", "Swin Transformer",
    "CLIP", "BLIP", "BLIP-2", "LLaVA", "Flamingo", "CogVLM",
    "Whisper", "Stable Diffusion",
    # Older / classic
    "LSTM", "BiLSTM", "GRU", "CNN", "TextCNN",
    "Word2Vec", "GloVe", "FastText", "ELMo",
    "seq2seq", "Seq2Seq",
    # Techniques
    "LoRA", "QLoRA", "adapter tuning", "prefix tuning", "prompt tuning",
    "knowledge distillation", "self-attention", "cross-attention",
    "multi-head attention", "sparse attention", "flash attention",
    "grouped-query attention", "GQA", "rotary position embedding", "RoPE",
    "mixture of experts", "MoE",
    "beam search", "nucleus sampling", "top-k sampling",
    "speculative decoding",
    "contrastive learning", "reinforcement learning from human feedback",
    "RLHF", "DPO", "PPO", "GRPO", "ORPO", "KTO",
    "chain-of-thought", "CoT", "tree-of-thought", "ReAct",
    "retrieval-augmented generation", "RAG",
    "in-context learning", "ICL", "few-shot", "zero-shot",
    "fine-tuning", "pre-training", "masked language modeling", "MLM",
    "causal language modeling", "CLM",
    "GPTQ", "AWQ", "GGUF",
    "DPR", "dense passage retrieval",
    "BPE", "SentencePiece", "WordPiece",
    "layer normalization", "RMSNorm",
    "BM25", "TF-IDF",
    # Frameworks
    "spaCy", "NLTK", "Hugging Face", "HuggingFace",
    "vLLM", "SGLang",
    "PyTorch", "TensorFlow", "JAX",
]

# ── Dataset patterns ─────────────────────────────────────────────────

DATASET_NAMES: list[str] = [
    # Question answering
    "SQuAD", "SQuAD 2.0", "Natural Questions", "TriviaQA", "HotpotQA",
    "QuAC", "CoQA", "DROP", "BoolQ", "RACE",
    "CommonsenseQA", "CSQA",
    # Sentiment / classification
    "SST", "SST-2", "SST-5", "IMDb", "Yelp", "Amazon Reviews",
    "AG News", "DBpedia", "20 Newsgroups",
    # NLI / entailment
    "MNLI", "SNLI", "MultiNLI", "ANLI", "RTE", "WNLI", "QNLI", "XNLI",
    # Summarization
    "CNN/DailyMail", "CNN/Daily Mail", "XSum", "Gigaword",
    "Multi-News", "SAMSum", "DialogSum",
    # Translation
    "WMT", "WMT14", "WMT16", "WMT19", "IWSLT",
    "Europarl", "ParaCrawl", "FLORES-200", "FLORES",
    # Named entity recognition
    "CoNLL-2003", "CoNLL-03", "OntoNotes", "ACE 2005", "WNUT",
    # Similarity / paraphrase
    "STS-B", "MRPC", "QQP", "PAWS",
    # Benchmarks — classic
    "GLUE", "SuperGLUE", "MMLU", "HellaSwag", "ARC",
    "WinoGrande", "Winograd", "LAMBADA", "PIQA",
    "TruthfulQA", "GSM8K", "MATH", "HumanEval", "MBPP",
    "MTEB", "BEIR",
    # Benchmarks — recent (2023-2025)
    "MMLU-Pro", "GPQA", "IFEval", "MT-Bench", "BigBench", "BIG-Bench",
    "SWE-bench", "MS MARCO", "Arena-Hard", "AlpacaEval",
    "LiveBench", "MMMU", "AGIEval", "WildBench",
    "Spider", "DocVQA", "ChartQA", "InfiBench",
    # Safety / alignment
    "HH-RLHF", "ToxiGen", "RealToxicityPrompts",
    # Retrieval
    "MS MARCO", "DL-HARD",
    # Multilingual
    "XTREME", "XTREME-R", "MEGA",
    # Dialogue
    "PersonaChat", "DailyDialog", "MultiWOZ", "Wizard of Wikipedia",
    # Other
    "WikiText", "WikiText-2", "WikiText-103", "C4", "The Pile",
    "Common Crawl", "OpenWebText", "RedPajama", "FineWeb",
    "LAMA", "FEVER", "MultiRC",
    "Penn Treebank", "PTB", "Universal Dependencies",
]

# ── Task patterns ────────────────────────────────────────────────────

TASK_NAMES: list[str] = [
    "machine translation", "neural machine translation",
    "text classification", "sentiment analysis", "sentiment classification",
    "aspect-based sentiment analysis",
    "named entity recognition", "NER",
    "entity linking",
    "question answering", "open-domain question answering",
    "reading comprehension", "long-form question answering",
    "table question answering",
    "text summarization", "abstractive summarization", "extractive summarization",
    "text generation", "language generation", "natural language generation",
    "language modeling", "masked language modeling", "causal language modeling",
    "relation extraction", "information extraction",
    "semantic similarity", "textual entailment",
    "natural language inference", "NLI",
    "part-of-speech tagging", "POS tagging",
    "dependency parsing", "constituency parsing", "syntactic parsing",
    "coreference resolution",
    "word sense disambiguation",
    "dialogue generation", "dialogue systems", "conversational AI",
    "dialogue state tracking",
    "intent detection", "slot filling",
    "machine reading comprehension",
    "paraphrase detection", "paraphrase generation",
    "text-to-SQL", "semantic parsing",
    "code generation", "code completion",
    "image captioning", "visual question answering", "VQA",
    "speech recognition", "automatic speech recognition", "ASR",
    "document classification", "document retrieval",
    "information retrieval", "dense retrieval", "passage reranking",
    "knowledge graph completion",
    "fact verification", "fact checking",
    "stance detection", "hate speech detection",
    "toxicity detection", "emotion detection",
    "topic modeling",
    "text simplification",
    "grammatical error correction",
    "mathematical reasoning", "multi-hop reasoning",
    "instruction following",
    "cross-lingual transfer",
]


# ── Topic patterns ───────────────────────────────────────────────────
# High-level research themes / areas — broader than tasks or methods.
# Automatically deduplicated against METHOD_NAMES, DATASET_NAMES, and
# TASK_NAMES at module load time so there is zero overlap.

_TOPIC_NAMES_RAW: list[str] = [
    # Paradigms & approaches
    "multimodal", "multi-modal", "cross-modal", "cross-lingual", "multilingual",
    "agentic", "AI agent", "LLM agent", "autonomous agent",
    "agentic RAG", "GraphRAG",
    "prompting", "prompt engineering", "instruction tuning", "instruction following",
    "alignment", "AI alignment", "value alignment", "safety alignment",
    "preference optimization",
    "explainability", "interpretability", "model interpretability",
    "mechanistic interpretability",
    "fairness", "bias", "debiasing", "social bias",
    "robustness", "adversarial robustness", "adversarial examples",
    "jailbreaking", "red teaming",
    "hallucination", "factual consistency", "faithfulness",
    "efficiency", "model compression", "pruning", "quantization",
    "scaling", "scaling law", "emergent abilities",
    "data augmentation", "synthetic data", "data contamination",
    "self-supervised learning", "semi-supervised learning",
    "active learning", "curriculum learning", "meta-learning",
    "transfer learning", "domain adaptation", "cross-domain",
    "continual learning", "lifelong learning", "catastrophic forgetting",
    "federated learning", "privacy-preserving", "differential privacy",
    "low-resource", "few-shot learning", "zero-shot learning",
    "model merging", "model editing",
    "state space model", "SSM",
    "vision-language model", "VLM",
    "LLM evaluation", "LLM-as-a-judge",
    "AI safety", "human-AI interaction",
    "reasoning", "chain-of-thought reasoning",
    # Application areas
    "biomedical NLP", "clinical NLP", "medical NLP",
    "legal NLP", "financial NLP",
    "NLP for education", "educational technology", "tutoring",
    "social media", "misinformation", "fake news",
    "commonsense reasoning", "commonsense knowledge",
    "script knowledge", "script learning",
    "grounding", "embodied AI", "situated language",
    "tool use", "tool learning", "tool augmented",
    "multimodal reasoning", "visual reasoning", "spatial reasoning",
    "temporal reasoning", "numerical reasoning", "mathematical reasoning",
    "causal reasoning", "causal inference",
    "knowledge graph", "knowledge base", "knowledge representation",
    "open-domain", "closed-domain",
    "long context", "long document", "context window",
    "code intelligence", "software engineering",
    "multilingual NLP", "machine translation",
    "speech and language", "spoken language understanding",
    "document understanding", "table understanding",
    "graph neural network", "structured prediction",
    "neuro-symbolic", "hybrid AI",
    "evaluation", "benchmark design", "human evaluation",
    "annotation", "crowdsourcing", "data collection",
    "reproducibility", "open science",
    "ethics", "AI ethics", "responsible AI",
    "toxicity", "content moderation", "harmful content",
    "watermarking", "AI-generated text detection",
]

# Build a set of lowercased names from the other three lists for deduplication.
_OTHER_NAMES_LOWER: set[str] = {
    n.lower() for n in METHOD_NAMES + DATASET_NAMES + TASK_NAMES
}

TOPIC_NAMES: list[str] = [
    t for t in _TOPIC_NAMES_RAW if t.lower() not in _OTHER_NAMES_LOWER
]


def _build_pattern(name: str) -> re.Pattern:
    """Build a case-insensitive word-boundary regex for a name.

    Handles special characters in names (e.g., "GPT-2", "CNN/DailyMail").
    """
    escaped = re.escape(name)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


# Pre-compile all patterns at module load time.
_METHOD_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, _build_pattern(name)) for name in METHOD_NAMES
]
_DATASET_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, _build_pattern(name)) for name in DATASET_NAMES
]
_TASK_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, _build_pattern(name)) for name in TASK_NAMES
]
_TOPIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, _build_pattern(name)) for name in TOPIC_NAMES
]


def extract_methods_regex(text: str) -> list[dict]:
    """Extract method/model mentions from text using regex patterns.

    Returns:
        List of {"method_name": str, "method_type": str | None}
    """
    found: list[dict] = []
    seen: set[str] = set()
    for canonical_name, pattern in _METHOD_PATTERNS:
        if pattern.search(text):
            key = canonical_name.lower()
            if key not in seen:
                seen.add(key)
                found.append({"method_name": canonical_name, "method_type": None})
    return found


def extract_datasets_regex(text: str) -> list[dict]:
    """Extract dataset mentions from text using regex patterns.

    Returns:
        List of {"dataset_name": str, "task_type": str | None}
    """
    found: list[dict] = []
    seen: set[str] = set()
    for canonical_name, pattern in _DATASET_PATTERNS:
        if pattern.search(text):
            key = canonical_name.lower()
            if key not in seen:
                seen.add(key)
                found.append({"dataset_name": canonical_name, "task_type": None})
    return found


def extract_tasks_regex(text: str) -> list[str]:
    """Extract task mentions from text using regex patterns.

    Returns:
        List of task name strings (deduplicated).
    """
    found: list[str] = []
    seen: set[str] = set()
    for canonical_name, pattern in _TASK_PATTERNS:
        if pattern.search(text):
            key = canonical_name.lower()
            if key not in seen:
                seen.add(key)
                found.append(canonical_name)
    return found


def extract_topics_regex(text: str) -> list[str]:
    """Extract research topic/theme mentions from text using regex patterns.

    Returns:
        List of topic name strings (deduplicated).
    """
    found: list[str] = []
    seen: set[str] = set()
    for canonical_name, pattern in _TOPIC_PATTERNS:
        if pattern.search(text):
            key = canonical_name.lower()
            if key not in seen:
                seen.add(key)
                found.append(canonical_name)
    return found


def extract_all_regex(text: str) -> dict:
    """Run all regex extractors on a text.

    Returns:
        {
            "methods": [{"method_name": str, "method_type": str | None}, ...],
            "datasets": [{"dataset_name": str, "task_type": str | None}, ...],
            "tasks": [str, ...],
            "topics": [str, ...],
        }
    """
    return {
        "methods": extract_methods_regex(text),
        "datasets": extract_datasets_regex(text),
        "tasks": extract_tasks_regex(text),
        "topics": extract_topics_regex(text),
    }
