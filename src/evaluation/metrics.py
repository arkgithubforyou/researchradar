"""Retrieval and generation evaluation metrics.

Computes standard IR and NLG metrics for evaluating the RAG pipeline:
- Retrieval: Precision@K, Recall@K, MRR, NDCG@K
- Generation: ROUGE-L (F1), answer coverage (keyword overlap)
"""

import math
import re
from collections import Counter


# ── Retrieval metrics ────────────────────────────────────────────────


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean reciprocal rank — 1/rank of the first relevant result."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0

    # Ideal DCG (all relevant docs ranked first)
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


# ── Generation metrics ───────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def rouge_l_f1(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence.

    A lightweight, dependency-free implementation.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS length via dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def answer_coverage(prediction: str, reference_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the prediction.

    Case-insensitive word-boundary matching.
    """
    if not reference_keywords:
        return 0.0

    pred_lower = prediction.lower()
    hits = sum(1 for kw in reference_keywords if kw.lower() in pred_lower)
    return hits / len(reference_keywords)


# ── Aggregation ──────────────────────────────────────────────────────


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int] | None = None,
) -> dict:
    """Compute a full suite of retrieval metrics."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {"mrr": mrr(retrieved_ids, relevant_ids)}
    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    return results


def evaluate_generation(
    prediction: str,
    reference: str,
    keywords: list[str] | None = None,
) -> dict:
    """Compute generation quality metrics."""
    results = {"rouge_l_f1": rouge_l_f1(prediction, reference)}
    if keywords:
        results["answer_coverage"] = answer_coverage(prediction, keywords)
    return results
