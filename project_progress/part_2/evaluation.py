"""Part 2: Information Retrieval evaluation metrics.

Conventions (also reused by Part 3):

* ``labels`` is the 0/1 relevance of a ranked list, best first.
* ``total_relevant`` (R) is the number of judged relevant products for the
  query in the pool being used, NOT the number of relevant products in the
  whole corpus (which is unknown).
* AP@K is truncated: sum over i <= K of P@i * rel_i, divided by min(R, K),
  so a list that puts min(R, K) relevant products first scores 1.
"""

import math
from numbers import Integral


def _check_k(k):
    """Reject cut-offs that would silently give meaningless values."""
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError(f"K must be a positive integer, got {k!r}")


def precision_at_k(labels, k):
    """Proportion of relevant documents in the top K."""
    _check_k(k)
    return sum(labels[:k]) / k


def recall_at_k(labels, k, total_relevant):
    """Proportion of all relevant documents retrieved in the top K."""
    _check_k(k)
    if total_relevant == 0:
        return 0.0

    return sum(labels[:k]) / total_relevant


def f1_at_k(labels, k, total_relevant):
    """Harmonic mean of Precision@K and Recall@K."""
    precision = precision_at_k(labels, k)
    recall = recall_at_k(labels, k, total_relevant)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def average_precision_at_k(labels, k, total_relevant):
    """Truncated AP@K = sum_{i<=K} P@i * rel_i / min(R, K)."""
    _check_k(k)
    if total_relevant == 0:
        return 0.0

    relevant_found = 0
    precision_sum = 0.0

    for rank, label in enumerate(labels[:k], start=1):
        if label == 1:
            relevant_found += 1
            precision_sum += relevant_found / rank

    return precision_sum / min(total_relevant, k)


def mean_average_precision(all_labels, total_relevants, k):
    """Mean Average Precision across multiple queries."""
    if not all_labels:
        return 0.0

    if len(all_labels) != len(total_relevants):
        raise ValueError("One total_relevant value is needed per query")

    scores = [
        average_precision_at_k(labels, k, total)
        for labels, total in zip(all_labels, total_relevants, strict=True)
    ]

    return sum(scores) / len(scores)


def reciprocal_rank(labels):
    """Reciprocal position of the first relevant document."""
    for rank, label in enumerate(labels, start=1):
        if label == 1:
            return 1 / rank

    return 0.0


def mean_reciprocal_rank(all_labels):
    """Mean Reciprocal Rank across multiple queries."""
    if not all_labels:
        return 0.0

    return sum(reciprocal_rank(labels) for labels in all_labels) / len(all_labels)


def ndcg_at_k(labels, k, total_relevant):
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""
    _check_k(k)
    dcg = sum(
        label / math.log2(rank + 1)
        for rank, label in enumerate(labels[:k], start=1)
    )

    ideal_count = min(total_relevant, k)

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


if __name__ == "__main__":

    # Example relevance judgments for two queries
    query_1 = [1, 0, 1, 0]
    query_2 = [0, 1, 0]

    # Total number of relevant documents for each query
    total_1 = 3
    total_2 = 1

    k = 3

    print("Evaluation metrics - Query 1")
    print("Precision@3:", round(precision_at_k(query_1, k), 3))
    print("Recall@3:", round(recall_at_k(query_1, k, total_1), 3))
    print("F1@3:", round(f1_at_k(query_1, k, total_1), 3))
    print("AP@3:", round(average_precision_at_k(query_1, k, total_1), 3))
    print("NDCG@3:", round(ndcg_at_k(query_1, k, total_1), 3))

    all_labels = [query_1, query_2]
    total_relevants = [total_1, total_2]

    print("\nOverall evaluation")
    print("MAP@3:", round(mean_average_precision(all_labels, total_relevants, k), 3))
    print("MRR:", round(mean_reciprocal_rank(all_labels), 3))