
"""Tests for IRWA Part 2: Indexing, Ranking and Evaluation."""

import math

from project_progress.part_2.indexing import (
    build_inverted_index,
    search_and,
)

from project_progress.part_2.ranking import rank_tfidf

from project_progress.part_2.evaluation import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision_at_k,
    mean_average_precision,
    reciprocal_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
)


def toy_corpus():
    """Create a small document collection for testing."""
    documents = {
        "a": ["blue", "jean", "blue"],
        "b": ["blue", "shirt"],
        "c": ["jean", "slim"],
        "d": ["cotton", "shirt"],
    }

    index = build_inverted_index(documents)
    return documents, index


# TEST 1: Inverted index
def test_inverted_index():
    documents, index = toy_corpus()

    assert index["blue"] == ["a", "b"]
    assert index["jean"] == ["a", "c"]


# TEST 2: Boolean AND search
def test_and_search():
    documents, index = toy_corpus()

    assert search_and("blue jeans", index) == ["a"]
    assert search_and("blue cotton", index) == []


# TEST 3: Empty queries
def test_empty_query():
    documents, index = toy_corpus()

    assert search_and("the and", index) == []


# TEST 4: TF-IDF ranking
def test_tfidf_ranking():
    documents, index = toy_corpus()

    results = rank_tfidf(
        "blue jeans",
        documents,
        index,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0][0] == "a"
    assert 0 <= results[0][1] <= 1


# TEST 5: Precision, Recall, F1 and AP
def test_basic_metrics():
    labels = [1, 0, 1, 0]

    assert math.isclose(precision_at_k(labels, 3), 2/3)
    assert math.isclose(recall_at_k(labels, 3, 3), 2/3)
    assert math.isclose(f1_at_k(labels, 3, 3), 2/3)

    expected_ap = (1 + 2/3) / 3

    assert math.isclose(
        average_precision_at_k(labels, 3, 3),
        expected_ap,
    )


# TEST 6: MAP, MRR and NDCG
def test_ranking_metrics():

    assert math.isclose(
        reciprocal_rank([0, 0, 1]), 1/3
    )

    assert math.isclose(
        mean_reciprocal_rank([[1], [0, 1]]),
        0.75,
    )

    assert math.isclose(
        mean_average_precision(
            [[1, 0], [0, 1]],
            [1, 1],
            2,
        ),
        0.75,
    )

    assert 0 <= ndcg_at_k([1, 0, 1], 3, 2) <= 1


# TEST 7: No relevant documents
def test_no_relevant_documents():

    labels = [0, 0]

    assert recall_at_k(labels, 2, 0) == 0
    assert average_precision_at_k(labels, 2, 0) == 0
    assert ndcg_at_k(labels, 2, 0) == 0
    assert reciprocal_rank(labels) == 0
