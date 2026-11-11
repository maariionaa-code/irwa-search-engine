"""Tests for IRWA Part 3 on a small toy corpus (no dataset or gensim needed).

Run from the repository root:
    python -m pytest project_progress/part_3/test_part_3.py -v
"""

import math

import numpy as np
import pandas as pd
import pytest

from project_progress.part_2.ranking import rank_tfidf as part2_rank_tfidf
from project_progress.part_3.compare_rankings import (
    jaccard_at_k,
    kendall_tau_full,
    overlap_at_k,
)
from project_progress.part_3.corpus import SearchIndex
from project_progress.part_3.embedded_ranking import (
    cosine,
    rank_word2vec,
    text_vector,
)
from project_progress.part_3.ranking_methods import (
    bm25_idf,
    bm25_term_score,
    combine_our_score,
    our_score_components,
    rank_bm25,
    rank_our_score,
    rank_tfidf,
    rating_feature,
)


@pytest.fixture
def toy():
    """Four documents; "blue jeans" matches a, e and f."""
    documents = {
        "a": ["blue", "jean", "blue"],
        "b": ["blue", "shirt"],
        "c": ["jean", "slim"],
        "d": ["cotton", "shirt"],
        "e": ["blue", "jean", "slim", "fit", "cotton", "denim", "wash", "men"],
        "f": ["jean", "blue"],
    }
    products = pd.DataFrame({
        "pid": list(documents),
        "title_tokens": [["blue", "jean"], ["shirt"], ["jean"], ["shirt"],
                         ["slim", "jean"], ["blue", "jean"]],
        "rating": [4.0, 3.0, None, 5.0, 4.5, 4.0],
        "discount_pct": [50.0, 10.0, None, 0.0, 20.0, 50.0],
        "out_of_stock": [False, False, False, False, False, True],
    }).set_index("pid", drop=False)
    return SearchIndex(documents), products


# --- AND semantics shared by every method --------------------------------

def test_all_methods_rank_the_same_and_candidates(toy):
    index, products = toy
    expected = {"a", "e", "f"}
    assert set(index.candidates("blue jeans")) == expected
    for ranking in (rank_tfidf("blue jeans", index),
                    rank_bm25("blue jeans", index),
                    rank_our_score("blue jeans", index, products)):
        assert {pid for pid, _ in ranking} == expected


def test_unknown_or_stopword_query_returns_nothing(toy):
    index, products = toy
    assert rank_bm25("zzzunknown", index) == []
    assert rank_our_score("the and", index, products) == []


# --- TF-IDF ----------------------------------------------------------------

def test_tfidf_matches_part2(toy):
    index, _ = toy
    part2 = part2_rank_tfidf("blue jeans", index.documents, index.inverted_index)
    part3 = rank_tfidf("blue jeans", index)
    assert [p for p, _ in part2] == [p for p, _ in part3]
    for (_, s2), (_, s3) in zip(part2, part3):
        assert math.isclose(s2, s3)


# --- BM25 ------------------------------------------------------------------

def test_bm25_idf_hand_value(toy):
    index, _ = toy
    # N = 6, df(blue) = 4  ->  ln(1 + (6 - 4 + 0.5) / (4 + 0.5))
    assert math.isclose(bm25_idf(index, "blue"), math.log(1 + 2.5 / 4.5))


def test_bm25_score_hand_value(toy):
    index, _ = toy
    # avgdl = (3+2+2+2+8+2)/6 = 19/6. Document a: length 3, blue=2, jean=1.
    avgdl = 19 / 6
    norm = 1 - 0.75 + 0.75 * 3 / avgdl
    idf_blue = math.log(1 + (6 - 4 + 0.5) / 4.5)
    idf_jean = math.log(1 + (6 - 4 + 0.5) / 4.5)
    expected = (idf_blue * 2 * 2.2 / (2 + 1.2 * norm)
                + idf_jean * 1 * 2.2 / (1 + 1.2 * norm))
    scores = dict(rank_bm25("blue jeans", index))
    assert math.isclose(scores["a"], expected)


def test_bm25_saturates_term_frequency():
    # Doubling tf must less than double the contribution (k1 saturation).
    one = bm25_term_score(1, 10, 10, idf=1.0)
    two = bm25_term_score(2, 10, 10, idf=1.0)
    hundred = bm25_term_score(100, 10, 10, idf=1.0)
    assert one < two < 2 * one
    assert hundred < 1.0 + 1.2  # upper bound idf * (k1 + 1)


def test_bm25_penalises_long_documents():
    short = bm25_term_score(1, 5, 10, idf=1.0)
    long = bm25_term_score(1, 20, 10, idf=1.0)
    assert short > long
    # With b = 0 there is no length normalisation.
    assert math.isclose(bm25_term_score(1, 5, 10, 1.0, b=0),
                        bm25_term_score(1, 20, 10, 1.0, b=0))


def test_bm25_prefers_short_exact_document(toy):
    index, _ = toy
    ranking = [pid for pid, _ in rank_bm25("blue jeans", index)]
    # e is long (8 tokens) with the words once: it must be ranked last.
    assert ranking[-1] == "e"


# --- Our score -------------------------------------------------------------

def test_rating_feature_scaling_and_missing():
    assert rating_feature(5.0, 0.5) == 1.0
    assert rating_feature(1.0, 0.5) == 0.0
    assert rating_feature(None, 0.42) == 0.42
    assert rating_feature(float("nan"), 0.42) == 0.42


def test_our_score_in_unit_interval_and_out_of_stock_demoted(toy):
    index, products = toy
    components = our_score_components("blue jeans", index, products)
    assert components["text"].max() == 1.0
    scores = combine_our_score(components)
    assert ((scores >= 0) & (scores <= 1)).all()
    # f has the same words and title as a but is out of stock.
    in_stock = combine_our_score(components, out_of_stock_factor=1.0)
    assert math.isclose(scores["f"], 0.5 * in_stock["f"])


def test_our_score_title_coverage(toy):
    index, products = toy
    components = our_score_components("blue jeans", index, products)
    assert components.loc["a", "title"] == 1.0   # blue + jean in title
    assert components.loc["e", "title"] == 0.5   # only jean in title


# --- word2vec ----------------------------------------------------------------

def test_text_vector_is_mean_of_known_word_vectors():
    vectors = {"blue": np.array([1.0, 0.0]), "jean": np.array([0.0, 1.0])}
    assert np.allclose(text_vector(["blue", "jean", "unknown"], vectors),
                       [0.5, 0.5])
    assert np.allclose(text_vector(["blue", "blue", "jean"], vectors),
                       [2 / 3, 1 / 3])
    assert text_vector(["unknown"], vectors) is None


def test_cosine_values():
    assert math.isclose(cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])), 1.0)
    assert math.isclose(cosine(np.array([1.0, 0.0]), np.array([0.0, 3.0])), 0.0)
    assert cosine(np.zeros(2), np.array([1.0, 1.0])) == 0.0


def test_word2vec_ranking_uses_and_and_cosine(toy):
    index, _ = toy
    vectors = {
        "blue": np.array([1.0, 0.0, 0.0]),
        "jean": np.array([1.0, 0.0, 0.0]),
        "slim": np.array([0.0, 1.0, 0.0]),
        "fit": np.array([0.0, 1.0, 0.0]),
        "cotton": np.array([0.0, 0.0, 1.0]),
    }
    ranking = rank_word2vec("blue jeans", index, vectors)
    assert {pid for pid, _ in ranking} == {"a", "e", "f"}
    scores = dict(ranking)
    assert math.isclose(scores["a"], 1.0)  # every word points the same way
    assert scores["e"] < scores["a"]       # extra unrelated words


# --- Ranking comparison --------------------------------------------------------

def test_agreement_measures():
    a = ["p1", "p2", "p3", "p4"]
    assert overlap_at_k(a, ["p2", "p1", "p9", "p8"], 2) == 1.0
    assert math.isclose(jaccard_at_k(a, ["p1", "p5"], 2), 1 / 3)
    assert math.isclose(kendall_tau_full(a, a), 1.0)
    assert math.isclose(kendall_tau_full(a, list(reversed(a))), -1.0)