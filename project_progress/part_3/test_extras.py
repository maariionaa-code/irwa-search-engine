"""Tests for the Part 3 extra analyses (no dataset needed).

Run from the repository root:
    python -m pytest project_progress/part_3/test_extras.py -v
"""

import math

import pandas as pd

from project_progress.part_3.ablation import evaluate_ranking
from project_progress.part_3.compare_rankings import score_spread
from project_progress.part_3.evaluate_methods import significance_table
from project_progress.part_3.rank_statistics import (
    paired_permutation_test,
    rank_biased_overlap,
)


# --- paired randomisation test ------------------------------------------------

def test_permutation_identical_methods_not_significant():
    assert paired_permutation_test([0.5, 0.6, 0.4], [0.5, 0.6, 0.4]) == (0.0, 1.0)


def test_permutation_exact_p_value_for_consistent_win():
    # A beats B by 1 on all 8 queries: only the all-+ and all-- sign
    # assignments are as extreme, so p = 2 / 2^8.
    diff, p = paired_permutation_test([1.0] * 8, [0.0] * 8)
    assert diff == 1.0
    assert math.isclose(p, 2 / 256)


def test_permutation_is_symmetric():
    a, b = [0.9, 0.2, 0.7, 0.4], [0.5, 0.3, 0.1, 0.4]
    d1, p1 = paired_permutation_test(a, b)
    d2, p2 = paired_permutation_test(b, a)
    assert math.isclose(d1, -d2) and math.isclose(p1, p2)


# --- rank-biased overlap ---------------------------------------------------------

def test_rbo_identical_disjoint_and_hand_value():
    a = list("abcdefghij")
    assert math.isclose(rank_biased_overlap(a, a, p=0.9, depth=10), 1 - 0.9 ** 10)
    assert rank_biased_overlap(a, list("klmnopqrst"), p=0.9, depth=10) == 0.0
    # Depth 3: overlaps 0/1, 2/2, 3/3 -> 0.1 * (0 + 0.9 + 0.81) = 0.171
    assert math.isclose(rank_biased_overlap(list("abc"), list("bac"), 0.9, 3), 0.171)


def test_rbo_weights_top_disagreement_more():
    base = list("abcdefghij")
    swap_top = list("bacdefghij")
    swap_bottom = list("abcdefghji")
    assert rank_biased_overlap(base, swap_bottom) > rank_biased_overlap(base, swap_top)


# --- score spread and ablation helper ------------------------------------------

def test_score_spread_counts_near_ties():
    scored = [(f"p{i}", s) for i, s in enumerate(
        [1.0] + [0.80] * 12 + [0.5] * 10)]
    row = score_spread("Q1", "tfidf", scored)
    assert row["score@1"] == 1.0 and row["score@10"] == 0.80
    assert row["near_ties_around_rank10"] == 12
    assert math.isclose(row["relative_drop_1_to_20"], 0.5)


def test_evaluate_ranking_treats_unjudged_as_non_relevant():
    labels = {"a": 1, "b": 0, "c": 1}
    result = evaluate_ranking(["a", "x", "c", "b"], labels)
    assert result["P@10"] == 0.2          # 2 relevant in the top 10
    assert result["judged@10"] == 3       # "x" was never judged


# --- significance table -----------------------------------------------------------

def test_significance_table_compares_with_baselines():
    rows = []
    for q in range(8):
        for method, ap in (("bm25", 0.5), ("tfidf", 0.4), ("our_score", 0.9)):
            rows.append({"method": method, "query_id": f"Q{q}", "k": 10,
                         "AP@K_pool": ap, "NDCG@K_pool": ap, "P@K": ap})
    table = significance_table(pd.DataFrame(rows))
    row = table[(table["baseline"] == "bm25") & (table["method"] == "our_score")
                & (table["metric"] == "AP@10_pool")].iloc[0]
    assert math.isclose(row["mean_difference"], 0.4)
    assert row["significant_5pct"]
