"""Tests for the Part 3 extra analyses (no dataset needed).

Run from the repository root:
    python -m pytest project_progress/part_3/test_extras.py -v
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from project_progress.part_2.evaluation import (
    average_precision_at_k,
    mean_average_precision,
    precision_at_k,
)
from project_progress.part_3.ablation import evaluate_ranking
from project_progress.part_3.compare_rankings import score_spread
from project_progress.part_3.corpus import check_product_ids
from project_progress.part_3.embedded_ranking import fingerprint
from project_progress.part_3.evaluate_methods import (
    check_rankings,
    load_labels,
    significance_table,
)
from project_progress.part_3.export_judgment_pool import (
    build_pool,
    label_conflicts,
    read_pool,
)
from project_progress.part_3.label_agreement import cohen_kappa, draw_sample
from project_progress.part_3.queries import ALL_QUERIES
from project_progress.part_3.run_rankings import check_same_candidates
from project_progress.part_3.rank_statistics import (
    holm_adjust,
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
    # Condensed list drops "x": a, c, b -> 2 relevant out of 10 positions,
    # AP = (1/1 + 2/2) / min(2, 10) = 1.
    assert result["P@10_condensed"] == 0.2
    assert result["AP@10_condensed"] == 1.0


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


def test_holm_adjustment_hand_values():
    # m = 3: sorted p = 0.01, 0.03, 0.04 -> 3*0.01, max(0.03, 2*0.03), max(0.06, 1*0.04)
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    # 30 tests: the smallest possible 8-query p-value (2/256) is no longer significant
    adjusted = holm_adjust([2 / 256] + [1.0] * 29)
    assert adjusted[0] == pytest.approx(30 * 2 / 256) and adjusted[0] > 0.05
    assert max(holm_adjust([0.5, 0.9])) <= 1.0


def test_significance_table_reports_holm_adjusted_p_values():
    rows = []
    for q in range(8):
        for method, ap in (("bm25", 0.5), ("tfidf", 0.4), ("our_score", 0.9)):
            rows.append({"method": method, "query_id": f"Q{q}", "k": 10,
                         "AP@K_pool": ap, "NDCG@K_pool": ap, "P@K": ap})
    table = significance_table(pd.DataFrame(rows))
    assert (table["p_holm"] >= table["p_value"]).all()
    assert table["p_holm"].tolist() == pytest.approx(holm_adjust(table["p_value"]))


# --- reproducible embeddings ------------------------------------------------------

EMBEDDING_SCRIPT = """
from project_progress.part_3.embedded_ranking import (
    infer_query_vector, train_doc2vec, train_word2vec)
docs = {str(i): ["blue", "jean", "slim", "fit", "cotton", "shirt"][i % 3:] * 3
        for i in range(30)}
w2v = train_word2vec(docs)
d2v = train_doc2vec(docs)
print(w2v.wv["jean"][:3].tolist(), d2v.dv["0"][:3].tolist(),
      infer_query_vector(d2v, ["slim", "jean"])[:3].tolist())
"""


def test_embeddings_do_not_depend_on_python_hash_seed():
    """Same vectors even when Python's string hash differs between runs."""
    pytest.importorskip("gensim")
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run([sys.executable, "-c", EMBEDDING_SCRIPT],
                                capture_output=True, text=True, env=env,
                                cwd=Path(__file__).resolve().parents[2],
                                check=True)
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_query_inference_equals_gensim_with_a_stable_hash():
    """infer_query_vector runs gensim's own algorithm, only with a stable seed."""
    pytest.importorskip("gensim")
    from functools import partial

    from gensim.models import doc2vec

    from project_progress.part_3.embedded_ranking import (
        infer_query_vector, stable_hash, train_doc2vec)

    docs = {str(i): ["blue", "jean", "slim", "fit", "cotton", "shirt"][i % 3:] * 3
            for i in range(30)}
    model = train_doc2vec(docs)
    ours = infer_query_vector(model, ["slim", "jean"], epochs=20)
    original = doc2vec.pseudorandom_weak_vector
    doc2vec.pseudorandom_weak_vector = partial(original, hashfxn=stable_hash)
    try:
        model.random.seed(42)
        reference = model.infer_vector(["slim", "jean"], epochs=20)
    finally:
        doc2vec.pseudorandom_weak_vector = original
    assert ours == pytest.approx(reference)
    assert doc2vec.pseudorandom_weak_vector is original  # no global state left


# --- metric conventions ----------------------------------------------------------

def test_ap_at_k_is_truncated_by_min_r_k():
    # R = 30 judged relevant but K = 5: a perfect top 5 must score 1.
    assert average_precision_at_k([1] * 5, 5, 30) == 1.0
    # [1, 0, 1], R = 3, K = 3 -> (1/1 + 2/3) / min(3, 3)
    assert math.isclose(average_precision_at_k([1, 0, 1], 3, 3), (1 + 2 / 3) / 3)


def test_metrics_reject_bad_k_and_mismatched_lists():
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            precision_at_k([1, 0], bad)
    with pytest.raises(ValueError):
        mean_average_precision([[1, 0], [0, 1]], [1], 2)


# --- judgment pool ------------------------------------------------------------------

def _products(pids):
    return pd.DataFrame({
        "pid": pids, "title": pids, "brand": "b", "category": "c",
        "sub_category": "s", "description": "d", "product_details": "[]",
        "url": "u",
    }).set_index("pid")


def test_pool_keeps_judged_product_that_left_the_rankings():
    existing = pd.DataFrame([
        {"query_id": "Q1", "query": "q", "pid": "old", "relevance": 1,
         "label_source": "part3", "found_by": "bm25", "reason": "exact match"},
        {"query_id": "Q1", "query": "q", "pid": "kept", "relevance": 0,
         "label_source": "part2", "found_by": "bm25", "reason": "wrong gender"},
    ])
    rankings = pd.DataFrame([
        {"query_id": "Q1", "query": "q", "method": "tfidf", "pid": "kept"},
        {"query_id": "Q1", "query": "q", "method": "bm25", "pid": "new"},
    ])
    pool = build_pool(rankings, _products(["old", "kept", "new"]), existing,
                      part2={("Q1", "new"): 1}).set_index("pid")

    assert set(pool.index) == {"old", "kept", "new"}
    assert pool.loc["old", "relevance"] == 1                  # label survives
    assert pool.loc["old", "reason"] == "exact match"        # extra column kept
    assert not pool.loc["old", "in_current_top20"]
    assert pool.loc["old", "found_by"] == ""
    assert pool.loc["kept", "found_by"] == "tfidf"           # provenance refreshed
    assert pool.loc["kept", "label_source"] == "part2"       # label provenance kept
    assert pool.loc["new", "relevance"] == 1                 # reused Part 2 label
    assert pool.loc["new", "label_source"] == "part2"


def test_pool_reports_conflicts_and_rejects_duplicates(tmp_path):
    pool = pd.DataFrame([
        {"query_id": "Q1", "pid": "a", "relevance": 1, "label_source": "part2"},
        {"query_id": "Q1", "pid": "b", "relevance": 1, "label_source": "part2_corrected"},
    ])
    conflicts = label_conflicts(pool, {("Q1", "a"): 0, ("Q1", "b"): 0})
    assert conflicts["pid"].tolist() == ["a"]      # "b" is an acknowledged fix

    path = tmp_path / "pool.csv"
    pd.concat([pool, pool.head(1)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_pool(path)


def test_load_labels_validates_the_pool(tmp_path):
    rows = [{"query_id": q, "pid": "p", "relevance": 1} for q in ALL_QUERIES]
    path = tmp_path / "pool.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert len(load_labels(path)) == len(ALL_QUERIES)

    pd.DataFrame(rows + rows[:1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Repeated"):
        load_labels(path)
    pd.DataFrame(rows[1:]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="No judgments"):
        load_labels(path)
    pd.DataFrame([{**rows[0], "relevance": 2}] + rows[1:]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="0 or 1"):
        load_labels(path)


def test_check_rankings_rejects_gaps_and_repeats():
    good = pd.DataFrame([{"query_id": q, "method": "bm25", "rank": r, "pid": f"p{r}"}
                         for q in ALL_QUERIES for r in (1, 2)])
    check_rankings(good)
    gap = good.copy()
    gap.loc[1, "rank"] = 3
    with pytest.raises(ValueError, match="ranks"):
        check_rankings(gap)
    repeat = good.copy()
    repeat.loc[1, "pid"] = "p1"
    with pytest.raises(ValueError, match="twice"):
        check_rankings(repeat)


def test_candidate_set_check():
    check_same_candidates("Q1", "bm25", [("a", 1.0), ("b", 0.5)], {"a", "b"})
    with pytest.raises(ValueError):
        check_same_candidates("Q1", "bm25", [("a", 1.0)], {"a", "b"})
    with pytest.raises(ValueError):
        check_same_candidates("Q1", "bm25", [("a", 1.0), ("a", 0.5)], {"a"})


def test_product_ids_must_be_unique_and_present():
    check_product_ids(pd.Series(["a", "b"]))
    for bad in (["a", "a"], ["a", None], ["a", " "]):
        with pytest.raises(ValueError):
            check_product_ids(pd.Series(bad))


def test_model_fingerprint_changes_with_data_or_parameters():
    docs = {"a": ["blue", "jean"], "b": ["red", "shirt"]}
    base = fingerprint(docs, {"vector_size": 100})
    # training order changes the model, so it changes the fingerprint too
    assert base != fingerprint(dict(reversed(docs.items())), {"vector_size": 100})
    assert base != fingerprint({**docs, "b": ["red"]}, {"vector_size": 100})
    assert base != fingerprint(docs, {"vector_size": 50})


# --- inter-annotator agreement -----------------------------------------------------

def test_cohen_kappa_hand_values():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # p_o = 0.5, p_e = 0.5 -> kappa = 0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
    # p_o = 0.8, pa = 0.6, pb = 0.4 -> p_e = 0.48, kappa = 0.32 / 0.52
    assert math.isclose(cohen_kappa([1, 1, 1, 0, 0], [1, 1, 0, 0, 0]), 0.32 / 0.52)


def test_agreement_sample_is_blind_stratified_and_reproducible():
    pool = pd.DataFrame({
        "query_id": ["Q1"] * 30 + ["Q2"] * 10, "query": "q",
        "pid": [f"p{i}" for i in range(40)], "title": "t", "sub_category": "s",
        "description": "d", "product_details": "[]", "relevance": 1,
        "found_by": "bm25", "url": "u",
    })
    sample = draw_sample(pool, size=8)
    assert sample.groupby("query_id").size().to_dict() == {"Q1": 6, "Q2": 2}
    assert sample["relevance"].isna().all()
    assert not {"found_by", "url"} & set(sample.columns)
    assert sample.equals(draw_sample(pool, size=8))
