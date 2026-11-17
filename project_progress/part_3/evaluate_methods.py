"""Evaluate every ranking method with the pooled relevance judgments.

For each query, the set of known relevant products is the same for all
methods (the relevant products in the pool), so recall, AP and NDCG are
comparable across methods. The Part 2 metric functions are reused.

Metric conventions:
* R (``relevant_in_pool``) is the number of judged relevant products in
  the pool, not in the whole corpus, so R@K, AP@K and NDCG@K measure how
  much of the KNOWN relevant set is found. They are not directly comparable
  with Part 2, whose pool (one method's top 20) was smaller.
* AP@K = sum_{i<=K} P@i * rel_i / min(R, K) (truncated AP).
* RR@K = 1 / rank of the first relevant product within the top K, else 0.

It also writes ``query_group_summary.csv``: the means over the queries
that state a gender (Q1-Q5) and the ones that do not (Q6-Q8).

It also tests whether the differences between methods are statistically
significant (paired randomisation test over queries, see rank_statistics.py)
and writes them to results/significance.csv.

Usage from the repository root (after labelling judgments_pool.csv):
    python -m project_progress.part_3.evaluate_methods
"""

from pathlib import Path

import pandas as pd

from project_progress.part_2.evaluation import (
    average_precision_at_k,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from project_progress.part_3.export_judgment_pool import POOL_CSV
from project_progress.part_3.queries import (
    ALL_QUERIES,
    GENDER_NEUTRAL_QUERIES,
    PART2_QUERIES,
    TOP_K,
)
from project_progress.part_3.rank_statistics import holm_adjust, paired_permutation_test
from project_progress.part_3.run_rankings import RANKINGS_CSV, RESULTS_DIR


K_VALUES = (5, 10, 15, 20)
METRICS_CSV = RESULTS_DIR / "method_metrics.csv"
SUMMARY_CSV = RESULTS_DIR / "method_summary.csv"
SIGNIFICANCE_CSV = RESULTS_DIR / "significance.csv"
GROUPS_CSV = RESULTS_DIR / "query_group_summary.csv"
QUERY_GROUPS = {"gender_stated": PART2_QUERIES,
                "gender_unstated": GENDER_NEUTRAL_QUERIES}
POOL_REQUIRED = ["query_id", "pid", "relevance"]
RANKING_REQUIRED = ["query_id", "method", "rank", "pid"]
BASELINES = ("bm25", "tfidf")           # every method is compared with these
TESTED_METRICS = ("AP@K_pool", "NDCG@K_pool", "P@K")
SIGNIFICANCE_K = 10


def _require_columns(table: pd.DataFrame, columns: list[str], name: str):
    absent = [col for col in columns if col not in table.columns]
    if absent:
        raise ValueError(f"{name} is missing the columns {absent}")


def load_labels(path=POOL_CSV) -> pd.DataFrame:
    """Read the pool and check that every pair has exactly one 0/1 label.

    Returns:
        The pool with an integer ``relevance`` column.

    Raises:
        ValueError: If columns, queries or labels are missing, a pair is
            repeated, or a label is not binary.
    """
    pool = pd.read_csv(path, encoding="utf-8-sig",
                       dtype={"pid": str, "query_id": str})
    _require_columns(pool, POOL_REQUIRED, "The judgment pool")
    repeated = pool[pool.duplicated(["query_id", "pid"], keep=False)]
    if not repeated.empty:
        raise ValueError("Repeated (query_id, pid) pairs:\n"
                         + repeated[POOL_REQUIRED].to_string(index=False))
    absent = sorted(set(ALL_QUERIES) - set(pool["query_id"]))
    if absent:
        raise ValueError(f"No judgments for the queries {absent}")
    missing = pool[pool["relevance"].isna()]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} pool rows have no label yet, e.g.:\n"
            + missing[["query_id", "pid", "title"]].head(10).to_string(index=False)
        )
    if not pool["relevance"].isin([0, 1]).all():
        raise ValueError("Relevance labels must be 0 or 1.")
    pool["relevance"] = pool["relevance"].astype(int)
    return pool


def check_rankings(rankings: pd.DataFrame) -> None:
    """Each (method, query) list must have ranks 1..n, no repeated product,
    and every method must cover every query.

    Raises:
        ValueError: On the first inconsistency found.
    """
    _require_columns(rankings, RANKING_REQUIRED, "The rankings file")
    for (method, query_id), group in rankings.groupby(["method", "query_id"]):
        ranks = sorted(group["rank"])
        if ranks != list(range(1, len(ranks) + 1)) or len(ranks) > TOP_K:
            raise ValueError(f"{method}/{query_id}: ranks are not 1..n (n <= {TOP_K})")
        if group["pid"].duplicated().any():
            raise ValueError(f"{method}/{query_id}: a product appears twice")
    coverage = rankings.groupby("method")["query_id"].apply(set)
    for method, queries in coverage.items():
        if queries != set(ALL_QUERIES):
            raise ValueError(f"{method} misses the queries "
                             f"{sorted(set(ALL_QUERIES) - queries)}")


def reciprocal_rank_at_k(labels: list[int], k: int) -> float:
    """MRR@K component: 1/rank of the first relevant result within K, else 0."""
    return reciprocal_rank(labels[:k])


def evaluate(rankings: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Compute the metrics for every (method, query, K).

    Args:
        rankings: Output of ``run_rankings``.
        pool: Labelled pool from :func:`load_labels`.

    Returns:
        DataFrame with one row per method, query and K.
    """
    check_rankings(rankings)
    label_of = {(r.query_id, r.pid): r.relevance for r in pool.itertuples()}
    total_relevant = pool.groupby("query_id")["relevance"].sum().to_dict()

    rows = []
    for (method, query_id), group in rankings.groupby(["method", "query_id"]):
        ranked = group.sort_values("rank")["pid"]
        unjudged = [pid for pid in ranked if (query_id, pid) not in label_of]
        if unjudged:
            raise ValueError(f"{method}/{query_id}: {len(unjudged)} ranked "
                             f"products are not in the pool, e.g. {unjudged[0]}; "
                             "run export_judgment_pool and label them")
        labels = [label_of[(query_id, pid)] for pid in ranked]
        total = total_relevant[query_id]
        for k in K_VALUES:
            rows.append({
                "method": method,
                "query_id": query_id,
                "k": k,
                "P@K": precision_at_k(labels, k),
                "R@K_pool": recall_at_k(labels, k, total),
                "F1@K_pool": f1_at_k(labels, k, total),
                "AP@K_pool": average_precision_at_k(labels, k, total),
                "NDCG@K_pool": ndcg_at_k(labels, k, total),
                "RR@K": reciprocal_rank_at_k(labels, k),
                "relevant_in_pool": total,
            })
    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame, query_ids) -> pd.DataFrame:
    """Average the per-query metrics (MAP@K, MRR@K, mean P/R/F1/NDCG).

    Args:
        metrics: Output of :func:`evaluate`.
        query_ids: Queries to include (e.g. only the five Part 2 queries).

    Returns:
        One row per method and K.
    """
    subset = metrics[metrics["query_id"].isin(list(query_ids))]
    summary = subset.groupby(["method", "k"]).agg(
        mean_P=("P@K", "mean"),
        mean_R_pool=("R@K_pool", "mean"),
        mean_F1_pool=("F1@K_pool", "mean"),
        MAP_pool=("AP@K_pool", "mean"),
        mean_NDCG_pool=("NDCG@K_pool", "mean"),
        MRR=("RR@K", "mean"),
    ).reset_index()
    return summary


def group_summary(metrics: pd.DataFrame, k: int = SIGNIFICANCE_K) -> pd.DataFrame:
    """Mean P@K and MAP@K per method for queries with and without a gender.

    The groups also differ in product type, difficulty and pool size, so a
    difference between them is not caused by gender alone.

    Returns:
        One row per (group, method) with the number of queries.
    """
    at_k = metrics[metrics["k"] == k]
    rows = []
    for group, ids in QUERY_GROUPS.items():
        subset = at_k[at_k["query_id"].isin(list(ids))]
        for method, scores in subset.groupby("method"):
            rows.append({
                "group": group,
                "method": method,
                "queries": scores["query_id"].nunique(),
                f"mean_P@{k}": scores["P@K"].mean(),
                f"MAP@{k}_pool": scores["AP@K_pool"].mean(),
                "mean_relevant_in_pool": scores["relevant_in_pool"].mean(),
            })
    return pd.DataFrame(rows)


def significance_table(metrics: pd.DataFrame, k: int = SIGNIFICANCE_K) -> pd.DataFrame:
    """Paired randomisation tests of every method against each baseline.

    Each test pairs the two methods' scores on the SAME queries, so query
    difficulty cancels out. With 8 queries all 2^8 = 256 sign flips are
    enumerated (exact p-value); the smallest possible p-value is 2/256.

    Args:
        metrics: Output of :func:`evaluate`.
        k: Cut-off at which the metrics are compared.

    Returns:
        One row per (baseline, method, metric) with the mean difference
        (method - baseline), the raw p-value, the Holm-adjusted p-value over
        ALL the tests in the table, and whether each is below 0.05. The tests
        are exploratory: only the adjusted column controls the family-wise
        error of running many comparisons.
    """
    at_k = metrics[metrics["k"] == k]
    rows = []
    for baseline in BASELINES:
        for method in sorted(set(at_k["method"]) - {baseline}):
            for metric in TESTED_METRICS:
                table = at_k.pivot(index="query_id", columns="method", values=metric)
                if baseline not in table or method not in table:
                    continue
                diff, p_value = paired_permutation_test(table[method], table[baseline])
                rows.append({
                    "baseline": baseline,
                    "method": method,
                    "metric": metric.replace("K", str(k)),
                    "queries": len(table),
                    "mean_difference": diff,
                    "p_value": p_value,
                    "significant_5pct": p_value < 0.05,
                })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm_adjust(table["p_value"])
        table["significant_5pct_holm"] = table["p_holm"] < 0.05
    return table


def main():
    """Evaluate all methods and write the metrics CSV files."""
    rankings = pd.read_csv(RANKINGS_CSV, encoding="utf-8-sig",
                           dtype={"pid": str, "query_id": str})
    pool = load_labels()
    metrics = evaluate(rankings, pool)
    metrics.to_csv(METRICS_CSV, index=False, float_format="%.3f")

    summaries = []
    for name, ids in (("part2_queries", PART2_QUERIES),
                      ("all_queries", metrics["query_id"].unique())):
        summary = summarise(metrics, ids)
        summary.insert(0, "query_set", name)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.3f")

    groups = group_summary(metrics)
    groups.to_csv(GROUPS_CSV, index=False, float_format="%.3f")

    significance = significance_table(metrics)
    significance.to_csv(SIGNIFICANCE_CSV, index=False, float_format="%.4f")

    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    print("Per query, K = 10:")
    print(metrics[metrics["k"] == 10].to_string(index=False, float_format=fmt))
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=fmt))
    print(f"\nGender stated (Q1-Q5) vs not stated (Q6-Q8), K = {SIGNIFICANCE_K}:")
    print(groups.to_string(index=False, float_format=fmt))
    print(f"\nSignificance (K = {SIGNIFICANCE_K}, paired randomisation test):")
    print(significance.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()