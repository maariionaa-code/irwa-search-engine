"""Evaluate every ranking method with the pooled relevance judgments.

For each query, the set of known relevant products is the same for all
methods (the relevant products in the pool), so recall, AP and NDCG are
comparable across methods. The Part 2 metric functions are reused.

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
from project_progress.part_3.queries import PART2_QUERIES
from project_progress.part_3.run_rankings import RANKINGS_CSV, RESULTS_DIR


K_VALUES = (5, 10, 20)
METRICS_CSV = RESULTS_DIR / "method_metrics.csv"
SUMMARY_CSV = RESULTS_DIR / "method_summary.csv"


def load_labels() -> pd.DataFrame:
    """Read the pool and check that every pair has a 0/1 label.

    Returns:
        The pool with an integer ``relevance`` column.

    Raises:
        ValueError: If labels are missing or not binary.
    """
    pool = pd.read_csv(POOL_CSV, encoding="utf-8-sig",
                       dtype={"pid": str, "query_id": str})
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
    label_of = {(r.query_id, r.pid): r.relevance for r in pool.itertuples()}
    total_relevant = pool.groupby("query_id")["relevance"].sum().to_dict()

    rows = []
    for (method, query_id), group in rankings.groupby(["method", "query_id"]):
        labels = [label_of[(query_id, pid)]
                  for pid in group.sort_values("rank")["pid"]]
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

    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    print("Per query, K = 10:")
    print(metrics[metrics["k"] == 10].to_string(index=False, float_format=fmt))
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()