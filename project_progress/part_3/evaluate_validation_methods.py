"""Evaluate every Part 3 method on the PROFESSOR'S validation labels.

These labels (data/validation_labels.csv, 20 judged products for each of two
queries) were made by the course staff, not by us, so they are an
independent check of our own judgments.

Two complementary views, as in Part 2:

* ``pool``  - keep only the judged products, in the order the method ranks
  them (unjudged products are skipped). Measures how well each method
  ORDERS the judged products.
* ``full``  - the method's real top-K over all AND candidates; unjudged
  products count as non-relevant (a conservative lower bound). The column
  ``judged@K`` reports how many of the top K were actually judged, so the
  reader can see how much of the score rests on unknown products.

Usage from the repository root:
    python -m project_progress.part_3.evaluate_validation_methods [--no-embeddings]
"""

import argparse

import pandas as pd

from project_progress.part_2.evaluation import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from project_progress.part_3.corpus import PROJECT_ROOT
from project_progress.part_3.pipeline import SearchEngine
from project_progress.part_3.run_rankings import RESULTS_DIR


LABELS_PATH = PROJECT_ROOT / "data" / "validation_labels.csv"
OUTPUT_CSV = RESULTS_DIR / "validation_methods.csv"
SUMMARY_CSV = RESULTS_DIR / "validation_methods_summary.csv"

# The two queries of validation_labels.csv (same text as in Part 2).
VALIDATION_QUERIES = {
    1: "women full sleeve sweatshirt cotton",
    2: "men slim jeans blue",
}
K_VALUES = (5, 10, 15, 20)


def load_validation_labels() -> dict[int, dict[str, int]]:
    """Read the professor's labels.

    Returns:
        ``{query_id: {pid: 0 or 1}}``.
    """
    labels = pd.read_csv(LABELS_PATH, dtype={"pid": str})
    return {
        int(query_id): dict(zip(group["pid"], group["labels"].astype(int)))
        for query_id, group in labels.groupby("query_id")
    }


def metric_rows(labels: list[int], total: int, judged: list[bool], **keys):
    """Metrics of one ranked label list at every K.

    Args:
        labels: 0/1 relevance of the ranked products.
        total: Number of relevant judged products for the query.
        judged: Whether each ranked product has a label (full view only).
        **keys: Identifying columns (method, query_id, view).

    Returns:
        One dict per K.
    """
    rows = []
    for k in K_VALUES:
        rows.append({
            **keys,
            "k": k,
            "P@K": precision_at_k(labels, k),
            "R@K": recall_at_k(labels, k, total),
            "AP@K": average_precision_at_k(labels, k, total),
            "NDCG@K": ndcg_at_k(labels, k, total),
            "RR@K": reciprocal_rank(labels[:k]),
            "judged@K": sum(judged[:k]),
        })
    return rows


def evaluate(engine: SearchEngine) -> pd.DataFrame:
    """Run every method on both validation queries, in both views.

    Args:
        engine: A built :class:`SearchEngine`.

    Returns:
        DataFrame with one row per method, query, view and K.
    """
    judgments = load_validation_labels()
    rows = []
    for query_id, query in VALIDATION_QUERIES.items():
        judged = judgments[query_id]
        total = sum(judged.values())
        for method in engine.methods():
            ranking = [pid for pid, _ in engine.rank(query, method)]

            in_pool = [pid for pid in ranking if pid in judged]
            rows += metric_rows([judged[pid] for pid in in_pool], total,
                                [True] * len(in_pool), method=method,
                                query_id=query_id, view="pool",
                                judged_retrieved=len(in_pool))

            top = ranking[:max(K_VALUES)]
            rows += metric_rows([judged.get(pid, 0) for pid in top], total,
                                [pid in judged for pid in top], method=method,
                                query_id=query_id, view="full",
                                judged_retrieved=len(in_pool))
    return pd.DataFrame(rows)


def main():
    """Write the per-query metrics and the means over the two queries."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    engine = SearchEngine(use_embeddings=not args.no_embeddings)
    RESULTS_DIR.mkdir(exist_ok=True)

    results = evaluate(engine)
    results.to_csv(OUTPUT_CSV, index=False, float_format="%.3f")

    summary = (results.groupby(["view", "method", "k"])
               [["P@K", "R@K", "AP@K", "NDCG@K", "RR@K", "judged@K"]]
               .mean().reset_index()
               .rename(columns={"AP@K": "MAP@K", "RR@K": "MRR@K"}))
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.3f")

    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    print("Judged products that pass the AND filter (same for every method):")
    first = results[(results["view"] == "pool") & (results["k"] == 5)]
    print(first[["method", "query_id", "judged_retrieved"]]
          .drop_duplicates("query_id").to_string(index=False))
    print("\nMeans over the two validation queries, K = 10:")
    print(summary[summary["k"] == 10].to_string(index=False, float_format=fmt))
    print(f"\nSaved {OUTPUT_CSV.name} and {SUMMARY_CSV.name}")


if __name__ == "__main__":
    main()
