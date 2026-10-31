
"""Evaluate the five custom queries using human relevance labels."""

from pathlib import Path
import pandas as pd

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


PART_DIR = Path(__file__).resolve().parent
INPUT_CSV = PART_DIR / "test_queries_judgments.csv"
OUTPUT_CSV = PART_DIR / "test_queries_metrics.csv"
SUMMARY_CSV = PART_DIR / "test_queries_summary.csv"

EXPECTED_QUERIES = {
    "Q1": "women cotton kurta",
    "Q2": "men slim fit jeans",
    "Q3": "men running shoes",
    "Q4": "women full sleeve jacket",
    "Q5": "men cotton sports shorts",
}

K_VALUES = [5, 10, 20]


def read_judgments():
    """Load and validate the relevance judgments."""

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_CSV}")

    # Detect comma or semicolon CSV separator
    with INPUT_CSV.open("r", encoding="utf-8-sig") as f:
        header = f.readline()

    separator = ";" if header.count(";") > header.count(",") else ","

    df = pd.read_csv(
        INPUT_CSV,
        encoding="utf-8-sig",
        sep=separator,
        dtype={"pid": str, "query_id": str},
    )

    required = {"query_id", "query", "rank", "pid", "relevance"}

    if not required.issubset(df.columns):
        raise ValueError("Missing required CSV columns.")

    if len(df) != 100:
        raise ValueError(f"Expected 100 products, found {len(df)}")

    # Verify relevance values
    labels = pd.to_numeric(df["relevance"], errors="coerce")

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("All relevance values must be 0 or 1.")

    df["relevance"] = labels.astype(int)
    df["rank"] = pd.to_numeric(df["rank"]).astype(int)

    # Validate each query and its 20 results
    if set(df["query_id"]) != set(EXPECTED_QUERIES):
        raise ValueError("Unexpected query IDs.")

    for query_id, query in EXPECTED_QUERIES.items():

        subset = df[df["query_id"] == query_id]

        if len(subset) != 20:
            raise ValueError(f"{query_id} must contain 20 products.")

        if set(subset["query"].astype(str).str.strip()) != {query}:
            raise ValueError(f"Incorrect query text for {query_id}.")

        if sorted(subset["rank"]) != list(range(1, 21)):
            raise ValueError(f"Invalid ranks for {query_id}.")

        if subset["pid"].duplicated().any():
            raise ValueError(f"Duplicate product IDs in {query_id}.")

    # Check for potential Excel decimal conversion errors
    if "tfidf_score" in df.columns:

        scores = pd.to_numeric(df["tfidf_score"], errors="coerce")

        if scores.isna().any() or (scores < 0).any() or (scores > 1).any():
            print("WARNING: TF-IDF scores may have been altered by Excel.")
            print("Evaluation will use the original rank order.")

    return df


def evaluate_test_queries():
    """Calculate evaluation metrics for all five queries."""

    df = read_judgments()

    results = []
    all_labels = []
    total_relevants = []

    for query_id in EXPECTED_QUERIES:

        subset = (
            df[df["query_id"] == query_id]
            .sort_values("rank")
        )

        labels = subset["relevance"].tolist()
        total_relevant = sum(labels)

        all_labels.append(labels)
        total_relevants.append(total_relevant)

        for k in K_VALUES:

            results.append({
                "query_id": query_id,
                "k": k,
                "P@K": precision_at_k(labels, k),
                "R@K_pool": recall_at_k(labels, k, total_relevant),
                "F1@K_pool": f1_at_k(labels, k, total_relevant),
                "AP@K_pool": average_precision_at_k(
                    labels, k, total_relevant
                ),
                "NDCG@K_pool": ndcg_at_k(
                    labels, k, total_relevant
                ),
                "RR@20": reciprocal_rank(labels),
            })

    # Individual query metrics
    metrics_df = pd.DataFrame(results)

    metrics_df.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.3f"
    )

    # Overall evaluation metrics
    summary = []

    for k in K_VALUES:

        subset = metrics_df[metrics_df["k"] == k]

        summary.append({
            "k": k,
            "MAP@K_pool": mean_average_precision(
                all_labels, total_relevants, k
            ),
            "mean_P@K": subset["P@K"].mean(),
            "mean_NDCG@K_pool": subset["NDCG@K_pool"].mean(),
            "MRR@20": mean_reciprocal_rank(all_labels),
        })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
        float_format="%.3f"
    )

    print("\nEVALUATION RESULTS PER QUERY")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nOVERALL EVALUATION")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nFiles generated:")
    print(OUTPUT_CSV.name)
    print(SUMMARY_CSV.name)

    print("\nNote: Recall and related metrics use the")
    print("20 labeled candidates, not the full dataset.")


if __name__ == "__main__":
    evaluate_test_queries()
