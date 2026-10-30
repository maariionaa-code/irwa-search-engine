"""Evaluate the two predefined queries using validation_labels.csv."""

from pathlib import Path
import pandas as pd

from project_progress.part_2.evaluation import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation_labels.csv"


def evaluate_validation():
    """Calculate evaluation metrics for the provided relevance judgments."""

    df = pd.read_csv(VALIDATION_PATH)

    # Keep the original CSV order within each query
    labels_by_query = {
        query_id: group["labels"].astype(int).tolist()
        for query_id, group in df.groupby("query_id", sort=True)
    }

    all_labels = list(labels_by_query.values())
    total_relevants = [sum(labels) for labels in all_labels]

    for k in [5, 10, 20]:
        print(f"\n========== K = {k} ==========")

        for query_id, labels in labels_by_query.items():
            total_relevant = sum(labels)

            print(f"\nQuery {query_id}")
            print(f"P@{k}: {precision_at_k(labels, k):.3f}")
            print(f"R@{k}: {recall_at_k(labels, k, total_relevant):.3f}")
            print(f"F1@{k}: {f1_at_k(labels, k, total_relevant):.3f}")
            print(f"AP@{k}: {average_precision_at_k(labels, k, total_relevant):.3f}")
            print(f"NDCG@{k}: {ndcg_at_k(labels, k, total_relevant):.3f}")

        map_score = mean_average_precision(
            all_labels, total_relevants, k
        )
        print(f"\nMAP@{k}: {map_score:.3f}")

    print(f"\nMRR: {mean_reciprocal_rank(all_labels):.3f}")


if __name__ == "__main__":
    evaluate_validation()