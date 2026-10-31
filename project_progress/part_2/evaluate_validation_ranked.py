
"""Evaluate TF-IDF ranking using the professor's relevance labels."""

from pathlib import Path
import pandas as pd

from project_progress.part_2.indexing import (
    build_inverted_index,
    load_processed_documents,
)
from project_progress.part_2.ranking import rank_tfidf
from project_progress.part_2.evaluation import (
    average_precision_at_k,
    f1_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = ROOT / "data" / "validation_labels.csv"
OUTPUT_PATH = (
    Path(__file__).resolve().parent / "validation_ranked_metrics.csv"
)

QUERY_TEXT = {
    1: "women full sleeve sweatshirt cotton",
    2: "men slim jeans blue",
}

K_VALUES = (5, 10, 20)


def evaluate():
    """Evaluate the ranked results within the judged product pool."""

    judgments = pd.read_csv(
        LABELS_PATH,
        dtype={"pid": str}
    )

    expected_columns = {"title", "pid", "query_id", "labels"}

    if not expected_columns.issubset(judgments.columns):
        raise ValueError("Unexpected validation CSV columns")

    # Build the search engine
    documents = load_processed_documents()
    inverted_index = build_inverted_index(documents)

    metrics_rows = []
    relevance_sequences = []
    relevant_totals = []

    for query_id, query_text in QUERY_TEXT.items():

        judged = judgments[
            judgments["query_id"] == query_id
        ].copy()

        if len(judged) != 20 or judged["pid"].duplicated().any():
            raise ValueError(
                f"Query {query_id}: expected 20 distinct products"
            )

        if not judged["pid"].isin(documents).all():
            raise ValueError(
                f"Query {query_id}: missing product IDs"
            )

        if not judged["labels"].isin([0, 1]).all():
            raise ValueError(
                f"Query {query_id}: invalid relevance labels"
            )

        judged_labels = dict(
            zip(judged["pid"], judged["labels"].astype(int))
        )

        total_relevant = sum(judged_labels.values())

        # Run our AND retrieval and TF-IDF ranking
        full_ranking = rank_tfidf(
            query_text,
            documents,
            inverted_index
        )

        # Retain only products judged by the professor
        judged_ranking = [
            pid for pid, score in full_ranking
            if pid in judged_labels
        ]

        labels = [
            judged_labels[pid]
            for pid in judged_ranking
        ]

        relevance_sequences.append(labels)
        relevant_totals.append(total_relevant)

        print(
            f"Query {query_id}: "
            f"{len(judged_ranking)}/20 judged products retrieved; "
            f"{sum(labels)}/{total_relevant} relevant retrieved"
        )

        # Calculate metrics
        for k in K_VALUES:

            metrics_rows.append({
                "query_id": query_id,
                "k": k,
                "P@K_pool": precision_at_k(labels, k),
                "R@K_pool": recall_at_k(
                    labels, k, total_relevant
                ),
                "F1@K_pool": f1_at_k(
                    labels, k, total_relevant
                ),
                "AP@K_pool": average_precision_at_k(
                    labels, k, total_relevant
                ),
                "NDCG@K_pool": ndcg_at_k(
                    labels, k, total_relevant
                ),
            })

    # Save results
    results_df = pd.DataFrame(metrics_rows)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        float_format="%.3f"
    )

    print("\nRANKED EVALUATION WITHIN JUDGED POOL")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}"
        )
    )

    # Overall metrics
    for k in K_VALUES:

        map_score = mean_average_precision(
            relevance_sequences,
            relevant_totals,
            k
        )

        print(f"MAP@{k}_pool: {map_score:.3f}")

    mrr = mean_reciprocal_rank(relevance_sequences)

    print(f"MRR_pool: {mrr:.3f}")

    print(f"\nSaved: {OUTPUT_PATH}")

    print(
        "\nNote: Evaluation is restricted to the "
        "20 judged products per query."
    )


if __name__ == "__main__":
    evaluate()
