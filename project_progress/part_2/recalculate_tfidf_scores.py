
"""Recalculate TF-IDF scores while preserving human judgments."""

from pathlib import Path
import pandas as pd

from project_progress.part_2.indexing import (
    load_processed_documents,
    build_inverted_index,
)
from project_progress.part_2.ranking import rank_tfidf


PART_DIR = Path(__file__).resolve().parent

INPUT_PATH = PART_DIR / "test_queries_judgments.csv"
OUTPUT_PATH = PART_DIR / "test_queries_judgments_verified.csv"


def recalculate_scores():

    # Load our labeled CSV
    with INPUT_PATH.open("r", encoding="utf-8-sig") as f:
        header = f.readline()

    separator = ";" if header.count(";") > header.count(",") else ","

    df = pd.read_csv(
        INPUT_PATH,
        sep=separator,
        encoding="utf-8-sig",
        dtype={"pid": str, "query_id": str},
    )

    assert len(df) == 100
    assert df["relevance"].isin([0, 1]).all()

    # Check the revised relevance judgments
    expected = {
        "Q1": 20,
        "Q2": 4,
        "Q3": 12,
        "Q4": 20,
        "Q5": 14,
    }

    actual = df.groupby("query_id")["relevance"].sum().to_dict()
    assert actual == expected, f"Unexpected labels: {actual}"

    # Build our search engine using Part 1 preprocessing
    documents = load_processed_documents()
    inverted_index = build_inverted_index(documents)

    # Recompute the 20 ranked results for every query
    for query_id, group in df.groupby("query_id"):

        group = group.sort_values("rank")
        query = group["query"].iloc[0]

        ranked = rank_tfidf(
            query,
            documents,
            inverted_index,
            top_k=20,
        )

        saved_pids = group["pid"].tolist()
        recalculated_pids = [pid for pid, score in ranked]

        # Ensure that we are not modifying the ranking
        if saved_pids != recalculated_pids:
            raise ValueError(
                f"Ranking mismatch for {query_id}. "
                "Review the original CSV before continuing."
            )

        # Replace only the scores
        for row_index, (pid, score) in zip(group.index, ranked):
            df.at[row_index, "tfidf_score"] = round(score, 6)

        print(f"{query_id}: scores recalculated successfully")

    # Check that all scores are valid cosine similarities
    assert df["tfidf_score"].between(0, 1).all()

    # Save a new CSV without overwriting the original
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    print("\nAll 100 scores verified!")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    recalculate_scores()
