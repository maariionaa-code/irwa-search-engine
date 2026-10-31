"""Export ranked product candidates for manual relevance assessment (IRWA Part 2).

Usage from the repository root:
    python -m project_progress.part_2.prepare_test_queries

This script DOES NOT assign relevance labels; those require human judgment.
"""

from pathlib import Path

import pandas as pd

from project_progress.part_2.indexing import (
    build_inverted_index,
    load_processed_documents,
)
from project_progress.part_2.ranking import rank_tfidf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "fashion_products_dataset.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "test_queries_judgments.csv"

# Five distinct information needs chosen from the clothing corpus.
TEST_QUERIES = {
    "Q1": "women cotton kurta",
    "Q2": "men slim fit jeans",
    "Q3": "men running shoes",
    "Q4": "women full sleeve jacket",
    "Q5": "men cotton sports shorts",
}
TOP_K = 20


def export_candidates() -> None:
    """Rank each query and export the top K with empty manual relevance labels."""
    documents = load_processed_documents()
    inverted_index = build_inverted_index(documents)
    products = pd.read_json(DATASET_PATH).set_index("pid")

    rows = []
    for query_id, query in TEST_QUERIES.items():
        ranked = rank_tfidf(query, documents, inverted_index, top_k=TOP_K)
        print(f"{query_id}: {query!r} -> {len(ranked)} candidates exported")

        for rank, (pid, score) in enumerate(ranked, start=1):
            product = products.loc[pid]
            rows.append({
                "query_id": query_id,
                "query": query,
                "rank": rank,
                "pid": pid,
                "tfidf_score": round(score, 6),
                "title": product["title"],
                "brand": product["brand"],
                "category": product["category"],
                "sub_category": product["sub_category"],
                "description": product["description"],
                "product_details": str(product["product_details"]),
                "url": product["url"],
                "relevance": "",  # Fill manually with 1 (relevant) or 0 (not relevant).
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nExported {len(df)} rows to: {OUTPUT_PATH}")
    print("Fill the 'relevance' column manually; do not invent labels.")


if __name__ == "__main__":
    export_candidates()