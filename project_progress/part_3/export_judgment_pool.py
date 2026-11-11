"""Build the relevance-judgment pool for Part 3 (pooling method).

Different rankers return different top-20 lists. To compare them fairly,
every product that appears in ANY method's top 20 for a query must be
judged (this is the "pooling" approach used in TREC). Labels from Part 2
are reused automatically; only new (query, product) pairs are left empty.

Usage from the repository root (run after run_rankings):
    python -m project_progress.part_3.export_judgment_pool

The script never deletes or overwrites an existing label: if
``judgments_pool.csv`` already exists, it only appends new pairs.
"""

from pathlib import Path

import pandas as pd

from project_progress.part_3.corpus import DATASET_PATH
from project_progress.part_3.run_rankings import RANKINGS_CSV


PART_DIR = Path(__file__).resolve().parent
PART2_JUDGMENTS = PART_DIR.parent / "part_2" / "test_queries_judgments.csv"
POOL_CSV = PART_DIR / "judgments_pool.csv"

POOL_COLUMNS = [
    "query_id", "query", "pid", "relevance", "label_source", "found_by",
    "title", "brand", "category", "sub_category", "description",
    "product_details", "url",
]


def load_existing_labels() -> dict[tuple[str, str], tuple[int, str]]:
    """Collect labels already assigned in Part 2 and in a previous pool.

    Returns:
        ``{(query_id, pid): (label, source)}``. Pool labels take priority
        over Part 2 labels (they may contain later corrections).
    """
    labels = {}
    if PART2_JUDGMENTS.exists():
        part2 = pd.read_csv(PART2_JUDGMENTS, encoding="utf-8-sig",
                            dtype={"pid": str, "query_id": str})
        for _, row in part2.iterrows():
            labels[(row["query_id"], row["pid"])] = (int(row["relevance"]), "part2")
    if POOL_CSV.exists():
        pool = pd.read_csv(POOL_CSV, encoding="utf-8-sig",
                           dtype={"pid": str, "query_id": str})
        for _, row in pool.dropna(subset=["relevance"]).iterrows():
            labels[(row["query_id"], row["pid"])] = (
                int(row["relevance"]), row["label_source"]
            )
    return labels


def build_pool(rankings: pd.DataFrame, products: pd.DataFrame,
               labels: dict) -> pd.DataFrame:
    """Create one row per distinct (query, product) in any top-20 list.

    Args:
        rankings: Output of ``run_rankings`` (query_id, query, method, pid...).
        products: Raw dataset indexed by pid (for the judging columns).
        labels: Existing labels from :func:`load_existing_labels`.

    Returns:
        DataFrame with ``POOL_COLUMNS``; ``relevance`` is empty for pairs
        that still need a human judgment.
    """
    found_by = (
        rankings.groupby(["query_id", "query", "pid"])["method"]
        .apply(lambda methods: ",".join(sorted(set(methods))))
        .reset_index(name="found_by")
    )
    rows = []
    for _, row in found_by.iterrows():
        product = products.loc[row["pid"]]
        label, source = labels.get((row["query_id"], row["pid"]), (None, "new"))
        rows.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "pid": row["pid"],
            "relevance": label,
            "label_source": source,
            "found_by": row["found_by"],
            "title": product["title"],
            "brand": product["brand"],
            "category": product["category"],
            "sub_category": product["sub_category"],
            "description": product["description"],
            "product_details": str(product["product_details"]),
            "url": product["url"],
        })
    pool = pd.DataFrame(rows, columns=POOL_COLUMNS)
    pool["relevance"] = pool["relevance"].astype("Int64")
    return pool.sort_values(["query_id", "label_source", "pid"])


def main():
    """Write (or extend) ``judgments_pool.csv`` and report what is missing."""
    rankings = pd.read_csv(RANKINGS_CSV, encoding="utf-8-sig",
                           dtype={"pid": str, "query_id": str})
    products = pd.read_json(DATASET_PATH, dtype={"pid": str}).set_index("pid")

    pool = build_pool(rankings, products, load_existing_labels())
    pool.to_csv(POOL_CSV, index=False, encoding="utf-8-sig")

    summary = pool.groupby("query_id").agg(
        pool_size=("pid", "size"),
        labelled=("relevance", lambda s: s.notna().sum()),
        to_label=("relevance", lambda s: s.isna().sum()),
    )
    print(summary.to_string())
    print(f"\nSaved {len(pool)} rows to {POOL_CSV}")
    print("Fill the empty 'relevance' cells with 1 or 0 and set "
          "'label_source' to 'part3'. Do not open/save the file with a "
          "locale that changes the separator.")


if __name__ == "__main__":
    main()