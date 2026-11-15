"""Build the relevance-judgment pool for Part 3 (pooling method).

Different rankers return different top-20 lists. To compare them fairly,
every product that appears in ANY method's top 20 for a query must be
judged (this is the "pooling" approach used in TREC). Labels from Part 2
are reused automatically; only new (query, product) pairs are left empty.

Usage from the repository root (run after run_rankings):
    python -m project_progress.part_3.export_judgment_pool

The script never deletes or overwrites an existing label. The new pool is
the UNION of the rows already in ``judgments_pool.csv`` and the pairs in
the current top-20 lists: a judged product that no method retrieves any
more keeps its row (``in_current_top20`` becomes False), so its label
still counts in the recall / AP / NDCG denominators. Only the retrieval
provenance (``found_by``, ``in_current_top20``) is refreshed; labels,
``label_source`` and any extra columns (e.g. ``reason``) are kept as they
are. If the pool and Part 2 disagree on a label, the pool wins; the
conflict is printed until it is acknowledged by setting the row's
``label_source`` to ``part2_corrected``.
"""

from pathlib import Path

import pandas as pd

from project_progress.part_3.corpus import DATASET_PATH
from project_progress.part_3.run_rankings import RANKINGS_CSV


PART_DIR = Path(__file__).resolve().parent
PART2_JUDGMENTS = PART_DIR.parent / "part_2" / "test_queries_judgments.csv"
POOL_CSV = PART_DIR / "judgments_pool.csv"

KEY = ["query_id", "pid"]
POOL_COLUMNS = [
    "query_id", "query", "pid", "relevance", "label_source", "found_by",
    "in_current_top20", "title", "brand", "category", "sub_category",
    "description", "product_details", "url",
]
PRODUCT_COLUMNS = ["title", "brand", "category", "sub_category",
                   "description", "product_details", "url"]


def read_pool(path: Path = POOL_CSV) -> pd.DataFrame:
    """Read an existing pool file, or return an empty pool.

    Raises:
        ValueError: If a (query, product) pair appears more than once.
    """
    if not path.exists():
        return pd.DataFrame(columns=POOL_COLUMNS)
    pool = pd.read_csv(path, encoding="utf-8-sig",
                       dtype={"pid": str, "query_id": str})
    duplicates = pool[pool.duplicated(KEY, keep=False)]
    if not duplicates.empty:
        raise ValueError("Duplicate (query_id, pid) rows in the pool:\n"
                         + duplicates[KEY + ["relevance"]].to_string(index=False))
    return pool


def read_part2_labels(path: Path = PART2_JUDGMENTS) -> dict[tuple[str, str], int]:
    """Labels assigned in Part 2: ``{(query_id, pid): 0/1}``."""
    if not path.exists():
        return {}
    part2 = pd.read_csv(path, encoding="utf-8-sig",
                        dtype={"pid": str, "query_id": str})
    return {(r.query_id, r.pid): int(r.relevance) for r in part2.itertuples()}


def label_conflicts(pool: pd.DataFrame, part2: dict) -> pd.DataFrame:
    """Unacknowledged pool labels that differ from the Part 2 label.

    Rows marked ``label_source == "part2_corrected"`` are deliberate,
    documented corrections and are not reported.
    """
    labelled = pool.dropna(subset=["relevance"])
    labelled = labelled[labelled["label_source"] != "part2_corrected"]
    rows = [
        {"query_id": r.query_id, "pid": r.pid,
         "pool_label": int(r.relevance), "part2_label": part2[(r.query_id, r.pid)]}
        for r in labelled.itertuples()
        if (r.query_id, r.pid) in part2
        and int(r.relevance) != part2[(r.query_id, r.pid)]
    ]
    return pd.DataFrame(rows, columns=["query_id", "pid", "pool_label", "part2_label"])


def build_pool(rankings: pd.DataFrame, products: pd.DataFrame,
               existing: pd.DataFrame, part2: dict) -> pd.DataFrame:
    """Union of the existing pool and the current top-20 pairs.

    Args:
        rankings: Output of ``run_rankings`` (query_id, query, method, pid...).
        products: Raw dataset indexed by pid (for the judging columns).
        existing: Current pool from :func:`read_pool` (may be empty).
        part2: Part 2 labels from :func:`read_part2_labels`.

    Returns:
        DataFrame with ``POOL_COLUMNS`` (plus any extra columns of
        ``existing``); ``relevance`` is empty only for pairs that nobody
        has judged yet.
    """
    current = (
        rankings.groupby(["query_id", "query", "pid"])["method"]
        .apply(lambda methods: ",".join(sorted(set(methods))))
        .reset_index(name="found_by")
    )
    found_by = {(r.query_id, r.pid): r.found_by for r in current.itertuples()}

    pool = existing.copy()
    known = set(zip(pool["query_id"], pool["pid"]))
    new_rows = []
    for row in current.itertuples():
        if (row.query_id, row.pid) in known:
            continue
        label = part2.get((row.query_id, row.pid))
        product = products.loc[row.pid]
        new_rows.append({
            "query_id": row.query_id,
            "query": row.query,
            "pid": row.pid,
            "relevance": label,
            "label_source": "new" if label is None else "part2",
            **{col: str(product[col]) if col == "product_details" else product[col]
               for col in PRODUCT_COLUMNS},
        })
    if new_rows:
        pool = pd.concat([pool, pd.DataFrame(new_rows)], ignore_index=True)

    # Refresh retrieval provenance only; labels stay untouched.
    keys = list(zip(pool["query_id"], pool["pid"]))
    pool["found_by"] = [found_by.get(key, "") for key in keys]
    pool["in_current_top20"] = [key in found_by for key in keys]
    # A label filled in by hand on a "new" row is a Part 3 judgment.
    pool.loc[pool["relevance"].notna() & (pool["label_source"] == "new"),
             "label_source"] = "part3"

    extra = [col for col in pool.columns if col not in POOL_COLUMNS]
    pool = pool[POOL_COLUMNS + extra]
    pool["relevance"] = pool["relevance"].astype("Int64")
    return pool.sort_values(["query_id", "label_source", "pid"]).reset_index(drop=True)


def main():
    """Write (or extend) ``judgments_pool.csv`` and report what is missing."""
    rankings = pd.read_csv(RANKINGS_CSV, encoding="utf-8-sig",
                           dtype={"pid": str, "query_id": str})
    products = pd.read_json(DATASET_PATH, dtype={"pid": str}).set_index("pid")
    existing = read_pool()
    part2 = read_part2_labels()

    conflicts = label_conflicts(existing, part2)
    if not conflicts.empty:
        print("Labels that differ from Part 2 (the pool label is kept; set "
              "label_source to 'part2_corrected' once checked):")
        print(conflicts.to_string(index=False), "\n")

    pool = build_pool(rankings, products, existing, part2)
    pool.to_csv(POOL_CSV, index=False, encoding="utf-8-sig")

    summary = pool.groupby("query_id").agg(
        pool_size=("pid", "size"),
        labelled=("relevance", lambda s: s.notna().sum()),
        to_label=("relevance", lambda s: s.isna().sum()),
        not_in_current_top20=("in_current_top20", lambda s: (~s).sum()),
    )
    print(summary.to_string())
    print(f"\nSaved {len(pool)} rows to {POOL_CSV}")
    print("Fill the empty 'relevance' cells with 1 or 0. Do not open/save "
          "the file with a locale that changes the separator.")


if __name__ == "__main__":
    main()
