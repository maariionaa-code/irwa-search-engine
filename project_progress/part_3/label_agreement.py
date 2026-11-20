"""Inter-annotator agreement on a blind sample of the judgment pool.

Why: our pool labels were assigned manually by the group. The judging
rules use the title and structured attributes, which the intent score
also reads, so the labels could share assumptions (and mistakes) with
that ranker. A second group member who does not see the existing labels
or which method retrieved a product gives an independent check.

Steps, from the repository root:
    python -m project_progress.part_3.label_agreement --export   # 1. write the sample
    # 2. a second person fills the empty ``relevance`` column of
    #    agreement_sample.csv (1 = relevant, 0 = not), using only the
    #    judging rules of the report (Section 6.3)
    python -m project_progress.part_3.label_agreement            # 3. agreement

The sample is stratified by query (the same share of every query's pool)
and drawn with a fixed seed, so it is reproducible. It contains no
existing label, ``found_by`` or URL.
"""

import argparse

import pandas as pd

from project_progress.part_3.export_judgment_pool import POOL_CSV

SAMPLE_CSV = POOL_CSV.parent / "agreement_sample.csv"
SAMPLE_SIZE = 100
SEED = 42
BLIND_COLUMNS = ["query_id", "query", "pid", "title", "sub_category",
                 "description", "product_details"]


def draw_sample(pool: pd.DataFrame, size: int = SAMPLE_SIZE,
                seed: int = SEED) -> pd.DataFrame:
    """Stratified random sample of (query, product) pairs, shuffled.

    Each query contributes in proportion to its pool size (at least one
    pair), and the rows are shuffled so that the order reveals nothing.

    Returns:
        The blind columns plus an empty ``relevance`` column.
    """
    share = size / len(pool)
    parts = [group.sample(n=max(1, round(len(group) * share)), random_state=seed)
             for _, group in pool.groupby("query_id")]
    sample = pd.concat(parts).sample(frac=1, random_state=seed)
    sample = sample[[c for c in BLIND_COLUMNS if c in sample.columns]].copy()
    sample["relevance"] = pd.NA
    return sample.reset_index(drop=True)


def cohen_kappa(a, b) -> float:
    """Cohen's kappa for two binary label lists of equal length.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is the observed agreement
    and p_e the agreement expected by chance from each annotator's
    share of 1s. Returns 1.0 when both annotators use a single identical
    label throughout (p_e = 1).
    """
    a, b = list(map(int, a)), list(map(int, b))
    if len(a) != len(b) or not a:
        raise ValueError("Two non-empty label lists of equal length are needed")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    p_e = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def agreement(pool: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Compare the second annotator's labels with the pool labels.

    Returns:
        Number of pairs, observed agreement, Cohen's kappa and the
        disagreements (query_id, pid, title, pool label, second label).
    """
    done = sample.dropna(subset=["relevance"])
    merged = done.merge(pool[["query_id", "pid", "relevance"]],
                        on=["query_id", "pid"], suffixes=("_second", "_pool"))
    first = merged["relevance_pool"].astype(int)
    second = merged["relevance_second"].astype(int)
    return {
        "pairs": len(merged),
        "observed_agreement": float((first == second).mean()) if len(merged) else float("nan"),
        "cohen_kappa": cohen_kappa(first, second) if len(merged) else float("nan"),
        "disagreements": merged.loc[first != second,
                                    ["query_id", "pid", "title",
                                     "relevance_pool", "relevance_second"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", action="store_true",
                        help="write the blind sample (never overwrites labels)")
    args = parser.parse_args()
    pool = pd.read_csv(POOL_CSV, encoding="utf-8-sig",
                       dtype={"pid": str, "query_id": str})
    if args.export:
        if SAMPLE_CSV.exists():
            print(f"{SAMPLE_CSV.name} already exists; not overwritten")
            return
        draw_sample(pool).to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")
        print(f"Wrote {SAMPLE_SIZE} blind pairs to {SAMPLE_CSV.name}")
        return
    sample = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig",
                         dtype={"pid": str, "query_id": str})
    result = agreement(pool, sample)
    print(f"Pairs labelled by both: {result['pairs']}")
    print(f"Observed agreement: {result['observed_agreement']:.3f}")
    print(f"Cohen's kappa:      {result['cohen_kappa']:.3f}")
    if not result["disagreements"].empty:
        print("\nDisagreements:")
        print(result["disagreements"].to_string(index=False))


if __name__ == "__main__":
    main()
