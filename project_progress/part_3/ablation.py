"""Ablation study and BM25 parameter sensitivity (Part 3 extras).

Questions answered:
1. Does every component of OUR SCORE help? Each variant removes one
   component (weight set to 0, or the stock factor set to 1).
2. Does every factor of the INTENT SCORE help? Each variant replaces one
   factor (category K, gender G, attributes A) by 1.
3. How sensitive is BM25 to its parameters k1 and b?

Every variant is evaluated on two label sets:
* ``validation`` - the professor's labels (independent of our judgments);
* ``pool``       - our judged pool for Q1-Q8.
Variants can surface products that nobody judged. Those count as
non-relevant (conservative for P@10 only) and ``judged@10`` shows how many
of the top 10 were actually judged. Because the variants do not have equal
judgment coverage, the ``*_condensed`` columns repeat P@10 and AP@10 on the
"condensed" list (unjudged products removed, Sakai 2007), which compares the
variants only on what was judged. Both views are reported; a conclusion is
treated as provisional unless they agree.

Products that only an ablation variant or a BM25 grid cell puts in its
top 10 are not in the method pool. They are judged separately, in
``judgments_ablation.csv``, so that the method comparison (which uses the
pool only) does not change. The ablation uses the union of both files.

Usage from the repository root (after labelling judgments_pool.csv):
    python -m project_progress.part_3.ablation --export-unjudged   # 1. list them
    # 2. fill the empty ``relevance`` cells of judgments_ablation.csv
    python -m project_progress.part_3.ablation                     # 3. evaluate
"""

import argparse

import pandas as pd

from project_progress.part_2.evaluation import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
)
from project_progress.part_3.evaluate_validation_methods import (
    VALIDATION_QUERIES,
    load_validation_labels,
)
from project_progress.part_3.export_judgment_pool import POOL_CSV
from project_progress.part_3.pipeline import SearchEngine
from project_progress.part_3.queries import ALL_QUERIES
from project_progress.part_3.ranking_methods import (
    OUR_SCORE_WEIGHTS,
    combine_our_score,
    our_score_components,
    rank_bm25,
)
from project_progress.part_3.run_rankings import RESULTS_DIR


ABLATION_CSV = RESULTS_DIR / "ablation.csv"
EXTRA_LABELS_CSV = POOL_CSV.parent / "judgments_ablation.csv"
BM25_GRID_CSV = RESULTS_DIR / "bm25_grid.csv"
K = 10

K1_VALUES = (0.6, 0.9, 1.2, 1.5, 2.0)
B_VALUES = (0.25, 0.5, 0.75, 1.0)


def _sorted_pids(scores: pd.Series) -> list[str]:
    """pids by descending score, ties broken by pid (as in every ranker)."""
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def our_score_variants(engine: SearchEngine, query: str) -> dict[str, list[str]]:
    """Full rankings of our score with one component removed at a time.

    Args:
        engine: A built search engine.
        query: Raw query text.

    Returns:
        ``{variant name: ranked pids}``.
    """
    components = our_score_components(query, engine.index, engine.products)
    if components.empty:
        return {}
    variants = {"full": (OUR_SCORE_WEIGHTS, None)}
    for name in OUR_SCORE_WEIGHTS:
        variants[f"without_{name}"] = ({**OUR_SCORE_WEIGHTS, name: 0.0}, None)
    variants["without_stock"] = (OUR_SCORE_WEIGHTS, 1.0)
    rankings = {}
    for name, (weights, stock) in variants.items():
        kwargs = {} if stock is None else {"out_of_stock_factor": stock}
        rankings[name] = _sorted_pids(combine_our_score(components, weights, **kwargs))
    return rankings


def intent_variants(engine: SearchEngine, query: str) -> dict[str, list[str]]:
    """Full rankings of the intent score with one factor removed at a time.

    Args:
        engine: A built search engine.
        query: Raw query text.

    Returns:
        ``{variant name: ranked pids}``.
    """
    table = engine.intent.components(query)
    if table.empty:
        return {}
    factors = ["T", "K", "G", "A", "S"]
    rankings = {}
    for removed in (None, "K", "G", "A"):
        kept = [f for f in factors if f != removed]
        score = table[kept].prod(axis=1)
        rankings["full" if removed is None else f"without_{removed}"] = _sorted_pids(score)
    return rankings


def evaluate_ranking(ranking: list[str], labels: dict[str, int]) -> dict:
    """P@K, AP@K and NDCG@K of one ranking; unjudged products count as 0.

    Args:
        ranking: Ranked pids.
        labels: ``{pid: 0/1}`` for the judged products of this query.

    Returns:
        Metrics at ``K``, ``judged@K``, and P@K / AP@K on the condensed
        list (only judged products, in the same order).
    """
    top = ranking[:K]
    rel = [labels.get(pid, 0) for pid in top]
    condensed = [labels[pid] for pid in ranking if pid in labels][:K]
    total = sum(labels.values())
    return {
        f"P@{K}": precision_at_k(rel, K),
        f"AP@{K}": average_precision_at_k(rel, K, total),
        f"NDCG@{K}": ndcg_at_k(rel, K, total),
        f"judged@{K}": sum(pid in labels for pid in top),
        f"P@{K}_condensed": precision_at_k(condensed, K),
        f"AP@{K}_condensed": average_precision_at_k(condensed, K, total),
    }


def _read_labels(path) -> dict[str, dict[str, int]]:
    """``{query_id: {pid: 0/1}}`` from a judgment file (unlabelled rows skipped)."""
    table = pd.read_csv(path, encoding="utf-8-sig",
                        dtype={"pid": str, "query_id": str})
    table = table.dropna(subset=["relevance"])
    return {qid: dict(zip(g["pid"], g["relevance"].astype(int)))
            for qid, g in table.groupby("query_id")}


def label_sets() -> list[tuple[str, dict[str, str], dict]]:
    """The two label sets: (name, {query_id: query}, {query_id: {pid: 0/1}}).

    The ``pool`` set is the method pool plus the extra ablation judgments
    (``judgments_ablation.csv``, if present); a pool label always wins.
    """
    sets = [("validation", VALIDATION_QUERIES, load_validation_labels())]
    if POOL_CSV.exists():
        labels = _read_labels(POOL_CSV)
        if EXTRA_LABELS_CSV.exists():
            for qid, extra in _read_labels(EXTRA_LABELS_CSV).items():
                labels[qid] = {**extra, **labels.get(qid, {})}
        sets.append(("pool", ALL_QUERIES, labels))
    return sets


def all_variant_rankings(engine: SearchEngine, query: str) -> dict[str, list[str]]:
    """Every ranking the ablation evaluates for one query, by a readable name."""
    rankings = {}
    for method, variants in (("our_score", our_score_variants(engine, query)),
                             ("intent_score", intent_variants(engine, query))):
        for variant, ranking in variants.items():
            rankings[f"{method}:{variant}"] = ranking
    for k1 in K1_VALUES:
        for b in B_VALUES:
            rankings[f"bm25:k1={k1},b={b}"] = [
                pid for pid, _ in rank_bm25(query, engine.index, k1=k1, b=b)]
    return rankings


def export_unjudged(engine: SearchEngine) -> pd.DataFrame:
    """Write the top-K products of any variant that have no pool label.

    Existing rows of ``judgments_ablation.csv`` (and their labels) are kept,
    so the file can be re-exported safely after labelling.

    Returns:
        The table written, one row per (query_id, pid).
    """
    pool = _read_labels(POOL_CSV)
    old = (pd.read_csv(EXTRA_LABELS_CSV, encoding="utf-8-sig",
                       dtype={"pid": str, "query_id": str})
           if EXTRA_LABELS_CSV.exists() else pd.DataFrame())
    kept = {(r.query_id, r.pid): r.relevance for r in old.itertuples()} if not old.empty else {}
    rows = {}
    for query_id, query in ALL_QUERIES.items():
        judged = pool.get(query_id, {})
        for name, ranking in all_variant_rankings(engine, query).items():
            for pid in ranking[:K]:
                if pid in judged:
                    continue
                key = (query_id, pid)
                if key not in rows:
                    product = engine.products.loc[pid]
                    rows[key] = {
                        "query_id": query_id, "query": query, "pid": pid,
                        "relevance": kept.get(key), "surfaced_by": [],
                        "title": product["title"],
                        "product_details": product["product_details"],
                        "description": product["description"],
                    }
                rows[key]["surfaced_by"].append(name)
    table = pd.DataFrame(rows.values())
    if not table.empty:
        table["surfaced_by"] = table["surfaced_by"].map(";".join)
        table = table.sort_values(["query_id", "pid"])
    table.to_csv(EXTRA_LABELS_CSV, index=False, encoding="utf-8-sig")
    return table


def main():
    """Run the ablations and the BM25 grid; write two CSV files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export-unjudged", action="store_true",
                        help="write judgments_ablation.csv and stop")
    args = parser.parse_args()

    engine = SearchEngine(use_embeddings=False)
    if args.export_unjudged:
        table = export_unjudged(engine)
        missing = table["relevance"].isna().sum() if not table.empty else 0
        print(f"{len(table)} products outside the pool; {missing} still need a label "
              f"-> {EXTRA_LABELS_CSV.name}")
        return
    RESULTS_DIR.mkdir(exist_ok=True)

    ablation_rows, grid_rows = [], []
    for set_name, queries, labels in label_sets():
        for query_id, query in queries.items():
            if query_id not in labels:
                continue
            for method, variants in (("our_score", our_score_variants(engine, query)),
                                     ("intent_score", intent_variants(engine, query))):
                for variant, ranking in variants.items():
                    ablation_rows.append({
                        "label_set": set_name, "query_id": query_id,
                        "method": method, "variant": variant,
                        **evaluate_ranking(ranking, labels[query_id]),
                    })
            for k1 in K1_VALUES:
                for b in B_VALUES:
                    ranking = [pid for pid, _ in rank_bm25(query, engine.index,
                                                           k1=k1, b=b)]
                    grid_rows.append({
                        "label_set": set_name, "query_id": query_id,
                        "k1": k1, "b": b,
                        **evaluate_ranking(ranking, labels[query_id]),
                    })

    ablation = pd.DataFrame(ablation_rows)
    grid = pd.DataFrame(grid_rows)
    ablation.to_csv(ABLATION_CSV, index=False, float_format="%.3f")
    grid.to_csv(BM25_GRID_CSV, index=False, float_format="%.3f")

    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    print(f"Ablation, mean over queries (K = {K}):")
    print(ablation.groupby(["label_set", "method", "variant"])
          .mean(numeric_only=True).to_string(float_format=fmt))
    print(f"\nBM25 grid, mean AP@{K} / mean judged@{K} (rows = k1, columns = b):")
    for set_name, group in grid.groupby("label_set"):
        for value in (f"AP@{K}", f"judged@{K}", f"AP@{K}_condensed"):
            print(f"\n[{set_name}] {value}")
            print(group.pivot_table(index="k1", columns="b", values=value)
                  .to_string(float_format=fmt))
    print(f"\nSaved {ABLATION_CSV.name} and {BM25_GRID_CSV.name}")


if __name__ == "__main__":
    main()
