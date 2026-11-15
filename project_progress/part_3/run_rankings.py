"""Rank every query with every method and save the top-20 lists.

Usage from the repository root:
    python -m project_progress.part_3.run_rankings            # all 6 methods
    python -m project_progress.part_3.run_rankings --no-embeddings

Output: ``results/rankings_top20.csv`` (one row per query x method x rank)
and ``results/candidate_counts.csv`` (size of the AND result per query).
"""

import argparse
from pathlib import Path

import pandas as pd

from project_progress.part_3.pipeline import SearchEngine
from project_progress.part_3.queries import ALL_QUERIES, TOP_K


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANKINGS_CSV = RESULTS_DIR / "rankings_top20.csv"
COUNTS_CSV = RESULTS_DIR / "candidate_counts.csv"


def check_same_candidates(query_id: str, method: str, ranking, candidates: set):
    """Every method must rank exactly the AND candidates, each once.

    Raises:
        ValueError: If a method drops, adds or repeats a product.
    """
    pids = [pid for pid, _ in ranking]
    if len(pids) != len(set(pids)) or set(pids) != candidates:
        raise ValueError(
            f"{method} on {query_id} ranks {len(set(pids))} distinct products "
            f"({len(pids)} rows) instead of the {len(candidates)} AND candidates"
        )


def run(engine: SearchEngine, queries: dict[str, str] = ALL_QUERIES,
        top_k: int = TOP_K) -> pd.DataFrame:
    """Rank each query with each available method.

    Args:
        engine: A built :class:`SearchEngine`.
        queries: ``{query_id: query text}``.
        top_k: Length of each ranked list.

    Before truncating to ``top_k``, each full ranking is checked against
    the AND candidate set (see :func:`check_same_candidates`).

    Returns:
        DataFrame with columns query_id, query, method, rank, pid, score,
        title, rating, discount_pct, out_of_stock, doc_len.
    """
    rows = []
    for query_id, query in queries.items():
        candidates = set(engine.index.candidates(query))
        for method in engine.methods():
            full = engine.rank(query, method)
            check_same_candidates(query_id, method, full, candidates)
            for rank, (pid, score) in enumerate(full[:top_k], start=1):
                product = engine.products.loc[pid]
                rows.append({
                    "query_id": query_id,
                    "query": query,
                    "method": method,
                    "rank": rank,
                    "pid": pid,
                    "score": round(score, 6),
                    "title": product["title"],
                    "rating": product["rating"],
                    "discount_pct": product["discount_pct"],
                    "out_of_stock": bool(product["out_of_stock"]),
                    "doc_len": engine.index.doc_len[pid],
                })
    return pd.DataFrame(rows)


def main():
    """Run all rankers and write the CSV files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-embeddings", action="store_true",
                        help="skip word2vec / Doc2Vec (no gensim needed)")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain the embedding models")
    args = parser.parse_args()

    engine = SearchEngine(use_embeddings=not args.no_embeddings,
                          retrain=args.retrain)
    RESULTS_DIR.mkdir(exist_ok=True)

    counts = pd.DataFrame([
        {"query_id": qid, "query": q,
         "and_candidates": len(engine.index.candidates(q))}
        for qid, q in ALL_QUERIES.items()
    ])
    counts.to_csv(COUNTS_CSV, index=False)
    print("\nAND candidates per query:")
    print(counts.to_string(index=False))

    rankings = run(engine)
    rankings.to_csv(RANKINGS_CSV, index=False, encoding="utf-8-sig")

    for (query_id, method), group in rankings.groupby(["query_id", "method"],
                                                      sort=False):
        print(f"\n{query_id} {group['query'].iloc[0]!r} - {method}")
        for _, row in group.head(5).iterrows():
            print(f"  {row['rank']:>2}. {row['score']:.4f}  {row['title'][:70]}")

    print(f"\nSaved {len(rankings)} rows to {RANKINGS_CSV}")


if __name__ == "__main__":
    main()