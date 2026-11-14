"""Explain our two original scores: every component for every top-20 product.

For each query, the top 20 of ``our_score`` and of ``intent_score`` are
written with all their components, plus the product's BM25 rank. This lets
the report show, with concrete products, WHY a product moved up or down
compared with plain BM25 (e.g. "out of stock, so S = 0.5" or "Fit: Jogger
contradicts 'slim', so A = 0.75").

Usage from the repository root:
    python -m project_progress.part_3.explain_scores
"""

import pandas as pd

from project_progress.part_3.pipeline import SearchEngine
from project_progress.part_3.queries import ALL_QUERIES, TOP_K
from project_progress.part_3.ranking_methods import (
    combine_our_score,
    our_score_components,
    rank_bm25,
)
from project_progress.part_3.run_rankings import RESULTS_DIR


COMPONENTS_CSV = RESULTS_DIR / "score_components.csv"


def _top(table: pd.DataFrame) -> pd.DataFrame:
    """Top ``TOP_K`` rows by descending score, ties broken by pid (as in rank_*)."""
    order = sorted(table.index, key=lambda pid: (-table.loc[pid, "score"], pid))
    return table.loc[order[:TOP_K]]


def explain_query(engine: SearchEngine, query_id: str, query: str) -> pd.DataFrame:
    """Components of both original scores for one query's top 20.

    Args:
        engine: A built search engine.
        query_id: Query identifier.
        query: Raw query text.

    Returns:
        One row per (method, product) in either top 20.
    """
    bm25_rank = {pid: r for r, (pid, _) in enumerate(rank_bm25(query, engine.index), 1)}
    frames = []

    ours = our_score_components(query, engine.index, engine.products)
    if not ours.empty:
        ours["score"] = combine_our_score(ours)
        ours = _top(ours)
        ours.insert(0, "method", "our_score")
        frames.append(ours.rename(columns={"title": "C_title", "text": "T_text",
                                           "rating": "R_rating",
                                           "discount": "D_discount"}))

    intent = engine.intent.components(query)
    if not intent.empty:
        intent = _top(intent)
        intent.insert(0, "method", "intent_score")
        frames.append(intent)

    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames)
    table.index.name = "pid"
    table = table.reset_index()
    table.insert(0, "query_id", query_id)
    table.insert(2, "rank", table.groupby("method").cumcount() + 1)
    table["bm25_rank"] = table["pid"].map(bm25_rank)
    table["title_text"] = table["pid"].map(engine.products["title"])
    return table


def main():
    """Write the component table for every query."""
    engine = SearchEngine(use_embeddings=False)
    RESULTS_DIR.mkdir(exist_ok=True)
    table = pd.concat([explain_query(engine, qid, q) for qid, q in ALL_QUERIES.items()],
                      ignore_index=True)
    table.to_csv(COMPONENTS_CSV, index=False, float_format="%.4f", encoding="utf-8-sig")

    moved = table.assign(change=table["bm25_rank"] - table["rank"])
    print("Biggest moves relative to BM25 (positive = moved up):")
    print(moved.sort_values("change", ascending=False)
          [["query_id", "method", "rank", "bm25_rank", "title_text"]]
          .head(15).to_string(index=False))
    print(f"\nSaved {COMPONENTS_CSV.name}")


if __name__ == "__main__":
    main()
