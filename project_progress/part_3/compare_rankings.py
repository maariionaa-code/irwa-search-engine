"""Measure how differently the methods order the same AND candidates. 

All methods rank exactly the same candidate set (the AND result), so the
differences are purely in the order. For every query and pair of methods:
    - ``overlap@10``  - share of the top 10 that both methods have in common;
    -  ``jaccard@20``  - |A n B| / |A u B| for the two top-20 sets;
    - ``kendall_tau`` - rank correlation over ALL candidates
      (1 = identical order, 0 = unrelated, -1 = reversed).

It also lists, for TF-IDF vs BM25, the products whose rank changes most,
with their length and query-term frequencies, to explain WHY they move.

Usage from the repository root:
    python -m project_progress.part_3.compare_rankings [--no-embeddings]
"""

import argparse
from itertools import combinations

import pandas as pd
from scipy.stats import kendalltau

from project_progress.part_1.preprocessing import preprocess_text
from project_progress.part_3.pipeline import SearchEngine
from project_progress.part_3.queries import ALL_QUERIES
from project_progress.part_3.run_rankings import RESULTS_DIR


AGREEMENT_CSV = RESULTS_DIR / "method_agreement.csv"
SHIFTS_CSV = RESULTS_DIR / "tfidf_vs_bm25_rank_shifts.csv"


def overlap_at_k(a: list[str], b: list[str], k: int) -> float:
    """Share of the top-k items that two rankings have in common.

    Args:
        a: First ranked list of pids.
        b: Second ranked list of pids.
        k: Cutoff.

    Returns:
        ``|top_k(a) n top_k(b)| / k`` (0 if k is 0).
    """
    return len(set(a[:k]) & set(b[:k])) / k if k else 0.0


def jaccard_at_k(a: list[str], b: list[str], k: int) -> float:
    """Jaccard similarity of the two top-k sets."""
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def kendall_tau_full(a: list[str], b: list[str]) -> float:
    """Kendall's tau between two orderings of the same candidates.

    Args:
        a: First full ranking (pids).
        b: Second full ranking of the same pids.

    Returns:
        Tau-b in [-1, 1]; 1.0 when there are fewer than two candidates.
    """
    if len(a) < 2:
        return 1.0
    position_b = {pid: i for i, pid in enumerate(b)}
    tau, _ = kendalltau(list(range(len(a))), [position_b[pid] for pid in a])
    return float(tau)


def agreement_table(full: dict) -> pd.DataFrame:
    """Pairwise agreement for every query.

    Args:
        full: ``{query_id: {method: full ranked list of pids}}``.

    Returns:
        DataFrame with one row per query and method pair.
    """
    rows = []
    for query_id, by_method in full.items():
        for m1, m2 in combinations(by_method, 2):
            a, b = by_method[m1], by_method[m2]
            rows.append({
                "query_id": query_id,
                "method_a": m1,
                "method_b": m2,
                "candidates": len(a),
                "overlap@10": overlap_at_k(a, b, 10),
                "jaccard@20": jaccard_at_k(a, b, 20),
                "kendall_tau": kendall_tau_full(a, b),
            })
    return pd.DataFrame(rows)


def rank_shifts(engine: SearchEngine, query_id: str, query: str,
                tfidf: list[str], bm25: list[str], n: int = 5) -> list[dict]:
    """Products in either top 20 whose position changes most.

    Args:
        engine: The built search engine.
        query_id: Query identifier.
        query: Query text.
        tfidf: Full TF-IDF ranking (pids).
        bm25: Full BM25 ranking (pids).
        n: How many products to report.

    Returns:
        Rows with both ranks, document length and query-term frequencies.
    """
    pos_t = {pid: i + 1 for i, pid in enumerate(tfidf)}
    pos_b = {pid: i + 1 for i, pid in enumerate(bm25)}
    terms = sorted(set(preprocess_text(query)))
    shown = set(tfidf[:20]) | set(bm25[:20])
    moves = sorted(shown, key=lambda pid: -abs(pos_t[pid] - pos_b[pid]))[:n]
    rows = []
    for pid in moves:
        tf = engine.index.term_freqs[pid]
        rows.append({
            "query_id": query_id,
            "pid": pid,
            "title": engine.products.loc[pid, "title"],
            "rank_tfidf": pos_t[pid],
            "rank_bm25": pos_b[pid],
            "doc_len": engine.index.doc_len[pid],
            "avg_doc_len": round(engine.index.avg_doc_len, 1),
            "query_tf": ", ".join(f"{t}={tf[t]}" for t in terms),
        })
    return rows


def main():
    """Compute the agreement table and the rank-shift examples."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    engine = SearchEngine(use_embeddings=not args.no_embeddings)
    RESULTS_DIR.mkdir(exist_ok=True)

    full, shifts = {}, []
    for query_id, query in ALL_QUERIES.items():
        full[query_id] = {
            method: [pid for pid, _ in engine.rank(query, method)]
            for method in engine.methods()
        }
        shifts += rank_shifts(engine, query_id, query,
                              full[query_id]["tfidf"], full[query_id]["bm25"])

    agreement = agreement_table(full)
    agreement.to_csv(AGREEMENT_CSV, index=False, float_format="%.3f")
    pd.DataFrame(shifts).to_csv(SHIFTS_CSV, index=False, encoding="utf-8-sig")

    mean = (agreement.groupby(["method_a", "method_b"])
            [["overlap@10", "jaccard@20", "kendall_tau"]].mean())
    print("Mean agreement across queries:")
    print(mean.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved {AGREEMENT_CSV.name} and {SHIFTS_CSV.name}")


if __name__ == "__main__":
    main()