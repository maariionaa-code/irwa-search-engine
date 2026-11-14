"""Measure how differently the methods order the same AND candidates. 

All methods rank exactly the same candidate set (the AND result), so the
differences are purely in the order. For every query and pair of methods:
    - ``overlap@10``  - share of the top 10 that both methods have in common;
    -  ``jaccard@20``  - |A n B| / |A u B| for the two top-20 sets;
    - ``kendall_tau`` - rank correlation over ALL candidates
      (1 = identical order, 0 = unrelated, -1 = reversed).

It also lists, for TF-IDF vs BM25, the products whose rank changes most,
with their length and query-term frequencies, to explain WHY they move.

Extra measures:
    - ``overlap@5`` / ``overlap@20`` - the same as overlap@10 at other depths;
    - ``rbo@20`` - rank-biased overlap (p = 0.9): like Kendall's tau it
      compares orders, but it gives most weight to the TOP positions. A high
      tau with a low RBO means: the methods agree on the bulk of the list
      but disagree on the first page;
    - ``score_spread.csv`` - how close the scores at the top of each list
      are. When many products have almost the same score (near-ties), small
      formula differences are enough to reshuffle the top 10.

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
from project_progress.part_3.rank_statistics import rank_biased_overlap
from project_progress.part_3.run_rankings import RESULTS_DIR


AGREEMENT_CSV = RESULTS_DIR / "method_agreement.csv"
SHIFTS_CSV = RESULTS_DIR / "tfidf_vs_bm25_rank_shifts.csv"
SPREAD_CSV = RESULTS_DIR / "score_spread.csv"


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
                "overlap@5": overlap_at_k(a, b, 5),
                "overlap@10": overlap_at_k(a, b, 10),
                "overlap@20": overlap_at_k(a, b, 20),
                "jaccard@20": jaccard_at_k(a, b, 20),
                "rbo@20": rank_biased_overlap(a, b, p=0.9, depth=20),
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


def score_spread(query_id: str, method: str, scored: list) -> dict:
    """How tightly packed the top scores of one ranking are.

    Args:
        query_id: Query identifier.
        method: Method name.
        scored: Full ranking as ``[(pid, score), ...]``.

    Returns:
        Scores at ranks 1, 10 and 20, the relative drop from rank 1 to 20,
        and how many candidates score within 1% of the rank-10 score
        (a large number = many near-ties around the first-page cut-off).
    """
    scores = [score for _, score in scored]

    def at(rank):
        return scores[rank - 1] if len(scores) >= rank else float("nan")

    s1, s10, s20 = at(1), at(10), at(20)
    near_ties = sum(abs(s - s10) <= 0.01 * abs(s10) for s in scores) if s10 == s10 else 0
    return {
        "query_id": query_id,
        "method": method,
        "candidates": len(scores),
        "score@1": s1,
        "score@10": s10,
        "score@20": s20,
        "relative_drop_1_to_20": (s1 - s20) / s1 if s1 else float("nan"),
        "near_ties_around_rank10": near_ties,
    }


def main():
    """Compute the agreement table and the rank-shift examples."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    engine = SearchEngine(use_embeddings=not args.no_embeddings)
    RESULTS_DIR.mkdir(exist_ok=True)

    full, shifts, spreads = {}, [], []
    for query_id, query in ALL_QUERIES.items():
        scored = {method: engine.rank(query, method) for method in engine.methods()}
        full[query_id] = {
            method: [pid for pid, _ in ranking]
            for method, ranking in scored.items()
        }
        spreads += [score_spread(query_id, method, ranking)
                    for method, ranking in scored.items()]
        shifts += rank_shifts(engine, query_id, query,
                              full[query_id]["tfidf"], full[query_id]["bm25"])

    agreement = agreement_table(full)
    agreement.to_csv(AGREEMENT_CSV, index=False, float_format="%.3f")
    pd.DataFrame(shifts).to_csv(SHIFTS_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(spreads).to_csv(SPREAD_CSV, index=False, float_format="%.4f")

    mean = (agreement.groupby(["method_a", "method_b"])
            [["overlap@10", "jaccard@20", "rbo@20", "kendall_tau"]].mean())
    print("Mean agreement across queries:")
    print(mean.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved {AGREEMENT_CSV.name}, {SHIFTS_CSV.name} and {SPREAD_CSV.name}")


if __name__ == "__main__":
    main()