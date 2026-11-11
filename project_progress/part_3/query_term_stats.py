"""Document frequency and IDF of every query term (justifies query choice).

Usage from the repository root:
    python -m project_progress.part_3.query_term_stats

Output: ``results/query_term_stats.csv``.
"""

import pandas as pd

from project_progress.part_1.preprocessing import preprocess_text
from project_progress.part_3.corpus import build_search_index, load_products
from project_progress.part_3.queries import ALL_QUERIES
from project_progress.part_3.ranking_methods import bm25_idf, tfidf_idf
from project_progress.part_3.run_rankings import RESULTS_DIR


def main():
    """Print df, share of corpus, TF-IDF IDF and BM25 IDF per query term."""
    index = build_search_index(load_products())
    rows = []
    for query_id, query in ALL_QUERIES.items():
        for term in preprocess_text(query):
            df = index.doc_freq.get(term, 0)
            rows.append({
                "query_id": query_id,
                "query": query,
                "term": term,
                "df": df,
                "share_of_corpus": df / index.n_docs,
                "idf_tfidf": tfidf_idf(index, term),
                "idf_bm25": bm25_idf(index, term),
                "and_candidates": len(index.candidates(query)),
            })
    stats = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    stats.to_csv(RESULTS_DIR / "query_term_stats.csv", index=False,
                 float_format="%.3f")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()