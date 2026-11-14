"""Build every Part 3 ranker once and run them on a query.

Usage from the repository root (search with any method and query):
    python -m project_progress.part_3.pipeline "women cotton kurta" --method bm25 --top-k 10
"""

import argparse

from project_progress.part_3.corpus import build_search_index, load_products
from project_progress.part_3.embedded_ranking import (
    load_or_train_doc2vec,
    load_or_train_word2vec,
    rank_doc2vec,
    rank_word2vec,
)
from project_progress.part_3.intent_score import IntentScorer
from project_progress.part_3.ranking_methods import (
    rank_bm25,
    rank_our_score,
    rank_tfidf,
)


METHODS = ["tfidf", "bm25", "our_score", "intent_score", "word2vec", "doc2vec"]
LEXICAL_METHODS = METHODS[:4]  # methods that do not need gensim


class SearchEngine:
    """All rankers over one shared corpus and AND index.

    Args:
        use_embeddings: Load or train word2vec and Doc2Vec (needs gensim).
        retrain: Retrain the embedding models even if saved ones exist.
    """

    def __init__(self, use_embeddings: bool = True, retrain: bool = False):
        print("Loading and preprocessing the dataset...")
        self.products = load_products()
        self.index = build_search_index(self.products)
        print(f"{self.index.n_docs:,} products, "
              f"{len(self.index.inverted_index):,} terms, "
              f"average length {self.index.avg_doc_len:.1f} tokens")

        # Second original score: precomputes product attributes and genders.
        self.intent = IntentScorer(self.index, self.products)

        self.word_vectors = None
        self.doc2vec = None
        self._doc_vectors = {}
        if use_embeddings:
            print("Loading or training word2vec and Doc2Vec...")
            self.word_vectors = load_or_train_word2vec(
                self.index.documents, retrain
            )
            self.doc2vec = load_or_train_doc2vec(self.index.documents, retrain)

    def methods(self) -> list[str]:
        """Return the names of the rankers that are available."""
        return METHODS if self.word_vectors is not None else LEXICAL_METHODS

    def rank(self, query: str, method: str, top_k: int | None = None):
        """Rank the AND results of ``query`` with one method.

        Args:
            query: Raw query text.
            method: One of ``METHODS``.
            top_k: Number of results to return (``None`` = all candidates).

        Returns:
            List of ``(pid, score)`` sorted by descending score.
        """
        if method == "tfidf":
            return rank_tfidf(query, self.index, top_k)
        if method == "bm25":
            return rank_bm25(query, self.index, top_k)
        if method == "our_score":
            return rank_our_score(query, self.index, self.products, top_k)
        if method == "intent_score":
            return self.intent.rank(query, top_k)
        if method == "word2vec":
            return rank_word2vec(query, self.index, self.word_vectors, top_k,
                                 self._doc_vectors)
        if method == "doc2vec":
            return rank_doc2vec(query, self.index, self.doc2vec, top_k)
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")


def main():
    """Command-line search: print the top results of one method."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="query text, e.g. 'men slim fit jeans'")
    parser.add_argument("--method", default="bm25", choices=METHODS)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    engine = SearchEngine(use_embeddings=args.method in ("word2vec", "doc2vec"))
    results = engine.rank(args.query, args.method, args.top_k)
    n_candidates = len(engine.index.candidates(args.query))

    print(f"\nQuery: {args.query!r} | method: {args.method} | "
          f"AND candidates: {n_candidates}")
    for rank, (pid, score) in enumerate(results, start=1):
        product = engine.products.loc[pid]
        print(f"{rank:>2}. {score:.4f}  {pid}  {product['title']}")


if __name__ == "__main__":
    main()