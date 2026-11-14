"""Sanity checks for the trained word2vec model (Part 3 extras).

Before trusting a ranking built on embeddings we check that the vectors
were actually learned:

1. Nearest neighbours of every query term. A healthy model returns related
   fashion words (e.g. "jean" -> "denim", "trouser"); a broken one returns
   unrelated or random tokens.
2. Related vs random pairs. The average cosine of word pairs that should be
   related must be clearly higher than that of random word pairs.
3. Degenerate vectors. No vector should be (almost) all zeros.

These checks matter on Apple-silicon Macs, where gensim may print
"Exception ignored in: ... our_dot_float" during training.

Usage from the repository root (after run_rankings has trained the model):
    python -m project_progress.part_3.inspect_embeddings
"""

import numpy as np
import pandas as pd

from project_progress.part_1.preprocessing import preprocess_text
from project_progress.part_3.embedded_ranking import MODELS_DIR
from project_progress.part_3.queries import ALL_QUERIES
from project_progress.part_3.run_rankings import RESULTS_DIR


NEIGHBOURS_CSV = RESULTS_DIR / "word2vec_neighbours.csv"
RELATED_PAIRS = [("jean", "denim"), ("shoe", "sneaker"), ("kurta", "kurti"),
                 ("jacket", "coat"), ("short", "trunk"), ("cotton", "linen"),
                 ("black", "white"), ("sweatshirt", "hoodi"), ("sport", "gym")]


def main():
    """Print and save the three sanity checks."""
    from gensim.models import Word2Vec

    wv = Word2Vec.load(str(MODELS_DIR / "word2vec.model")).wv
    print(f"Vocabulary: {len(wv.index_to_key):,} words, {wv.vector_size} dimensions")

    rows = []
    for query in ALL_QUERIES.values():
        for term in dict.fromkeys(preprocess_text(query)):
            if term not in wv:
                continue
            for neighbour, similarity in wv.most_similar(term, topn=5):
                rows.append({"term": term, "neighbour": neighbour,
                             "cosine": round(float(similarity), 3)})
    neighbours = pd.DataFrame(rows).drop_duplicates()
    neighbours.to_csv(NEIGHBOURS_CSV, index=False)
    print("\n1. Nearest neighbours of the query terms:")
    for term, group in neighbours.groupby("term", sort=False):
        print(f"  {term:<11} -> " + ", ".join(
            f"{n} ({c:.2f})" for n, c in zip(group["neighbour"], group["cosine"])))

    related = [wv.similarity(a, b) for a, b in RELATED_PAIRS if a in wv and b in wv]
    rng = np.random.default_rng(42)
    vocab = wv.index_to_key[:5000]
    random_pairs = [wv.similarity(*rng.choice(vocab, 2, replace=False))
                    for _ in range(1000)]
    print(f"\n2. Mean cosine of related pairs: {np.mean(related):.3f} "
          f"({len(related)} pairs found) vs random pairs: {np.mean(random_pairs):.3f}")

    norms = np.linalg.norm(wv.vectors, axis=1)
    print(f"\n3. Vector norms: min {norms.min():.3f}, median {np.median(norms):.3f}; "
          f"near-zero vectors: {(norms < 1e-3).sum()}")

    healthy = np.mean(related) > np.mean(random_pairs) + 0.1 and (norms < 1e-3).sum() == 0
    print("\nVerdict:", "the model looks healthy." if healthy else
          "WARNING: the vectors look poorly trained; retrain with "
          "`python -m project_progress.part_3.run_rankings --retrain`.")
    print(f"Saved {NEIGHBOURS_CSV.name}")


if __name__ == "__main__":
    main()
