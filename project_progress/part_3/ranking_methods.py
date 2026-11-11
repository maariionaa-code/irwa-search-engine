"""Part 3: three ways to rank the conjunctive (AND) results.

1. ``rank_tfidf``  - TF-IDF + cosine similarity (identical to Part 2).
2. ``rank_bm25``   - Okapi BM25.
3. ``rank_our_score`` - our own score: BM25 text relevance + where the
   query terms appear (title) + product quality signals taken from the
   numerical fields (rating, discount, stock).

All three functions take the same arguments and return the same format,
a list of ``(pid, score)`` sorted by descending score, ties broken by pid,
so their rankings can be compared directly.
"""

import math
from collections import Counter

import pandas as pd

from project_progress.part_1.preprocessing import preprocess_text
from project_progress.part_3.corpus import SearchIndex


# BM25 parameters: the usual defaults (Robertson & Zaragoza; Lucene).
BM25_K1 = 1.2
BM25_B = 0.75

# Weights of our own score (they sum to 1, so the score is in [0, 1]).
OUR_SCORE_WEIGHTS = {
    "text": 0.55,      # normalised BM25: does the product match the words?
    "title": 0.25,     # share of query terms found in the title
    "rating": 0.15,    # customer rating, rescaled from 1-5 to 0-1
    "discount": 0.05,  # discount percentage, rescaled to 0-1
}
OUT_OF_STOCK_FACTOR = 0.5  # demote, but keep, products that cannot be bought


def _sort(results: list[tuple[str, float]], top_k: int | None):
    """Sort by descending score, then pid, and cut to ``top_k``."""
    results.sort(key=lambda item: (-item[1], item[0]))
    return results[:top_k] if top_k is not None else results


# ---------------------------------------------------------------------------
# 1. TF-IDF + cosine similarity
# ---------------------------------------------------------------------------

def tfidf_idf(index: SearchIndex, term: str) -> float:
    """Smoothed IDF used in Part 2: ``ln((N + 1) / (df + 1)) + 1``.

    Args:
        index: The shared search index.
        term: A processed term.

    Returns:
        The IDF weight (always >= 1 for terms in the vocabulary).
    """
    df = index.doc_freq.get(term, 0)
    return math.log((index.n_docs + 1) / (df + 1)) + 1


def rank_tfidf(query: str, index: SearchIndex, top_k: int | None = None):
    """Rank the AND results by TF-IDF cosine similarity.

    Weights are ``(1 + ln tf) * idf`` for both the query and the document,
    and the score is the cosine of the angle between the two vectors.

    Args:
        query: Raw query text.
        index: The shared search index.
        top_k: Number of results to return (``None`` = all).

    Returns:
        List of ``(pid, score)`` with scores in [0, 1].
    """
    candidates = index.candidates(query)
    if not candidates:
        return []

    query_tf = Counter(preprocess_text(query))
    query_weights = {
        term: (1 + math.log(freq)) * tfidf_idf(index, term)
        for term, freq in query_tf.items()
    }
    query_norm = math.sqrt(sum(w * w for w in query_weights.values()))

    results = []
    for pid in candidates:
        doc_weights = {
            term: (1 + math.log(freq)) * tfidf_idf(index, term)
            for term, freq in index.term_freqs[pid].items()
        }
        doc_norm = math.sqrt(sum(w * w for w in doc_weights.values()))
        dot = sum(
            weight * doc_weights.get(term, 0.0)
            for term, weight in query_weights.items()
        )
        score = dot / (query_norm * doc_norm) if query_norm and doc_norm else 0.0
        results.append((pid, score))

    return _sort(results, top_k)


# ---------------------------------------------------------------------------
# 2. BM25
# ---------------------------------------------------------------------------

def bm25_idf(index: SearchIndex, term: str) -> float:
    """BM25 IDF (Lucene variant): ``ln(1 + (N - df + 0.5) / (df + 0.5))``.

    The ``1 +`` inside the log keeps the value positive even for terms that
    appear in more than half of the documents (e.g. "cotton").

    Args:
        index: The shared search index.
        term: A processed term.

    Returns:
        The BM25 IDF weight (> 0).
    """
    df = index.doc_freq.get(term, 0)
    return math.log(1 + (index.n_docs - df + 0.5) / (df + 0.5))


def bm25_term_score(
    tf: int, doc_len: int, avg_doc_len: float, idf: float,
    k1: float = BM25_K1, b: float = BM25_B,
) -> float:
    """Contribution of one query term to the BM25 score of one document.

    Args:
        tf: Frequency of the term in the document.
        doc_len: Length of the document (tokens).
        avg_doc_len: Average document length in the collection.
        idf: BM25 IDF of the term.
        k1: Term-frequency saturation parameter.
        b: Length-normalisation parameter (0 = none, 1 = full).

    Returns:
        ``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))``
    """
    if tf == 0:
        return 0.0
    length_norm = 1 - b + b * doc_len / avg_doc_len
    return idf * tf * (k1 + 1) / (tf + k1 * length_norm)


def bm25_scores(
    query: str, index: SearchIndex, k1: float = BM25_K1, b: float = BM25_B
) -> dict[str, float]:
    """BM25 score of every AND candidate (unsorted).

    Each distinct query term is counted once (queries are short keyword
    lists, so repeating a word in the query should not double its weight).

    Args:
        query: Raw query text.
        index: The shared search index.
        k1: Term-frequency saturation parameter.
        b: Length-normalisation parameter.

    Returns:
        ``{pid: score}`` for the AND candidates (empty dict if none).
    """
    candidates = index.candidates(query)
    terms = set(preprocess_text(query))
    idfs = {term: bm25_idf(index, term) for term in terms}

    return {
        pid: sum(
            bm25_term_score(
                index.term_freqs[pid][term], index.doc_len[pid],
                index.avg_doc_len, idfs[term], k1, b,
            )
            for term in terms
        )
        for pid in candidates
    }


def rank_bm25(
    query: str, index: SearchIndex, top_k: int | None = None,
    k1: float = BM25_K1, b: float = BM25_B,
):
    """Rank the AND results by Okapi BM25.

    Args:
        query: Raw query text.
        index: The shared search index.
        top_k: Number of results to return (``None`` = all).
        k1: Term-frequency saturation parameter.
        b: Length-normalisation parameter.

    Returns:
        List of ``(pid, score)``; scores are unbounded and positive.
    """
    scores = bm25_scores(query, index, k1, b)
    return _sort(list(scores.items()), top_k)


# ---------------------------------------------------------------------------
# 3. Our own score
# ---------------------------------------------------------------------------

def rating_feature(rating: float | None, default: float) -> float:
    """Rescale a 1-5 rating to [0, 1]; missing ratings get ``default``.

    Args:
        rating: Average customer rating, or None/NaN when unknown.
        default: Value used when the rating is missing (we pass the
            rescaled corpus median, so a missing rating is neutral).

    Returns:
        A value in [0, 1].
    """
    if rating is None or pd.isna(rating) or rating <= 0:
        return default
    return min(max((rating - 1) / 4, 0.0), 1.0)


def discount_feature(discount_pct: float | None) -> float:
    """Rescale a discount percentage to [0, 1]; missing means no discount.

    Args:
        discount_pct: Discount in percent (e.g. 69.0), or None/NaN.

    Returns:
        A value in [0, 1].
    """
    if discount_pct is None or pd.isna(discount_pct):
        return 0.0
    return min(max(discount_pct / 100, 0.0), 1.0)


def title_coverage(query_terms: set[str], title_tokens: list[str]) -> float:
    """Share of the distinct query terms that appear in the product title.

    Args:
        query_terms: Distinct processed query terms.
        title_tokens: Processed title tokens of the product.

    Returns:
        A value in [0, 1] (1 = every query word is in the title).
    """
    if not query_terms:
        return 0.0
    return len(query_terms & set(title_tokens)) / len(query_terms)


def our_score_components(
    query: str, index: SearchIndex, products: pd.DataFrame
) -> pd.DataFrame:
    """Compute every component of our score for each AND candidate.

    Args:
        query: Raw query text.
        index: The shared search index.
        products: DataFrame from ``load_products`` (indexed by pid) with
            ``title_tokens``, ``rating``, ``discount_pct``, ``out_of_stock``.

    Returns:
        DataFrame indexed by pid with columns ``bm25``, ``text``, ``title``,
        ``rating``, ``discount``, ``in_stock`` (empty if no candidates).
    """
    bm25 = bm25_scores(query, index)
    if not bm25:
        return pd.DataFrame(
            columns=["bm25", "text", "title", "rating", "discount", "in_stock"]
        )

    terms = set(preprocess_text(query))
    max_bm25 = max(bm25.values()) or 1.0
    median_rating = rating_feature(products["rating"].median(), 0.5)

    rows = {}
    for pid, raw in bm25.items():
        product = products.loc[pid]
        rows[pid] = {
            "bm25": raw,
            "text": raw / max_bm25,
            "title": title_coverage(terms, product["title_tokens"]),
            "rating": rating_feature(product["rating"], median_rating),
            "discount": discount_feature(product["discount_pct"]),
            "in_stock": not bool(product["out_of_stock"]),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def combine_our_score(
    components: pd.DataFrame,
    weights: dict[str, float] = OUR_SCORE_WEIGHTS,
    out_of_stock_factor: float = OUT_OF_STOCK_FACTOR,
) -> pd.Series:
    """Weighted sum of the components, multiplied by the stock factor.

    ``score = stock * (w_text*text + w_title*title + w_rating*rating
    + w_discount*discount)`` where ``stock`` is 1 if the product is in
    stock and ``out_of_stock_factor`` otherwise.

    Args:
        components: Output of :func:`our_score_components`.
        weights: Weight of each component.
        out_of_stock_factor: Multiplier for out-of-stock products.

    Returns:
        Series ``{pid: score}``.
    """
    linear = sum(weight * components[name] for name, weight in weights.items())
    stock = components["in_stock"].map(lambda ok: 1.0 if ok else out_of_stock_factor)
    return linear * stock


def rank_our_score(
    query: str, index: SearchIndex, products: pd.DataFrame,
    top_k: int | None = None,
    weights: dict[str, float] = OUR_SCORE_WEIGHTS,
    out_of_stock_factor: float = OUT_OF_STOCK_FACTOR,
):
    """Rank the AND results with our own score.

    Args:
        query: Raw query text.
        index: The shared search index.
        products: DataFrame from ``load_products``.
        top_k: Number of results to return (``None`` = all).
        weights: Weight of each component (see ``OUR_SCORE_WEIGHTS``).
        out_of_stock_factor: Multiplier for out-of-stock products.

    Returns:
        List of ``(pid, score)`` with scores in [0, 1].
    """
    components = our_score_components(query, index, products)
    if components.empty:
        return []
    scores = combine_our_score(components, weights, out_of_stock_factor)
    return _sort([(pid, float(s)) for pid, s in scores.items()], top_k)