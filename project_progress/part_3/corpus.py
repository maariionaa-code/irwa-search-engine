"""Shared corpus loading and index statistics for IRWA Part 3.

Every ranking method in Part 3 works on the same candidate set: the
products returned by the conjunctive (AND) search of Part 2. This module
loads the dataset once, reuses the Part 1 preprocessing and the Part 2
inverted index, and precomputes the statistics that TF-IDF, BM25 and our
own score need (term frequencies, document lengths, document frequencies).
"""

from collections import Counter
from pathlib import Path
import pandas as pd

from project_progress.part_1.preprocessing import (combine_search_tokens, parse_discount, parse_numeric_value, preprocess_text, product_details_to_text,)
from project_progress.part_2.indexing import build_inverted_index, search_and


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "fashion_products_dataset.json"

TEXT_FIELDS = ["title", "description", "brand", "category", "sub_category"]


def load_products(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the dataset and add processed text and numeric columns.

    Args:
        path: Location of ``fashion_products_dataset.json``.

    Returns:
        DataFrame indexed by ``pid`` with, among others, the columns
        ``search_tokens`` (combined searchable tokens, brand and seller
        excluded, as in Part 2), ``title_tokens``, ``rating``,
        ``discount_pct``, ``price`` and ``out_of_stock``.
    """
    df = pd.read_json(path, dtype={"pid": str})

    for field in TEXT_FIELDS:
        df[f"{field}_tokens"] = df[field].apply(preprocess_text)

    df["product_details_tokens"] = df["product_details"].apply(lambda details: preprocess_text(product_details_to_text(details)))
    df["search_tokens"] = df.apply(combine_search_tokens, axis=1)

    # Numeric fields stay numeric (Part 1 decision); they feed our own score.
    df["rating"] = df["average_rating"].apply(parse_numeric_value)
    df["discount_pct"] = df["discount"].apply(parse_discount)
    df["price"] = df["selling_price"].apply(parse_numeric_value)
    df["out_of_stock"] = df["out_of_stock"].fillna(False).astype(bool)

    return df.set_index("pid", drop=False)


class SearchIndex:
    """Inverted index plus the term statistics shared by all rankers.

    Args:
        documents: Mapping ``{pid: list of processed tokens}``. Repeated
            tokens are kept because term frequency matters for ranking.

    Attributes:
        documents: The input mapping.
        n_docs: Number of documents (N).
        inverted_index: ``{term: sorted list of pids}`` (Part 2 structure).
        doc_freq: ``{term: number of documents containing it}`` (df).
        term_freqs: ``{pid: Counter(term -> count)}`` (tf).
        doc_len: ``{pid: number of tokens}``.
        avg_doc_len: Mean document length, used by BM25.
    """

    def __init__(self, documents: dict[str, list[str]]):
        self.documents = documents
        self.n_docs = len(documents)
        self.inverted_index = build_inverted_index(documents)
        self.doc_freq = {term: len(pids) for term, pids in self.inverted_index.items()}
        self.term_freqs = {pid: Counter(tokens) for pid, tokens in documents.items()}
        self.doc_len = {pid: len(tokens) for pid, tokens in documents.items()}
        self.avg_doc_len = (sum(self.doc_len.values()) / self.n_docs if self.n_docs else 0.0)

    def candidates(self, query: str) -> list[str]:
        """Return the products that contain every query term (AND).

        Args:
            query: Raw query text; it is preprocessed exactly like documents.

        Returns:
            Sorted list of matching pids (empty if any term is unknown).
        """
        return search_and(query, self.inverted_index)


def build_search_index(products: pd.DataFrame) -> SearchIndex:
    """Build a :class:`SearchIndex` from the output of :func:`load_products`.

    Args:
        products: DataFrame returned by :func:`load_products`.

    Returns:
        SearchIndex over the ``search_tokens`` column.
    """
    documents = dict(zip(products["pid"], products["search_tokens"]))
    return SearchIndex(documents)