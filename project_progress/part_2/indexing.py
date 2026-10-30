from pathlib import Path
from collections import defaultdict
import pandas as pd

from project_progress.part_1.preprocessing import (
    preprocess_text,
    product_details_to_text,
    combine_search_tokens
)

# Location of the dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "fashion_products_dataset.json"


def load_processed_documents():
    """Load and preprocess all products using the Part 1 functions."""

    df = pd.read_json(DATASET_PATH)

    # Preprocess the textual fields
    fields = ["title", "description", "brand", "category", "sub_category"]

    for field in fields:
        df[f"{field}_tokens"] = df[field].apply(preprocess_text)

    # Preprocess product attributes
    df["product_details_tokens"] = df["product_details"].apply(
        lambda x: preprocess_text(product_details_to_text(x))
    )

    # Combine all searchable tokens
    df["search_tokens"] = df.apply(combine_search_tokens, axis=1)

    return dict(zip(df["pid"], df["search_tokens"]))


def build_inverted_index(documents):
    """Create a dictionary mapping each term to its product IDs."""

    index = defaultdict(set)

    for pid, tokens in documents.items():
        for term in tokens:
            index[term].add(pid)

    return {term: sorted(pids) for term, pids in index.items()}

def search_and(query, inverted_index):
    """Return products containing all query terms."""

    query_terms = preprocess_text(query)

    if not query_terms:
        return []

    postings_lists = [
        set(inverted_index.get(term, []))
        for term in query_terms
    ]

    results = set.intersection(*postings_lists)

    return sorted(results)

if __name__ == "__main__":

    documents = load_processed_documents()
    inverted_index = build_inverted_index(documents)

    print("Products indexed:", len(documents))
    print("Unique terms:", len(inverted_index))
    print("Products containing cotton:", len(inverted_index.get("cotton", [])))
    print("First 5 product IDs:", inverted_index.get("cotton", [])[:5])
    
    query = "blue jeans"
    results = search_and(query, inverted_index)

    print("\nQuery:", query)
    print("Matching products:", len(results))
    print("First 5 results:", results[:5])
    
    # Verify that every result contains all query terms
    query_terms = set(preprocess_text(query))

    assert all(
        query_terms.issubset(set(documents[pid]))
        for pid in results
    )

    # Verify that queries without matches return an empty list
    assert search_and("zznonexistentterm123", inverted_index) == []

    print("AND search validation passed!")