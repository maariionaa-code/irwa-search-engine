"""Expose the Part 3 ranker and product data to the web application."""

from urllib.parse import urlsplit

import pandas as pd

from project_progress.part_1.preprocessing import parse_numeric_value
from project_progress.part_3.pipeline import SearchEngine as RankingEngine


def _web_url(value: object) -> str | None:
    """Return an absolute HTTP(S) URL, if available."""
    if not isinstance(value, str):
        return None
    url = value.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    return url if parts.scheme in ("http", "https") and parts.hostname else None


def _display_product(row: pd.Series) -> dict:
    """Select the original catalogue fields and normalized prices and ratings."""
    product = {
        field: row[field] if pd.notna(row[field]) else None
        for field in (
            "pid", "title", "description", "brand", "category",
            "sub_category", "seller", "out_of_stock",
        )
    }
    for field, column in (
        ("selling_price", "price"),
        ("discount", "discount_pct"),
        ("average_rating", "rating"),
    ):
        product[field] = float(row[column]) if pd.notna(row[column]) else None

    product["actual_price"] = parse_numeric_value(row["actual_price"])
    product["url"] = _web_url(row["url"])

    details = row["product_details"]
    if isinstance(details, list):
        details = {
            label: value
            for entry in details if isinstance(entry, dict)
            for label, value in entry.items()
        }
    product["product_details"] = details if isinstance(details, dict) else {}

    images = row["images"]
    product["images"] = (
        [url for image in images if (url := _web_url(image))]
        if isinstance(images, list) else []
    )
    return product


class SearchEngine:
    """Build the Part 3 index once and reuse it for catalogue searches."""

    def __init__(self):
        self.ranker = RankingEngine(use_embeddings=False)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return up to top_k products in descending BM25 order."""
        ranked_products = self.ranker.rank(query, "bm25", top_k=top_k)
        return [_display_product(self.ranker.products.loc[pid]) for pid, _ in ranked_products]

    def get_product(self, pid: str) -> dict | None:
        """Return a product's display fields, or None for an unknown ID."""
        products = self.ranker.products
        if pid not in products.index:
            return None
        return _display_product(products.loc[pid])
