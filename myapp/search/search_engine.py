"""Expose the Part 3 ranker and product data to the web application."""

import re
from urllib.parse import urlsplit

import pandas as pd

from project_progress.part_1.preprocessing import parse_numeric_value
from project_progress.part_3.corpus import check_top_k
from project_progress.part_3.pipeline import SearchEngine as RankingEngine


PRICE_EXPRESSION = re.compile(
    r"\b(?P<operator>under|below|over|above|at\s+most|at\s+least)\s+"
    r"₹?\s*(?P<amount>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<decimal>\d{1,2}))?(?!\w|[,.]\d)",
    re.IGNORECASE,
)


def split_price_constraints(query: str) -> tuple[str, list[tuple[str, float]]]:
    """Extract explicit price comparisons in the catalogue's price units."""
    constraints = []

    def collect(match: re.Match) -> str:
        amount = match["amount"].replace(",", "")
        if match["decimal"] is not None:
            amount += "." + match["decimal"]
        constraints.append((" ".join(match["operator"].lower().split()), float(amount)))
        return " "

    return " ".join(PRICE_EXPRESSION.sub(collect, query).split()), constraints


def _matches_price(price: float, constraints: list[tuple[str, float]]) -> bool:
    if pd.isna(price):
        return False
    for operator, value in constraints:
        if operator in {"under", "below"} and price >= value:
            return False
        if operator in {"over", "above"} and price <= value:
            return False
        if operator == "at most" and price > value:
            return False
        if operator == "at least" and price < value:
            return False
    return True


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

    def search(self, query: str, top_k: int = 20, *, method: str = "bm25") -> list[dict]:
        """Return up to top_k products in the selected lexical ranking order."""
        check_top_k(top_k)
        text, constraints = split_price_constraints(query)
        if not text:
            return []
        ranked_products = self.ranker.rank(text, method, top_k=None if constraints else top_k)
        if constraints:
            prices = self.ranker.products["price"]
            ranked_products = [
                (pid, score) for pid, score in ranked_products if _matches_price(prices.at[pid], constraints)
            ][:top_k]
        return [_display_product(self.ranker.products.loc[pid]) for pid, _ in ranked_products]

    def get_product(self, pid: str) -> dict | None:
        """Return a product's display fields, or None for an unknown ID."""
        products = self.ranker.products
        if pid not in products.index:
            return None
        return _display_product(products.loc[pid])
