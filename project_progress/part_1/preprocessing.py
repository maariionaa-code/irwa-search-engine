"""
Text preprocessing utilities for IRWA Project: Part 1.
"""

import html
import re
import unicodedata
from typing import Any

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


# NLP configuration

STOP_WORDS = set(stopwords.words("english"))


TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+")


STEMMER = SnowballStemmer("english")


# Text preprocessing

def normalize_text(text: Any) -> str:
    """
    Normalize raw text before tokenization.

    Processing steps:
    1. Handle missing values.
    2. Convert the value to a string.
    3. Decode HTML entities.
    4. Normalize Unicode characters.
    5. Convert text to lowercase.
    6. Normalize apostrophes and English possessives.
    7. Replace hyphens, underscores and slashes with spaces.
    8. Collapse repeated whitespace.
    """
    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", text)
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: Any) -> list[str]:
    """
    Normalize and tokenize a text value.

    Punctuation marks are excluded by the tokenizer.
    """
    normalized = normalize_text(text)

    if not normalized:
        return []

    return TOKENIZER.tokenize(normalized)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove common English stop words from a list of tokens.
    """
    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Stem tokens using the English Snowball stemmer.

    Stemming reduces related words to a shared representation.
    """
    return [STEMMER.stem(token) for token in tokens]


def preprocess_text(text: Any) -> list[str]:
    """
    Apply the complete textual preprocessing pipeline.

    Pipeline:
        raw text -> normalization -> tokenization / punctuation removal -> stop-word removal -> stemming

    This function should also be used later to preprocess user queries,
    ensuring that documents and queries use exactly the same representation.
    """
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)

    return tokens


# Structured product metadata

def product_details_to_text(details: Any) -> str:
    """
    Convert the product_details field into searchable text.

    Product details can contain useful retrieval information such as fabric,
    colour, pattern, fit or closure type.

    Both attribute names and values are retained.
    """
    if details is None:
        return ""

    try:
        if pd.isna(details):
            return ""
    except (TypeError, ValueError):
        pass

    parts = []

    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                for key, value in item.items():
                    if key is not None:
                        parts.append(str(key))

                    if value is not None:
                        parts.append(str(value))

    elif isinstance(details, dict):
        for key, value in details.items():
            if key is not None:
                parts.append(str(key))

            if value is not None:
                parts.append(str(value))

    else:
        parts.append(str(details))

    return " ".join(parts)


# Numerical fields

def parse_numeric_value(value: Any) -> float | None:
    """
    Convert a numeric-like value to float.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip().replace(",", "")

    if not value:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", value)

    if match is None:
        return None

    return float(match.group())


def parse_discount(value: Any) -> float | None:
    """
    Convert a discount value into a numerical percentage.
    """
    return parse_numeric_value(value)


# Combined retrieval representation

SEARCHABLE_TOKEN_FIELDS = [
    "title_tokens",
    "description_tokens",
    "brand_tokens",
    "category_tokens",
    "sub_category_tokens",
    "product_details_tokens",
]


def combine_search_tokens(row: pd.Series) -> list[str]:
    """
    Combine descriptive token fields into one searchable representation.

    Separate processed fields are preserved so that later ranking algorithms
    can assign different weights depending on where a query term appears.

    Seller tokens are intentionally excluded from the default combined search
    representation because seller names usually describe the merchant rather
    than the product itself.

    Token duplicates are deliberately preserved because term frequency will
    be useful for TF-IDF and BM25 ranking in later project stages.
    """
    combined = []

    for field in SEARCHABLE_TOKEN_FIELDS:
        tokens = row.get(field, [])

        if isinstance(tokens, list):
            combined.extend(tokens)

    return combined