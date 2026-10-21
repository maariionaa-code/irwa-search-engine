"""
Text preprocessing utilities for IRWA Project - Part 1.

This module contains the reusable preprocessing functions used to transform
the raw fashion-product corpus into a representation suitable for indexing
and retrieval in later project stages.
"""

import html
import re
import unicodedata
from typing import Any

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


# English stop words provided by NLTK.
STOP_WORDS = set(stopwords.words("english"))

# Keep sequences of letters/numbers.
# Hyphenated terms are deliberately split:
# "full-sleeve" -> ["full", "sleeve"]
# because users may search "full sleeve".
TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+")

# English Snowball stemmer.
STEMMER = SnowballStemmer("english")

def normalize_text(text: Any) -> str:
    """
    Normalize raw text before tokenization.

    Steps:
    1. Handle missing values.
    2. Convert to string.
    3. Decode HTML entities.
    4. Normalize Unicode characters.
    5. Convert to lowercase.
    6. Normalize apostrophes and possessives.
    7. Replace hyphens/slashes with spaces.
    8. Collapse repeated whitespace.

    Parameters
    ----------
    text : Any
        Raw text value.

    Returns
    -------
    str
        Normalized text.
    """
    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(text)

    # Convert things such as "&amp;" to "&".
    text = html.unescape(text)

    # Standard Unicode representation.
    text = unicodedata.normalize("NFKC", text)

    # Lowercase so "Cotton" and "cotton" are treated equally.
    text = text.lower()

    # Normalize curly apostrophes.
    text = text.replace("’", "'")

    # Remove possessive endings:
    # "women's" -> "women"
    text = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", text)

    # Hyphenated product terms should be independently searchable.
    # "full-sleeve" -> "full sleeve"
    # "slim-fit" -> "slim fit"
    text = re.sub(r"[-_/]+", " ", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: Any) -> list[str]:
    """
    Normalize and tokenize text.

    Punctuation marks are excluded by the tokenizer.

    Parameters
    ----------
    text : Any
        Raw text.

    Returns
    -------
    list[str]
        Lowercase alphanumeric tokens.
    """
    normalized = normalize_text(text)

    if not normalized:
        return []

    return TOKENIZER.tokenize(normalized)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove common English stop words from a token list.

    Parameters
    ----------
    tokens : list[str]
        Tokenized text.

    Returns
    -------
    list[str]
        Tokens excluding stop words.
    """
    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Stem tokens using the English Snowball stemmer.

    Parameters
    ----------
    tokens : list[str]
        Tokens after stop-word removal.

    Returns
    -------
    list[str]
        Stemmed tokens.
    """
    return [STEMMER.stem(token) for token in tokens]


def preprocess_text(text: Any) -> list[str]:
    """
    Apply the complete preprocessing pipeline.

    Pipeline:
        raw text
        -> normalization
        -> tokenization / punctuation removal
        -> stop-word removal
        -> stemming

    Parameters
    ----------
    text : Any
        Raw text value.

    Returns
    -------
    list[str]
        Final tokens suitable for indexing.
    """
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)

    return tokens