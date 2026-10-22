"""
Tests for the IRWA Part 1 preprocessing pipeline.
"""

from preprocessing import (
    normalize_text,
    tokenize_text,
    remove_stopwords,
    stem_tokens,
    preprocess_text,
    product_details_to_text,
    parse_numeric_value,
    parse_discount,
)


# Text normalization

def test_normalize_lowercase():
    assert normalize_text("COTTON Shirt") == "cotton shirt"


def test_normalize_possessive():
    assert normalize_text("Women's Jeans") == "women jeans"


def test_normalize_hyphen():
    assert (
        normalize_text("Full-Sleeve Sweatshirt")
        == "full sleeve sweatshirt"
    )


def test_normalize_missing():
    assert normalize_text(None) == ""


def test_normalize_whitespace():
    assert normalize_text("blue   cotton\nshirt") == "blue cotton shirt"


# Tokenization

def test_tokenization():
    result = tokenize_text("Blue jeans, cotton shirt!")

    assert result == [
        "blue",
        "jeans",
        "cotton",
        "shirt",
    ]


def test_punctuation_removed():
    result = tokenize_text("shirt!!! jeans??? cotton...")

    assert result == [
        "shirt",
        "jeans",
        "cotton",
    ]


def test_hyphenated_words_split():
    result = tokenize_text("full-sleeve slim-fit")

    assert result == [
        "full",
        "sleeve",
        "slim",
        "fit",
    ]


def test_empty_tokenization():
    assert tokenize_text(None) == []


# Stop words

def test_stopword_removal():
    tokens = [
        "this",
        "is",
        "a",
        "blue",
        "cotton",
        "shirt",
    ]

    result = remove_stopwords(tokens)

    assert "this" not in result
    assert "is" not in result
    assert "a" not in result

    assert "blue" in result
    assert "cotton" in result
    assert "shirt" in result


# Stemming

def test_stemming():
    result = stem_tokens(
        [
            "shirts",
            "jeans",
            "sleeves",
        ]
    )

    assert result == [
        "shirt",
        "jean",
        "sleev",
    ]


# Complete pipeline

def test_complete_preprocessing():
    result = preprocess_text(
        "Women's Full-Sleeve Cotton Sweatshirts!"
    )

    assert "women" in result
    assert "full" in result
    assert "sleev" in result
    assert "cotton" in result
    assert "sweatshirt" in result


def test_complete_preprocessing_missing():
    assert preprocess_text(None) == []


# Product details

def test_product_details_list():
    details = [
        {"Fabric": "Cotton Blend"},
        {"Color": "Blue"},
        {"Pattern": "Solid"},
    ]

    result = product_details_to_text(details)

    assert "Fabric" in result
    assert "Cotton Blend" in result
    assert "Color" in result
    assert "Blue" in result
    assert "Pattern" in result
    assert "Solid" in result


def test_product_details_dictionary():
    details = {
        "Fabric": "Cotton",
        "Color": "Black",
    }

    result = product_details_to_text(details)

    assert "Fabric" in result
    assert "Cotton" in result
    assert "Color" in result
    assert "Black" in result


def test_product_details_missing():
    assert product_details_to_text(None) == ""


def test_product_details_can_be_preprocessed():
    details = [
        {"Fabric": "Cotton Blend"},
        {"Color": "Blue"},
    ]

    text = product_details_to_text(details)
    tokens = preprocess_text(text)

    assert "fabric" in tokens
    assert "cotton" in tokens
    assert "blend" in tokens
    assert "color" in tokens
    assert "blue" in tokens


# Numeric fields

def test_parse_price_with_comma():
    assert parse_numeric_value("2,999") == 2999.0


def test_parse_numeric_string():
    assert parse_numeric_value("921") == 921.0


def test_parse_integer():
    assert parse_numeric_value(921) == 921.0


def test_parse_rating():
    assert parse_numeric_value("3.9") == 3.9


def test_parse_discount():
    assert parse_discount("69% off") == 69.0


def test_parse_missing_numeric():
    assert parse_numeric_value(None) is None


def test_parse_empty_numeric():
    assert parse_numeric_value("") is None