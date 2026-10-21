from preprocessing import normalize_text


def test_normalize_lowercase():
    assert normalize_text("COTTON Shirt") == "cotton shirt"


def test_normalize_possessive():
    assert normalize_text("Women's Jeans") == "women jeans"


def test_normalize_hyphen():
    assert normalize_text("Full-Sleeve Sweatshirt") == "full sleeve sweatshirt"


def test_normalize_missing():
    assert normalize_text(None) == ""


from preprocessing import normalize_text, tokenize_text


def test_tokenization():
    result = tokenize_text("Blue jeans, cotton shirt!")
    assert result == ["blue", "jeans", "cotton", "shirt"]


def test_punctuation_removed():
    result = tokenize_text("shirt!!! jeans??? cotton...")
    assert result == ["shirt", "jeans", "cotton"]


def test_hyphenated_words_split():
    result = tokenize_text("full-sleeve slim-fit")
    assert result == ["full", "sleeve", "slim", "fit"]


from preprocessing import (
    normalize_text,
    tokenize_text,
    remove_stopwords,
)

def test_stopword_removal():
    tokens = ["this", "is", "a", "blue", "cotton", "shirt"]

    result = remove_stopwords(tokens)

    assert "this" not in result
    assert "is" not in result
    assert "a" not in result
    assert "blue" in result
    assert "cotton" in result
    assert "shirt" in result


from preprocessing import stem_tokens

def test_stemming():
    result = stem_tokens(["shirts", "jeans", "sleeves"])

    assert result == ["shirt", "jean", "sleev"]


from preprocessing import preprocess_text


def test_complete_preprocessing():
    result = preprocess_text(
        "Women's Full-Sleeve Cotton Sweatshirts!"
    )

    assert "women" in result
    assert "full" in result
    assert "sleev" in result
    assert "cotton" in result
    assert "sweatshirt" in result