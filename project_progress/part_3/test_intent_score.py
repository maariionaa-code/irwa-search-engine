"""Tests for the intent-aware score (no dataset needed).

Run from the repository root:
    python -m pytest project_progress/part_3/test_intent_score.py -v
"""

import pandas as pd
import pytest

from project_progress.part_3.corpus import SearchIndex
from project_progress.part_3.intent_score import (
    IntentScorer,
    iter_attributes,
    product_genders,
)


def make_product(pid, title, sub_category, details, tokens, out_of_stock=False):
    """One toy product: raw fields plus its processed search tokens."""
    return {"pid": pid, "title": title, "sub_category": sub_category,
            "product_details": details, "out_of_stock": out_of_stock,
            "title_tokens": title.lower().split(), "tokens": tokens}


@pytest.fixture
def scorer(monkeypatch):
    """Five products that all contain "men slim jean"; MIN_SUPPORT lowered."""
    monkeypatch.setattr("project_progress.part_3.intent_score.MIN_SUPPORT", 1)
    rows = [
        # Correct product: men, slim fit, jeans, bottomwear.
        make_product("p1", "slim men blue jean", "Bottomwear",
                     [{"Fit": "Slim"}, {"Ideal For": "Men"},
                      {"Generic Name": "Jeans"}],
                     ["slim", "men", "blue", "jean"]),
        # Jogger fit: "slim" only in the description -> attribute contradiction.
        make_product("p2", "jogger men grey jean", "Bottomwear",
                     [{"Fit": "Jogger"}, {"Ideal For": "Men"},
                      {"Generic Name": "Jeans"}],
                     ["jogger", "men", "grey", "jean", "slim"]),
        # Wrong audience.
        make_product("p3", "slim woman blue jean", "Bottomwear",
                     [{"Fit": "Slim"}, {"Ideal For": "Women"},
                      {"Generic Name": "Jeans"}],
                     ["slim", "woman", "blue", "jean", "men"]),
        # Wrong category: a shirt that mentions jeans.
        make_product("p4", "slim fit men shirt", "Topwear",
                     [{"Fit": "Slim"}, {"Ideal For": "Men"},
                      {"Generic Name": "Shirt"}],
                     ["slim", "fit", "men", "shirt", "jean"]),
        # Correct product but out of stock.
        make_product("p5", "slim men dark jean", "Bottomwear",
                     [{"Fit": "Slim"}, {"Ideal For": "Men"},
                      {"Generic Name": "Jeans"}],
                     ["slim", "men", "dark", "jean"], out_of_stock=True),
    ]
    products = pd.DataFrame(rows).set_index("pid", drop=False)
    index = SearchIndex(dict(zip(products["pid"], products["tokens"])))
    return IntentScorer(index, products)


def test_iter_attributes_handles_list_dict_and_missing():
    assert list(iter_attributes([{"Fit": "Slim"}])) == [("fit", "Slim")]
    assert list(iter_attributes({"Color": "Blue"})) == [("color", "Blue")]
    assert list(iter_attributes(None)) == []


def test_product_genders_from_ideal_for_then_title():
    assert product_genders({"ideal for": frozenset({"men"})}, []) == {"men"}
    assert product_genders({}, ["slim", "woman", "jean"]) == {"woman"}
    assert product_genders({"ideal for": frozenset({"unisex"})}, []) == {"men", "woman"}
    assert product_genders({}, ["blue", "jean"]) == frozenset()


def test_query_intent_detects_gender_and_attributes(scorer):
    intent = scorer.query_intent("men slim jeans")
    assert intent["genders"] == {"men"}
    assert "slim" in intent["attribute_terms"]       # a value of Fit
    assert "jean" in intent["attribute_terms"]       # a value of Generic Name
    assert "men" not in intent["attribute_terms"]    # handled as gender


def test_correct_product_ranks_first(scorer):
    ranking = [pid for pid, _ in scorer.rank("men slim jeans")]
    assert ranking[0] == "p1"


def test_each_contradiction_is_penalised(scorer):
    table = scorer.components("men slim jeans")
    assert table.loc["p2", "attr_contradict"] == 1      # Fit: Jogger
    assert table.loc["p3", "gender"] == "mismatch"      # women's jeans
    assert table.loc["p3", "G"] == 0.3
    assert table.loc["p4", "K"] < 1.0                   # Topwear, not Bottomwear
    assert table.loc["p5", "S"] == 0.5                  # out of stock
    for pid in ("p2", "p3", "p4", "p5"):
        assert table.loc[pid, "score"] < table.loc["p1", "score"]


def test_score_is_product_of_factors_and_in_unit_interval(scorer):
    table = scorer.components("men slim jeans")
    product = table["T"] * table["K"] * table["G"] * table["A"] * table["S"]
    assert ((table["score"] - product).abs() < 1e-12).all()
    assert table["score"].between(0, 1).all()


def test_gender_neutral_query_does_not_penalise_gender(scorer):
    table = scorer.components("slim jeans")
    assert (table["G"] == 1.0).all()


def test_unknown_query_returns_nothing(scorer):
    assert scorer.rank("zzzunknown") == []