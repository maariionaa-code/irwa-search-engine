from unittest.mock import Mock

import pandas as pd
import pytest

from myapp.search.search_engine import SearchEngine, split_price_constraints


@pytest.mark.parametrize(("query", "expected"), [
    ("men running shoes under 1000", ("men running shoes", [("under", 1000.0)])),
    ("black boots over 500 at most 1,199.50", ("black boots", [("over", 500.0), ("at most", 1199.50)])),
    ("hooded sweatshirt", ("hooded sweatshirt", [])),
])
def test_price_constraints_leave_the_text_query_intact(query, expected):
    assert split_price_constraints(query) == expected


def test_price_filter_runs_before_top_k(monkeypatch):
    search = SearchEngine.__new__(SearchEngine)
    search.ranker = Mock()
    search.ranker.products = pd.DataFrame(
        {"price": [1199, 1000, None, 899, 500, 600]},
        index=["expensive", "boundary", "unknown", "eligible", "too_low", "second"],
    )
    search.ranker.rank.return_value = [(pid, 10 - index) for index, pid in enumerate(search.ranker.products.index)]
    monkeypatch.setattr("myapp.search.search_engine._display_product", lambda row: {"pid": row.name})
    assert search.search("men running shoes under 1000 over 500", top_k=1) == [{"pid": "eligible"}]
    search.ranker.rank.assert_called_once_with("men running shoes", "bm25", top_k=None)


def test_inclusive_price_bounds():
    from myapp.search.search_engine import _matches_price

    assert _matches_price(899, [("at most", 899), ("at least", 899)])
    assert not _matches_price(899, [("over", 899)])
