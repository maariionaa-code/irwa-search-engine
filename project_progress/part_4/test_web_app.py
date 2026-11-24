"""Exercise the Part 4 routes without the course catalogue or a Groq key."""

import importlib.util
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock

import pytest

from myapp.generation.rag import RAGSummary


class Page(HTMLParser):
    def __init__(self, response):
        super().__init__()
        self.links = []
        self.click_id = None
        self.feed(response.get_data(as_text=True))

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if "data-analytics-click" in attrs:
            self.click_id = attrs["data-analytics-click"]

    def links_to(self, endpoint):
        return [link for link in self.links if urlsplit(link).path == endpoint]


@pytest.fixture
def site(monkeypatch):
    products = [
        {"pid": "P1", "title": "Slim Black Jeans", "description": "Cotton denim.",
         "selling_price": 600.0, "discount": 10.0, "average_rating": 4.2},
        {"pid": "P2", "title": "Slim Blue Jeans", "description": "Stretch denim.",
         "selling_price": 750.0, "discount": 0.0, "average_rating": 4.5},
    ]
    for product in products:
        product.update(
            brand="Example", category="Clothing", sub_category="Jeans",
            seller="Example", actual_price=900.0, out_of_stock=False, images=[],
            product_details={"Fabric": "Denim"},
            url=f"https://example.com/{product['pid']}",
        )
    by_id = {product["pid"]: product for product in products}
    search_engine = Mock()
    search_engine.get_product.side_effect = by_id.get
    search_engine.search.side_effect = lambda query: products if query == "men slim fit jeans" else []
    rag = Mock()
    rag.generate_response.side_effect = lambda query, hits: RAGSummary(
        status="unavailable" if hits else "no_match"
    )

    monkeypatch.setattr("myapp.search.search_engine.SearchEngine", lambda: search_engine)
    monkeypatch.setattr("myapp.generation.rag.RAGGenerator", lambda: rag)
    monkeypatch.delenv("IRWA_GEOIP_CSV", raising=False)
    app_file = Path(__file__).resolve().parents[2] / "web_app.py"
    spec = importlib.util.spec_from_file_location("irwa_web_app_under_test", app_file)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.app.config.update(TESTING=True)
    return module


def test_dashboard_starts_empty_and_has_navigation(site):
    client = site.app.test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert Page(home).links_to("/dashboard") == ["/dashboard"]
    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert b"No searches in this period." in dashboard.data


def test_search_detail_return_and_dashboard_counts(site):
    client = site.app.test_client()
    results = client.get("/search?q=men+slim+fit+jeans")
    assert results.status_code == 200
    assert b"The AI summary is unavailable" in results.data
    links = Page(results).links_to("/doc_details")
    assert [parse_qs(urlsplit(link).query)["pid"][0] for link in links] == ["P1", "P2"]

    detail = client.get(links[1])
    assert detail.status_code == 200
    assert b"Slim Blue Jeans" in detail.data and b"Product specifications" in detail.data
    assert Page(detail).links_to("/search")
    returned = client.get(Page(detail).links_to("/search")[0])
    assert returned.status_code == 200
    assert Page(returned).links_to("/doc_details") == links
    assert site.search_engine.search.call_count == 1
    assert site.rag_generator.generate_response.call_count == 1

    empty = client.get("/search?q=zzzxxyynoresults")
    assert b"No products matched" in empty.data
    assert b"No retrieved product had enough evidence" in empty.data
    for period in ("all", "24h", "7d"):
        dashboard = client.get("/dashboard", query_string={"period": period})
        assert dashboard.status_code == 200
        assert dashboard.data.count(b"50.0%") == 2
    metrics = site.analytics_data.dashboard("all")
    assert (metrics["searches"], metrics["clicks"], metrics["dwell_count"]) == (2, 1, 1)
    assert metrics["ranks"] == [(2, 1)]
    assert b"Slim Blue Jeans" in client.get("/stats").data


def test_summary_link_records_source_and_allows_one_return(site):
    site.rag_generator.generate_response.side_effect = None
    site.rag_generator.generate_response.return_value = RAGSummary(
        status="ok", pid="P2", title="Slim Blue Jeans", facts=(("Fabric", "Denim"),),
    )
    client = site.app.test_client()
    results = client.get("/search?q=men+slim+fit+jeans")
    assert b"Suggested match" in results.data
    link = Page(results).links_to("/doc_details")[0]
    params = parse_qs(urlsplit(link).query)
    assert params["source"] == ["summary"]
    click_id = Page(client.get(link)).click_id
    payload = {"click_id": click_id, "search_id": params["search_id"][0]}

    stranger = site.app.test_client()
    assert stranger.post("/analytics/return", json=payload).status_code == 404
    assert client.post("/analytics/return", json=payload).status_code == 204
    assert client.post("/analytics/return", json=payload).status_code == 204
    metrics = site.analytics_data.dashboard("all")
    assert metrics["summary_clicks"] == 1
    assert metrics["dwell_count"] == 1
    assert metrics["ranks"] == [(2, 1)]


def test_invalid_requests_have_no_search_or_click_events(site):
    client = site.app.test_client()
    assert client.get("/search?q=%20").status_code == 400
    assert client.get("/doc_details?pid=unknown").status_code == 404
    assert client.post("/analytics/return", json=[]).status_code == 400
    assert client.head("/search?q=men+slim+fit+jeans").status_code == 200
    metrics = site.analytics_data.dashboard("all")
    assert (metrics["requests"], metrics["request_errors"]) == (2, 2)
    assert (metrics["searches"], metrics["clicks"]) == (0, 0)
