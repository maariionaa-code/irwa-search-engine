"""Cover click attribution, time windows, location and mission metrics."""

from datetime import datetime, timedelta, timezone
from time import perf_counter

import pytest

from myapp.analytics.analytics_data import AnalyticsData, GeoLookup
from myapp.generation.rag import RAGSummary


DESKTOP = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)
MOBILE = (
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36"
)
NO_SUMMARY = RAGSummary(status="no_match")


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **duration):
        self.now += timedelta(**duration)


@pytest.fixture
def analytics():
    clock = Clock()
    return AnalyticsData(clock=clock), clock


def test_rank_ctr_dwell_and_empty_searches(analytics):
    data, clock = analytics
    session_id = data.visit("visitor", DESKTOP, None)
    search = data.log_search(session_id, "women cotton kurta", ("p1", "p2"), NO_SUMMARY)
    data.log_search(session_id, "zzzxxyynoresults", (), NO_SUMMARY)
    click = data.log_detail(session_id, "p2", search, "")
    data.log_detail(session_id, "p2", search, "")
    assert data.log_detail(session_id, "direct", None, "") is None
    clock.advance(seconds=12)
    assert data.log_return(click, search.id, "visitor")
    clock.advance(seconds=5)
    assert data.log_return(click, search.id, "visitor")

    metrics = data.dashboard("all")
    assert metrics["searches"] == 2
    assert metrics["empty_searches"] == 1
    assert metrics["clicked_searches"] == 1
    assert metrics["clicks"] == 2
    assert metrics["ranks"] == [(2, 2)]
    assert metrics["dwell_count"] == 1
    assert metrics["median_dwell"] == 12
    assert dict(data.product_views()) == {"p2": 2, "direct": 1}


def test_click_ownership_and_late_returns(analytics):
    data, clock = analytics
    session_id = data.visit("first", DESKTOP, None)
    search = data.log_search(session_id, "men slim fit jeans", ("p1",), NO_SUMMARY)
    click = data.log_detail(session_id, "p1", search, "")
    assert data.find_search(search.id, "first") == search
    assert data.find_search(search.id, "second") is None
    assert not data.log_return(click, search.id, "second")
    assert not data.log_return(click, "wrong-search", "first")
    clock.advance(minutes=30)
    assert not data.log_return(click, search.id, "first")
    assert data.dashboard("all")["median_dwell"] is None


def test_summary_click_keeps_its_rank(analytics):
    data, _ = analytics
    session_id = data.visit("visitor", DESKTOP, None)
    summary = RAGSummary(status="ok", pid="p2", title="Second product")
    search = data.log_search(session_id, "men slim fit jeans", ("p1", "p2"), summary)
    data.log_detail(session_id, "p2", search, "summary")
    data.log_detail(session_id, "p1", search, "summary")
    assert data.log_detail(session_id, "unranked", search, "") is None
    metrics = data.dashboard("all")
    assert metrics["summary_clicks"] == 1
    assert metrics["ranks"] == [(1, 1), (2, 1)]


def test_visits_and_missions_across_sessions(analytics):
    data, clock = analytics
    visit = data.visit("visitor", DESKTOP, None)
    first = data.log_search(visit, "men slim fit jeans", ("p1",), NO_SUMMARY)
    clock.advance(minutes=29)
    assert data.visit("visitor", DESKTOP, None) == visit
    clock.advance(minutes=30)
    next_visit = data.visit("visitor", DESKTOP, None)
    assert next_visit != visit
    related = data.log_search(next_visit, "men slim fit blue jeans", ("p1",), NO_SUMMARY)
    unrelated = data.log_search(next_visit, "hooded sweatshirt", ("p3",), NO_SUMMARY)
    assert related.mission_id == first.mission_id
    assert unrelated.mission_id != first.mission_id
    metrics = data.dashboard("all")
    assert (metrics["visitors"], metrics["sessions"]) == (1, 2)
    assert (metrics["missions"], metrics["multi_session_missions"]) == (2, 1)

    clock.advance(days=7)
    later = data.log_search(data.visit("visitor", DESKTOP, None), related.query, ("p1",), NO_SUMMARY)
    assert later.mission_id != related.mission_id


def test_dashboard_periods_use_search_cohorts(analytics):
    data, clock = analytics
    session_id = data.visit("desktop", DESKTOP, None)
    search = data.log_search(session_id, "men slim fit jeans", ("p1",), NO_SUMMARY)
    clock.advance(days=2)
    session_id = data.visit("desktop", DESKTOP, None)
    data.log_detail(session_id, "p1", search, "")
    data.log_request(session_id, "doc_details", "GET", 200, perf_counter())
    mobile = data.visit("mobile", MOBILE, None)
    data.log_search(mobile, "men running shoes", (), NO_SUMMARY)
    data.log_request(mobile, "doc_details", "GET", 404, perf_counter())

    today = data.dashboard("24h")
    assert today["searches"] == 1 and today["clicks"] == 0
    assert today["requests"] == 2 and today["request_errors"] == 1
    assert dict(today["devices"]) == {"Desktop": 1, "Mobile": 1}
    assert sum(count for _, count in today["timeline"]) == 1
    assert data.dashboard("7d")["clicks"] == 1


def test_optional_location_uses_longest_network_match(tmp_path):
    csv_path = tmp_path / "geo.csv"
    csv_path.write_text(
        "network,country,city\n192.0.2.0/24,Test Country,Wide\n"
        "192.0.2.64/26,Test Country,Narrow\n2001:db8::/32,Test Country,IPv6\n",
        encoding="utf-8",
    )
    geo = GeoLookup(str(csv_path))
    assert geo.locate("192.0.2.75") == ("Test Country", "Narrow")
    assert geo.locate("2001:db8::1") == ("Test Country", "IPv6")
    assert geo.locate(None) == ("", "")
    data = AnalyticsData(clock=Clock(), geo=geo)
    session_id = data.visit("visitor", DESKTOP, "192.0.2.75")
    data.log_request(session_id, "index", "GET", 200, perf_counter())
    assert data.dashboard("all")["locations"] == [("Narrow, Test Country", 1)]
