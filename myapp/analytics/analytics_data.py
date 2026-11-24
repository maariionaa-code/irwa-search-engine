"""In-memory search analytics with visitor, session and search attribution."""

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address, ip_network
from statistics import median
from threading import RLock
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

import httpagentparser

from project_progress.part_1.preprocessing import preprocess_text

if TYPE_CHECKING:
    from myapp.generation.rag import RAGSummary


SESSION_TIMEOUT = timedelta(minutes=30)
MISSION_LOOKBACK = timedelta(days=7)
PERIODS = {"all": None, "24h": timedelta(hours=24), "7d": timedelta(days=7)}


class GeoLookup:
    """Resolve client addresses locally using an optional network/country/city CSV."""

    def __init__(self, path: str):
        self.networks = {4: {}, 6: {}}
        with open(path, newline="", encoding="utf-8-sig") as source:
            rows = csv.DictReader(source)
            if not {"network", "country", "city"}.issubset(rows.fieldnames or ()):
                raise ValueError("GeoIP CSV requires network,country,city columns")
            for row in rows:
                network = ip_network(row["network"].strip())
                by_prefix = self.networks[network.version].setdefault(network.prefixlen, {})
                by_prefix[int(network.network_address)] = (row["country"].strip(), row["city"].strip())

    def locate(self, address: str | None) -> tuple[str, str]:
        try:
            ip = ip_address(address)
        except ValueError:
            return "", ""

        for prefix, entries in sorted(self.networks[ip.version].items(), reverse=True):
            host_bits = ip.max_prefixlen - prefix
            match = entries.get((int(ip) >> host_bits) << host_bits)
            if match is not None:
                return match
        return "", ""


@dataclass
class Visit:
    visitor_id: str
    last_seen: datetime
    browser: str
    os: str
    device: str
    country: str
    city: str


@dataclass
class Mission:
    id: str
    last_seen: datetime
    terms: frozenset[str]
    normalized_query: str


@dataclass(frozen=True)
class SearchEvent:
    id: str
    session_id: str
    mission_id: str
    at: datetime
    query: str
    terms: tuple[str, ...]
    pids: tuple[str, ...]
    summary: "RAGSummary"


@dataclass
class ClickEvent:
    search_id: str
    session_id: str
    at: datetime
    pid: str
    rank: int
    source: str
    dwell_seconds: float | None = None


@dataclass(frozen=True)
class RequestEvent:
    session_id: str
    at: datetime
    endpoint: str
    method: str
    status: int
    milliseconds: float


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnalyticsData:
    """Store activity for one Flask process and calculate dashboard measures."""

    def __init__(self, *, clock=_utc_now, geo: GeoLookup | None = None):
        self._clock = clock
        self._geo = geo
        self._lock = RLock()
        self._active_visits = {}
        self._visits = {}
        self._missions = defaultdict(list)
        self._searches = {}
        self._clicks = {}
        self._outbound = []
        self._requests = []
        self._product_views = Counter()

    def visit(self, visitor_id: str, user_agent: str, address: str | None) -> str:
        """Return the active session ID, starting a new visit after 30 idle minutes."""
        with self._lock:
            now = self._clock()
            active_id = self._active_visits.get(visitor_id)
            if active_id and now - self._visits[active_id].last_seen < SESSION_TIMEOUT:
                self._visits[active_id].last_seen = now
                return active_id

            agent = httpagentparser.detect(user_agent)
            browser = agent.get("browser", {}).get("name", "Unknown")
            os = agent.get("os", {}).get("name", "Unknown")
            ua = user_agent.lower()
            if any(term in ua for term in ("ipad", "tablet")) or ("android" in ua and "mobile" not in ua):
                device = "Tablet"
            elif any(term in ua for term in ("mobile", "iphone", "android")):
                device = "Mobile"
            else:
                device = "Desktop" if os != "Unknown" else "Unknown"
            country, city = self._geo.locate(address) if self._geo else ("", "")
            session_id = uuid4().hex
            self._visits[session_id] = Visit(visitor_id, now, browser, os, device, country, city)
            self._active_visits[visitor_id] = session_id
            return session_id

    def log_request(self, session_id: str, endpoint: str, method: str, status: int, start: float) -> None:
        """Capture one request, including HTTP status and elapsed server time."""
        with self._lock:
            self._requests.append(RequestEvent(
                session_id, self._clock(), endpoint, method, status, (perf_counter() - start) * 1000,
            ))

    def _mission(self, session_id: str, query: str, terms: tuple[str, ...], now: datetime) -> str:
        visitor = self._visits[session_id].visitor_id
        recent = [
            mission for mission in self._missions[visitor]
            if now - mission.last_seen < MISSION_LOOKBACK
        ]
        self._missions[visitor] = recent
        normalized = " ".join(query.casefold().split())
        tokens = frozenset(terms)
        for mission in sorted(recent, key=lambda item: item.last_seen, reverse=True):
            overlap = len(tokens & mission.terms)
            union = len(tokens | mission.terms)
            if normalized == mission.normalized_query or (overlap >= 2 and overlap * 2 >= union):
                mission.last_seen = now
                mission.terms = tokens
                mission.normalized_query = normalized
                return mission.id

        mission = Mission(uuid4().hex, now, tokens, normalized)
        recent.append(mission)
        return mission.id

    def log_search(self, session_id: str, query: str, pids: tuple[str, ...], summary: "RAGSummary") -> SearchEvent:
        """Persist the issued query, its mission, ordered hits and displayed summary."""
        terms = tuple(preprocess_text(query))
        with self._lock:
            now = self._clock()
            event = SearchEvent(
                uuid4().hex, session_id, self._mission(session_id, query, terms, now),
                now, query, terms, pids, summary,
            )
            self._searches[event.id] = event
            return event

    def find_search(self, search_id: str, visitor_id: str) -> SearchEvent | None:
        """Return an issued search belonging to this browser."""
        with self._lock:
            search = self._searches.get(search_id)
            if search and self._visits[search.session_id].visitor_id == visitor_id:
                return search
        return None

    def log_detail(
        self, session_id: str, pid: str, search: SearchEvent | None, source: str
    ) -> str | None:
        """Count a product view and record a ranked click when its result is known."""
        with self._lock:
            self._product_views[pid] += 1
            if search is None or pid not in search.pids:
                return None
            click_id = uuid4().hex
            from_summary = source == "summary" and search.summary.pid == pid
            self._clicks[click_id] = ClickEvent(
                search.id, session_id, self._clock(), pid,
                search.pids.index(pid) + 1, "summary" if from_summary else "result",
            )
            return click_id

    def log_outbound(self, search: SearchEvent | None, pid: str) -> None:
        """Record a retailer link separately from attributable detail opens."""
        if search is not None and pid in search.pids:
            with self._lock:
                self._outbound.append((search.id, pid, self._clock()))

    def log_return(self, click_id: str, search_id: str, visitor_id: str) -> bool:
        """Record one return to the click's results within the dwell window."""
        with self._lock:
            click = self._clicks.get(click_id)
            if (
                click is None or click.search_id != search_id
                or self._visits[click.session_id].visitor_id != visitor_id
            ):
                return False
            if click.dwell_seconds is not None:
                return True
            elapsed = self._clock() - click.at
            if not timedelta() <= elapsed < SESSION_TIMEOUT:
                return False
            click.dwell_seconds = elapsed.total_seconds()
            return True

    def product_views(self) -> list[tuple[str, int]]:
        """Return total detail views by product for Quick Stats."""
        with self._lock:
            return self._product_views.most_common()

    def dashboard(self, period: str) -> dict:
        """Summarize searches by issue time and clicks by their search cohort."""
        with self._lock:
            now = self._clock()
            window = PERIODS[period]
            since = now - window if window else None
            requests = [event for event in self._requests if since is None or event.at >= since]
            searches = [event for event in self._searches.values() if since is None or event.at >= since]
            search_ids = {event.id for event in searches}
            clicks = [event for event in self._clicks.values() if event.search_id in search_ids]
            visits = {
                event.session_id for event in requests
            } | {event.session_id for event in searches}
            profiles = [self._visits[visit] for visit in visits]
            dwell = [event.dwell_seconds for event in clicks if event.dwell_seconds is not None]

            queries = Counter(" ".join(search.query.casefold().split()) for search in searches)
            empty = Counter(" ".join(search.query.casefold().split()) for search in searches if not search.pids)
            mission_visits = defaultdict(set)
            for search in searches:
                mission_visits[search.mission_id].add(search.session_id)

            ordered_ranks = Counter(event.rank for event in clicks)
            hourly = period == "24h"
            stamp = "%d %b %H:00" if hourly else "%d %b %Y"
            block = timedelta(hours=1) if hourly else timedelta(days=1)
            trend = Counter(event.at.strftime(stamp) for event in searches)
            start = since or (min(event.at for event in searches) if searches else now)
            cursor = start.replace(minute=0, second=0, microsecond=0) if hourly else start.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            timeline = []
            while cursor <= now:
                label = cursor.strftime(stamp)
                timeline.append((label, trend[label]))
                cursor += block

            duration = [event.milliseconds for event in requests]
            return {
                "requests": len(requests),
                "request_errors": sum(event.status >= 400 for event in requests),
                "median_request_ms": median(duration) if duration else None,
                "sessions": len(visits),
                "visitors": len({profile.visitor_id for profile in profiles}),
                "searches": len(searches),
                "empty_searches": sum(empty.values()),
                "clicks": len(clicks),
                "clicked_searches": len({event.search_id for event in clicks}),
                "summary_clicks": sum(event.source == "summary" for event in clicks),
                "outbound_clicks": sum(search_id in search_ids for search_id, _, _ in self._outbound),
                "median_dwell": median(dwell) if dwell else None,
                "dwell_count": len(dwell),
                "missions": len(mission_visits),
                "multi_session_missions": sum(len(ids) > 1 for ids in mission_visits.values()),
                "timeline": timeline if searches else [],
                "ranks": sorted(ordered_ranks.items()),
                "queries": [(query, count, empty[query]) for query, count in queries.most_common(10)],
                "avg_terms": sum(len(event.terms) for event in searches) / len(searches) if searches else None,
                "devices": Counter(profile.device for profile in profiles).most_common(),
                "browsers": Counter(profile.browser for profile in profiles).most_common(5),
                "systems": Counter(profile.os for profile in profiles).most_common(5),
                "locations": Counter(
                    ", ".join(filter(None, (profile.city, profile.country)))
                    for profile in profiles if profile.city or profile.country
                ).most_common(5),
                "geo_enabled": self._geo is not None,
            }
