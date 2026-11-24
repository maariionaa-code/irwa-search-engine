"""Run the Flask interface for the fashion product search engine."""

import os
from datetime import timedelta
from secrets import token_hex
from time import perf_counter

from flask import Flask, abort, g, render_template, request, session

from myapp.analytics.analytics_data import PERIODS, AnalyticsData, GeoLookup
from myapp.generation.rag import RAGGenerator
from myapp.search.objects import StatsDocument
from myapp.search.search_engine import SearchEngine


app = Flask(__name__)
search_engine = SearchEngine()
rag_generator = RAGGenerator()
geo_path = os.getenv("IRWA_GEOIP_CSV")
analytics_data = AnalyticsData(geo=GeoLookup(geo_path) if geo_path else None)
app.config.update(
    SECRET_KEY=os.getenv("IRWA_SECRET_KEY") or token_hex(32),
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    SESSION_REFRESH_EACH_REQUEST=False,
    SESSION_COOKIE_SAMESITE="Lax",
)


@app.before_request
def begin_shopping_request():
    if request.method == "GET" and request.endpoint in {"index", "search", "doc_details"}:
        g.request_start = perf_counter()
        if "visitor_id" not in session:
            session["visitor_id"] = token_hex(16)
            session.permanent = True
        g.analytics_session = analytics_data.visit(
            session["visitor_id"], request.headers.get("User-Agent", ""), request.remote_addr,
        )


@app.after_request
def record_shopping_request(response):
    session_id = g.get("analytics_session")
    if session_id:
        analytics_data.log_request(
            session_id, request.endpoint, request.method, response.status_code, g.request_start,
        )
    return response


@app.get("/")
def index():
    return render_template("index.html", page_title="Search")


@app.get("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return render_template(
            "index.html", page_title="Search", error="Enter a product search."
        ), 400
    if request.method == "HEAD":
        return ""

    issued = analytics_data.find_search(request.args.get("search_id", ""), session["visitor_id"])
    if issued is not None and issued.query == query:
        returned_click = request.args.get("returned_click")
        if returned_click:
            analytics_data.log_return(returned_click, issued.id, session["visitor_id"])
        results = [search_engine.get_product(pid) for pid in issued.pids]
    else:
        results = search_engine.search(query)
        summary = rag_generator.generate_response(query, results)
        issued = analytics_data.log_search(
            g.analytics_session, query, tuple(product["pid"] for product in results), summary,
        )
    return render_template(
        "results.html", page_title="Search results", query=query,
        results=results, summary=issued.summary, search_id=issued.id,
    )


@app.get("/doc_details")
def doc_details():
    pid = request.args.get("pid", "")
    product = search_engine.get_product(pid)
    if product is None:
        abort(404)
    if request.method == "HEAD":
        return ""

    issued = analytics_data.find_search(request.args.get("search_id", ""), session["visitor_id"])
    click_id = analytics_data.log_detail(
        g.analytics_session, pid, issued, request.args.get("source", ""),
    )
    return render_template(
        "doc_details.html", page_title=product["title"], product=product,
        query=issued.query if click_id else request.args.get("q", ""),
        search_id=issued.id if click_id else None, click_id=click_id,
    )


@app.post("/analytics/return")
def analytics_return():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400)
    click_id, search_id = payload.get("click_id"), payload.get("search_id")
    if not isinstance(click_id, str) or not isinstance(search_id, str):
        abort(400)
    if not analytics_data.log_return(click_id, search_id, session.get("visitor_id", "")):
        abort(404)
    return "", 204


@app.get("/stats")
def stats():
    docs = []
    for pid, count in analytics_data.product_views():
        product = search_engine.get_product(pid)
        docs.append(StatsDocument(
            pid=pid, title=product["title"], description=product["description"],
            url=product["url"], count=count,
        ))
    return render_template("stats.html", page_title="Quick stats", clicks_data=docs)


@app.get("/dashboard")
def dashboard():
    period = request.args.get("period", "all")
    if period not in PERIODS:
        period = "all"
    return render_template(
        "dashboard.html", page_title="Analytics", period=period,
        metrics=analytics_data.dashboard(period),
    )


@app.get("/plot_number_of_views")
def plot_number_of_views():
    return render_template("product_views_chart.html", rows=analytics_data.product_views())


if __name__ == "__main__":
    app.run(port=8088)
