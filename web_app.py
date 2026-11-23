"""Run the Flask interface for the fashion product search engine."""

from flask import Flask, abort, render_template, request

from myapp.analytics.analytics_data import AnalyticsData
from myapp.generation.rag import RAGGenerator
from myapp.search.objects import StatsDocument
from myapp.search.search_engine import SearchEngine


app = Flask(__name__)
search_engine = SearchEngine()
rag_generator = RAGGenerator()
analytics_data = AnalyticsData()


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

    results = search_engine.search(query)
    summary = rag_generator.generate_response(query, results)
    return render_template(
        "results.html", page_title="Search results", query=query, results=results,
        summary=summary,
    )


@app.get("/doc_details")
def doc_details():
    pid = request.args.get("pid", "")
    product = search_engine.get_product(pid)
    if product is None:
        abort(404)

    analytics_data.fact_clicks[pid] = analytics_data.fact_clicks.get(pid, 0) + 1
    return render_template(
        "doc_details.html",
        page_title=product["title"],
        product=product,
        query=request.args.get("q", ""),
    )


@app.get("/stats")
def stats():
    ranked_clicks = sorted(
        analytics_data.fact_clicks.items(), key=lambda item: item[1], reverse=True
    )
    docs = []
    for pid, count in ranked_clicks:
        product = search_engine.get_product(pid)
        docs.append(StatsDocument(
            pid=pid,
            title=product["title"],
            description=product["description"],
            url=product["url"],
            count=count,
        ))
    return render_template("stats.html", page_title="Quick stats", clicks_data=docs)


@app.get("/dashboard")
def dashboard():
    return render_template("dashboard.html", page_title="Dashboard")


@app.get("/plot_number_of_views")
def plot_number_of_views():
    if not analytics_data.fact_clicks:
        return "<p>No product views yet.</p>"
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    app.run(port=8088)
