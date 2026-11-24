"""Inspect post-evaluation changes on the course catalogue without calling Groq."""

import json
import sys
import csv
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from myapp.generation.rag import prepare_context
from myapp.search.search_engine import SearchEngine
from project_progress.part_4.compare_rag import QUERIES


def _pool_counts(context, labels, relevant_total):
    if labels is None:
        return None
    pids = {product["pid"] for product in context}
    judged = pids & labels.keys()
    relevant = sum(labels[pid] for pid in judged)
    return {
        "judged_kept": len(judged),
        "relevant_kept": relevant,
        "precision_on_judged": round(relevant / len(judged), 3) if judged else None,
        "retained_relevant_share": round(relevant / relevant_total, 3) if relevant_total else None,
    }


def main() -> None:
    with redirect_stdout(sys.stderr):
        engine = SearchEngine()

    labels = {}
    source = Path(__file__).resolve().parents[1] / "part_3" / "judgments_pool.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            if item["relevance"] in {"0", "1"}:
                labels.setdefault(item["query"], {})[item["pid"]] = int(item["relevance"])

    cases = []
    for query in QUERIES:
        results = engine.search(query)
        exclusions = []
        strict = prepare_context(query, results, diagnostics=exclusions)
        relaxed = prepare_context(query, results, allow_slug_override=True)
        judged = labels.get(query)
        relevant_total = sum(judged.get(product["pid"], 0) for product in results) if judged else 0
        cases.append({
            "query": query,
            "hits": len(results),
            "strict_eligible": len(strict),
            "relaxed_eligible": len(relaxed),
            "url_warnings": sum(bool(product.get("audience_warning")) for product in relaxed),
            "strict_exclusions": dict(Counter(row["exclusion_reason"] for row in exclusions)),
            "judged_relevant_in_retrieved": relevant_total if judged else None,
            "strict_existing_judgements": _pool_counts(strict, judged, relevant_total),
            "relaxed_existing_judgements": _pool_counts(relaxed, judged, relevant_total),
        })

    price_results = [
        {"pid": row["pid"], "title": row["title"], "price": row["selling_price"]}
        for row in engine.search("men running shoes under 1000")
    ]
    comparison = {
        method: [
            {"pid": pid, "title": engine.get_product(pid)["title"]}
            for pid, _ in engine.ranker.rank("men running shoes", method, top_k=5)
        ]
        for method in ("bm25", "our_score")
    }
    print(json.dumps({"audience": cases, "price_results": price_results, "top_five": comparison}, indent=2))


if __name__ == "__main__":
    main()
