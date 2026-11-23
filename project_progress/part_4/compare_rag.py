"""Record the starter and improved RAG outputs over identical BM25 results."""

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

from groq import GroqError

from myapp.generation.rag import RAGGenerator, prepare_context
from myapp.search.search_engine import SearchEngine


QUERIES = (
    "women cotton kurta",
    "men running shoes",
    "men slim fit jeans",
    "zzzxxyynoresults",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="*", help="Queries to run instead of the default cases.")
    parser.add_argument("--inspect", action="store_true", help="Record inputs without any API calls.")
    parser.add_argument("--pause", type=float, default=25, help="Seconds between live query pairs.")
    parser.add_argument("--output", type=Path, default=Path("data/rag-comparison.json"))
    args = parser.parse_args()
    if args.pause < 0:
        parser.error("--pause must be non-negative")

    generator = RAGGenerator() if not args.inspect else None
    if generator is not None and not generator.configured:
        parser.error("Set GROQ_API_KEY in the environment or .env to compare live outputs.")

    with redirect_stdout(sys.stderr):
        engine = SearchEngine()

    cases = []
    for index, query in enumerate(args.queries or QUERIES):
        if index and generator is not None:
            time.sleep(args.pause)
        print(f"Checking {query!r}", file=sys.stderr)
        products = engine.search(query)
        context = prepare_context(query, products)
        case = {
            "query": query,
            "retrieved": [{"pid": item["pid"], "title": item["title"]} for item in products],
            "improved_context": context,
        }
        if generator is not None:
            try:
                case["baseline"] = {
                    "status": "completed", "text": generator.generate_baseline(query, products)
                }
            except GroqError as error:
                case["baseline"] = {"status": "unavailable", "error": type(error).__name__}
            case["improved"] = asdict(generator.generate_response(query, products))
        cases.append(case)

    report = {
        "mode": "live" if generator is not None else "inspect",
        "model": generator.model if generator is not None else None,
        "ranker": "bm25",
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
