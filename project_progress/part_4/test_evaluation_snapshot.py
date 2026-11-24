"""Check the frozen live run and the data used in the report's candidate chart."""

import json
import re
from pathlib import Path

from myapp.generation.rag import EVIDENCE_LABELS


HERE = Path(__file__).resolve().parent


def load_json(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def test_figure_four_matches_the_saved_live_run():
    run = load_json("rag-comparison.json")
    counts = load_json("figure4_counts.json")

    assert run["mode"] == "live"
    assert run["model"] == "openai/gpt-oss-20b"
    assert len(run["cases"]) == len({case["query"] for case in run["cases"]}) == 12
    assert counts["queries"] == [
        {
            "query": case["query"],
            "bm25": len(case["retrieved"]),
            "eligible": len(case["improved_context"]),
        }
        for case in run["cases"]
    ]


def test_frozen_recommendations_are_grounded_in_the_retrieved_context():
    for case in load_json("rag-comparison.json")["cases"]:
        retrieved = [product["pid"] for product in case["retrieved"]]
        eligible = {item["pid"]: item for item in case["improved_context"]}
        assert len(retrieved) <= 20
        assert list(eligible) == [pid for pid in retrieved if pid in eligible]

        response = case["improved"]
        if response["status"] == "ok":
            product = eligible[response["pid"]]
            facts = {EVIDENCE_LABELS[field]: value for field, value in product["evidence"].items()}
            assert response["title"] == product["title"]
            assert 1 <= len(response["facts"]) <= 3
            assert all(facts[label] == value for label, value in response["facts"])


def test_reported_product_ids_and_ranks_match_the_snapshot():
    evaluation = (HERE / "rag_evaluation.md").read_text(encoding="utf-8")
    rows = {
        parts[0]: parts
        for line in evaluation.splitlines()
        if line.startswith("| ")
        for parts in [[cell.strip() for cell in line.split("|")[1:-1]]]
    }
    for case in load_json("rag-comparison.json")["cases"]:
        ranked = [product["pid"] for product in case["retrieved"]]
        row = rows[case["query"]]
        assert row[1] == f'{len(ranked)} / {len(case["improved_context"])}'

        first_line = case["baseline"]["text"].splitlines()[0]
        baseline_match = re.search(r"\b[A-Z0-9]{10,20}\b", first_line)
        baseline_pid = baseline_match.group() if baseline_match else ""
        improved_pid = case["improved"]["pid"]
        for column, pid in ((2, baseline_pid), (4, improved_pid)):
            display = f"`{pid}` ({ranked.index(pid) + 1})" if pid else "Abstain"
            assert row[column] == display
