"""Verify grounded selection and fallback without contacting Groq."""

import json
from unittest.mock import Mock

import pytest
from groq import GroqError

from myapp.generation.rag import DEFAULT_MODEL, RAGGenerator, prepare_context


WOMENS_KURTA = {
    "pid": "KURTA1", "title": "Women Cotton Kurta", "out_of_stock": False,
    "selling_price": 499.0, "discount": 0.0, "average_rating": 4.2,
    "product_details": {"Ideal For": "Women", "Fabric": "Cotton"},
    "url": "https://example.com/women-cotton-kurta/p/1",
}


@pytest.fixture
def generator_and_completion(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr("myapp.generation.rag.load_dotenv", lambda: None)
    client = Mock()
    return RAGGenerator(client=client), client.chat.completions.create


def test_uses_only_verified_catalogue_values(generator_and_completion):
    generator, complete = generator_and_completion
    complete.return_value.choices = [
        Mock(message=Mock(content='{"pid":"KURTA1","evidence":["fabric","selling_price","discount"]}'))
    ]
    result = generator.generate_response("women cotton kurta", [WOMENS_KURTA])

    assert result.status == "ok"
    assert result.pid == "KURTA1"
    assert result.facts == (("Fabric", "Cotton"), ("Selling price", "499.00"), ("Discount", "0%"))
    request = complete.call_args.kwargs
    assert request["model"] == DEFAULT_MODEL
    assert request["response_format"] == {"type": "json_object"}
    assert json.loads(request["messages"][1]["content"])["products"][0]["pid"] == "KURTA1"


def test_excludes_conflicting_mismatched_and_unavailable_products():
    conflict = {**WOMENS_KURTA, "pid": "CONFLICT", "url": "https://example.com/men-kurta/p/2"}
    out_of_stock = {**WOMENS_KURTA, "pid": "SOLDOUT", "out_of_stock": True}
    men = {**WOMENS_KURTA, "pid": "MENS1", "title": "Men Cotton Kurta",
           "product_details": {"Ideal For": "Men"}, "url": "https://example.com/men-kurta/p/3"}
    unknown = {**WOMENS_KURTA, "pid": "UNKNOWN", "title": "Cotton Kurta",
               "product_details": {}, "url": "https://example.com/cotton-kurta/p/4"}

    assert [item["pid"] for item in prepare_context(
        "women cotton kurta", [conflict, out_of_stock, men, unknown, WOMENS_KURTA]
    )] == ["KURTA1"]
    assert prepare_context("cotton kurta", [conflict]) == []


def test_experimental_policy_accepts_agreeing_fields_with_url_warning():
    product = {**WOMENS_KURTA, "url": "https://example.com/men-kurta/p/1"}
    diagnostics = []
    assert prepare_context("women cotton kurta", [product], diagnostics=diagnostics) == []
    assert diagnostics == [{"pid": "KURTA1", "exclusion_reason": "url_audience_conflict"}]
    context = prepare_context("women cotton kurta", [product], allow_slug_override=True)
    assert len(context) == 1 and context[0]["audience_warning"]


def test_explicit_conflict_stays_excluded_under_experimental_policy():
    product = {**WOMENS_KURTA, "product_details": {"Ideal For": "Men"}}
    diagnostics = []
    assert prepare_context("women kurta", [product], allow_slug_override=True, diagnostics=diagnostics) == []
    assert diagnostics == [{"pid": "KURTA1", "exclusion_reason": "explicit_audience_conflict"}]


def test_prioritises_grounded_hood_attribute(generator_and_completion):
    generator, complete = generator_and_completion
    product = {**WOMENS_KURTA, "title": "Women Sweatshirt", "category": "Winter wear",
               "product_details": {"Hooded": "Yes", "Ideal For": "Women"}}
    complete.return_value.choices = [
        Mock(message=Mock(content='{"pid":"KURTA1","evidence":["category"]}'))
    ]
    result = generator.generate_response("hooded sweatshirt", [product])
    assert result.facts == (("Hooded", "Yes"), ("Category", "Winter wear"))


def test_no_eligible_product_skips_the_api(generator_and_completion):
    generator, complete = generator_and_completion
    assert generator.generate_response("men cotton kurta", [WOMENS_KURTA]).status == "no_match"
    assert generator.generate_response("zzzxxyynoresults", []).status == "no_match"
    complete.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ('{"pid":"","evidence":[]}', "no_match"),
        ('{"pid":"INVENTED","evidence":["fabric"]}', "invalid_response"),
        ('{"pid":"KURTA1","evidence":["free_shipping"]}', "invalid_response"),
        ('{"pid":"KURTA1","evidence":["fabric","fabric"]}', "invalid_response"),
        ("invalid JSON", "invalid_response"),
    ],
)
def test_handles_abstention_and_invalid_replies(generator_and_completion, payload, status):
    generator, complete = generator_and_completion
    complete.return_value.choices = [Mock(message=Mock(content=payload))]
    assert generator.generate_response("women cotton kurta", [WOMENS_KURTA]).status == status


def test_gracefully_handles_provider_failure(generator_and_completion):
    generator, complete = generator_and_completion
    complete.side_effect = GroqError("test error")
    assert generator.generate_response("women cotton kurta", [WOMENS_KURTA]).status == "unavailable"


def test_without_credentials(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("myapp.generation.rag.load_dotenv", lambda: None)
    assert RAGGenerator().generate_response("women cotton kurta", [WOMENS_KURTA]).status == "unavailable"


def test_baseline_uses_the_same_model_with_titles_only(generator_and_completion):
    generator, complete = generator_and_completion
    complete.return_value.choices = [Mock(message=Mock(content="baseline reply"))]
    assert generator.generate_baseline("kurta", [WOMENS_KURTA]) == "baseline reply"
    request = complete.call_args.kwargs
    assert request["model"] == DEFAULT_MODEL
    assert request["temperature"] == 0
    assert "PID: KURTA1, Title: Women Cotton Kurta" in request["messages"][0]["content"]
    assert "499" not in request["messages"][0]["content"]
    assert "response_format" not in request
