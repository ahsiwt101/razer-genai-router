"""Unit tests for the rule router. Run:  pytest -q"""
import pytest

from src.router.rules import RuleRouter

router = RuleRouter()


@pytest.mark.parametrize("query, route", [
    ("What is 245 * 18?", "calculator"),
    ("what's 12 times 7", "calculator"),
    ("square root of 144", "calculator"),
    ("What's 15% of 80?", "calculator"),
    ("What is the weather in Singapore today?", "weather"),
    ("Will I need an umbrella tomorrow?", "weather"),
    ("Who won the latest Formula 1 race?", "web_search"),
    ("search for the best ramen in Tokyo", "web_search"),
])
def test_strong_rules_fire(query, route):
    decision, _ = router.match(query)
    assert decision is not None and decision.route == route


@pytest.mark.parametrize("query", [
    "Explain gradient descent.",
    "How does rain form?",             # explanatory: science, not a forecast
    "What is the temperature of the sun?",
    "Tell me about covid-19",          # hyphen between letters and digits is not subtraction
    "What's 15% of Tesla's stock price?",  # needs a lookup before any maths
])
def test_rules_abstain_when_unsure(query):
    decision, _ = router.match(query)
    assert decision is None
