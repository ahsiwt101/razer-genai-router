"""Tests for the tools and argument extractors. Run:  pytest -q"""
import pytest

from src.tools import calculator, weather
from src.tools.extractors import extract_expression, extract_location, extract_search_query, words_to_numbers
from src.tools.registry import build_registry


# ------------------------------------------------------------------ calculator safety
@pytest.mark.parametrize("malicious", [
    "__import__('os').system('ls')",   # code execution
    "open('/etc/passwd').read()",      # file access
    "x + 1",                           # variables
    "(1).__class__",                   # attribute access
    "9 ** 9 ** 9",                     # freezes the machine
    "(10 ** 1000) ** 1000",            # million-digit result
    "1e300 * 1e300",                   # float overflow -> inf
])
def test_calculator_rejects_unsafe_input(malicious):
    assert calculator.run(malicious).ok is False


@pytest.mark.parametrize("expression, value", [
    ("245 * 18", 4410), ("2 ^ 10", 1024), ("(45 + 55) * 3", 300), ("sqrt(144)", 12),
    ("factorial(5)", 120), ("-3 + 5", 2), ("(100 - 32) * 5 / 9", 37.77777777777778),
])
def test_calculator_computes(expression, value):
    assert calculator.run(expression).data["value"] == pytest.approx(value)


def test_division_by_zero_is_an_error_not_a_crash():
    result = calculator.run("7 / 0")
    assert result.ok is False and "zero" in result.error


def test_division_by_zero_is_still_extracted():
    # it IS a calculation; the calculator (not the extractor) should tell the user it's undefined
    assert extract_expression("what is 7 / 0") == "7 / 0"


# ------------------------------------------------------------------ extractors
def test_words_to_numbers():
    assert words_to_numbers("two hundred and five times twenty-three") == "205 times 23"


@pytest.mark.parametrize("query, expected", [
    ("What is 245 * 18?", 4410),
    ("What's two hundred times three?", 600),
    ("what is 7% of 2,300", 161),
    ("split 90 dollars between 4 friends", 22.5),
    ("how many minutes are in a week", 10080),
    ("what's the average of 12, 18 and 30", 20),
    ("what's 20 percent off a 45 dollar shirt", 36),
])
def test_extract_expression(query, expected):
    expression = extract_expression(query)
    assert expression is not None and calculator.evaluate(expression) == pytest.approx(expected)


@pytest.mark.parametrize("query", ["what is photosynthesis", "what is 42", "What's 15% of Tesla's stock price?"])
def test_extract_expression_returns_none_when_nothing_to_compute(query):
    assert extract_expression(query) is None


@pytest.mark.parametrize("query, location", [
    ("What is the weather in Singapore today?", "Singapore"),
    ("is it going to rain tomorrow in Kuala Lumpur", "Kuala Lumpur"),
    ("weather in singapore today", "Singapore"),
    ("do I need a coat tonight", None),
    ("should I bring an umbrella to work", None),
])
def test_extract_location(query, location):
    assert extract_location(query) == location


def test_extract_search_query_strips_filler():
    assert extract_search_query("Can you search for the best ramen in Tokyo?") == "the best ramen in Tokyo"


# ------------------------------------------------------------------ mocks + registry
def test_weather_is_deterministic():
    assert weather.run("Oslo").output == weather.run("oslo ").output.replace("oslo ", "Oslo")


def test_registry_has_three_tools():
    assert set(build_registry()) == {"calculator", "weather", "web_search"}


def test_a_crashing_tool_returns_failed_result():
    from src.tools.base import Tool
    def boom(_):
        raise RuntimeError("bug")
    result = Tool("broken", "", lambda q: q, boom).run("x")
    assert result.ok is False and "RuntimeError" in result.error
