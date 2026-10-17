"""Tests for the evaluation metrics. Run:  pytest -q"""
import pytest

from src.data import LabelledQuery
from src.evaluation import bootstrap_ci, paired_bootstrap_diff, routing_metrics
from src.router.decision import RoutingDecision

COSTS = {"missed_calculator": 2, "missed_live_tool": 1.5, "wrong_tool": 3, "unneeded_tool": 1, "clarify": 1.1}


def d(route):
    return RoutingDecision(route, 0.9, "", stage="test")


def test_tool_rates_follow_their_definitions():
    queries = [LabelledQuery("q1", "weather", "clear"), LabelledQuery("q2", "weather", "clear"),
               LabelledQuery("q3", "calculator", "clear"), LabelledQuery("q4", "llm", "no_tool")]
    m = routing_metrics(queries, [d("weather"), d("web_search"), d("llm"), d("calculator")], COSTS)
    assert m["accuracy"] == 0.25
    assert m["incorrect_tool_selection_rate"] == pytest.approx(1 / 3)  # q2: needed weather, got web_search
    assert m["missed_tool_rate"] == pytest.approx(1 / 3)               # q3: needed calculator, got llm
    assert m["unnecessary_tool_rate"] == 1.0                           # q4: needed nothing, got a tool
    assert m["mean_mistake_cost"] == pytest.approx((0 + 3 + 2 + 1) / 4)


def test_acceptable_routes_count_as_correct():
    q = LabelledQuery("what's it like in Paris", "weather", "ambiguous", acceptable=["weather", "llm", "clarify"])
    assert routing_metrics([q], [d("clarify")], COSTS)["accuracy"] == 1.0


def test_bootstrap_interval_contains_the_mean():
    lo, hi = bootstrap_ci([1.0] * 8 + [0.0] * 2)
    assert lo <= 0.8 <= hi


def test_identical_routers_are_not_significantly_different():
    flags = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert paired_bootstrap_diff(flags, flags)["significant"] is False
