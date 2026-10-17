"""Tests for the hybrid router, planner and executor (mock LLM, no network). Run:  pytest -q"""
import pytest

from src.executor import Executor
from src.llm import LLMError, MockLLM
from src.router.decision import CALCULATOR, CLARIFY, LLM, WEATHER, WEB_SEARCH
from src.router.hybrid import HybridRouter
from src.router.planner import plan
from src.tools.base import Tool, ToolResult
from src.tools.registry import build_registry


@pytest.fixture(scope="module")
def router():
    return HybridRouter()


# ------------------------------------------------------------------ routing
@pytest.mark.parametrize("query, route, stage", [
    ("What is 245 * 18?", CALCULATOR, "rules"),
    ("Explain gradient descent.", LLM, "classifier"),
    ("What's 15% of Tesla's stock price?", WEB_SEARCH, "planner"),
    ("what's 10% of the current price of gold", WEB_SEARCH, "planner"),  # planner must beat the web-search rule
    ("book me a flight to Bali", LLM, "out_of_scope"),
    ("how do you calculate compound interest", LLM, "fallback"),          # calculator ruled out: no numbers
])
def test_hybrid_routes(router, query, route, stage):
    d = router.route(query)
    assert (d.route, d.stage) == (route, stage)


def test_tool_vs_tool_tie_asks_the_user(router):
    d = router.route("what about tomorrow")
    assert d.route == CLARIFY and d.question


def test_cost_function(router):
    assert router.cost(LLM, LLM) == 0
    assert router.cost(LLM, CALCULATOR) > router.cost(LLM, WEATHER)  # silent wrong maths is worst
    assert router.cost(WEATHER, WEB_SEARCH) > router.cost(WEATHER, LLM)  # wrong tool > unneeded tool


def test_planner_ignores_plain_arithmetic():
    assert plan("What is 15% of 80?") is None


# ------------------------------------------------------------------ executor
@pytest.fixture(scope="module")
def executor(router):
    return Executor(router, MockLLM(), default_location="Singapore")


def test_executor_runs_a_tool(executor):
    result = executor.run("What is 245 * 18?")
    assert "4,410" in result.answer and result.final_route == CALCULATOR


def test_executor_runs_a_multi_step_plan(executor):
    result = executor.run("What's 15% of Tesla's stock price?")
    assert [r.tool for r in result.tool_results] == [WEB_SEARCH, CALCULATOR]
    assert all(r.ok for r in result.tool_results)


def test_executor_uses_default_location(executor):
    result = executor.run("do I need a coat tonight")
    assert "Singapore" in result.answer and any("default location" in e for e in result.events)


def test_division_by_zero_is_answered_not_escalated(executor):
    result = executor.run("what is 7 / 0")
    assert "division by zero" in result.answer and result.final_route == CALCULATOR


def test_failing_tool_falls_back_to_llm(router):
    tools = build_registry()
    def broken(_):
        return ToolResult(WEATHER, ok=False, error="API timeout")
    tools[WEATHER] = Tool(WEATHER, "", tools[WEATHER].extract, broken)
    result = Executor(router, MockLLM(), tools=tools).run("What is the weather in Singapore today?")
    assert result.final_route == LLM and "[mock LLM]" in result.answer


def test_llm_outage_is_reported_not_raised(router):
    class DownLLM:
        model = "down"
        def generate(self, prompt):
            raise LLMError("network unreachable")
    result = Executor(router, DownLLM()).run("Explain gradient descent.")
    assert "couldn't reach" in result.answer and any("llm unavailable" in e for e in result.events)
