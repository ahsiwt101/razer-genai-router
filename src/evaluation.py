"""evaluations metrics

Definitions:
  accuracy                 predicted route is in the query's acceptable set (or equals its label)
  strict accuracy          predicted route equals the single labelled route
  incorrect tool selection of queries that NEED a tool: share sent to a DIFFERENT tool
  missed tool              of queries that NEED a tool: share answered by the LLM instead
  unnecessary tool call    of queries that need NO tool (no tool is even acceptable): share sent to a tool
  clarify / fallback rate  share of queries where the router asked the user / used the fallback
  mean mistake cost        average cost per query, using the SAME cost matrix the router optimises
"""
import random

import numpy as np

from .data import LabelledQuery
from .router.decision import CLARIFY, LLM, RoutingDecision
from .router.hybrid import mistake_cost

TOOLS = {"calculator", "weather", "web_search"}


def needs_tool(q: LabelledQuery) -> bool:
    return q.route in TOOLS


def needs_no_tool(q: LabelledQuery) -> bool:
    return not set(q.acceptable or [q.route]) & TOOLS


def query_cost(q: LabelledQuery, predicted: str, costs: dict) -> float:
    if q.is_correct(predicted):
        return 0.0
    if predicted == CLARIFY:
        return costs["clarify"]
    return mistake_cost(predicted, q.route, costs)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def latency_summary(ms: list[float]) -> dict:
    if not ms:
        return {"p50": None, "p95": None, "mean": None}
    return {"p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)), "mean": float(np.mean(ms))}


def routing_metrics(queries: list[LabelledQuery], decisions: list[RoutingDecision], costs: dict) -> dict:
    preds = [d.route for d in decisions]
    correct = [q.is_correct(p) for q, p in zip(queries, preds)]
    tool_needed = [(q, p, c) for q, p, c in zip(queries, preds, correct) if needs_tool(q)]
    no_tool = [(q, p) for q, p in zip(queries, preds) if needs_no_tool(q)]

    categories = sorted({q.category for q in queries})
    per_category = {cat: rate(sum(c for q, c in zip(queries, correct) if q.category == cat),
                              sum(q.category == cat for q in queries)) for cat in categories}

    multi = [(q, d) for q, d in zip(queries, decisions) if q.plan]
    plan_correct = sum([s["tool"] for s in d.plan] == q.plan for q, d in multi)

    stages = {}
    for q, d, c in zip(queries, decisions, correct):
        s = stages.setdefault(d.stage, {"count": 0, "correct": 0, "latencies": []})
        s["count"] += 1
        s["correct"] += c
        s["latencies"].append(d.latency_ms)
    by_stage = {name: {"count": s["count"], "accuracy": s["correct"] / s["count"],
                       "latency_p50_ms": float(np.percentile(s["latencies"], 50))} for name, s in stages.items()}

    by_route = {}
    for d in decisions:
        by_route.setdefault(d.route, []).append(d.latency_ms)

    return {
        "n": len(queries),
        "correct": int(sum(correct)),
        "accuracy": float(np.mean(correct)),
        "strict_accuracy": float(np.mean([p == q.route for q, p in zip(queries, preds)])),
        "incorrect_tool_selection_rate": rate(sum(p in TOOLS and not c for _, p, c in tool_needed), len(tool_needed)),
        "missed_tool_rate": rate(sum(p == LLM and not c for _, p, c in tool_needed), len(tool_needed)),
        "unnecessary_tool_rate": rate(sum(p in TOOLS for _, p in no_tool), len(no_tool)),
        "clarify_rate": float(np.mean([p == CLARIFY for p in preds])),
        "fallback_rate": float(np.mean([d.fallback_used for d in decisions])),
        "mean_mistake_cost": float(np.mean([query_cost(q, p, costs) for q, p in zip(queries, preds)])),
        "multi_step_plan_accuracy": rate(plan_correct, len(multi)),
        "per_category_accuracy": per_category,
        "by_stage": by_stage,
        "latency_ms": latency_summary([d.latency_ms for d in decisions]),
        "latency_by_route_ms": {route: latency_summary(ms) for route, ms in by_route.items()},
        "correct_flags": correct,  # kept for bootstrap comparisons; stripped before saving
    }


def bootstrap_ci(values: list[float], n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for a mean, by resampling the queries with replacement.

    Intuition: "if we'd happened to write a different 63 test queries of the same kind,
    how much could this number move?" Small test sets -> wide intervals.
    """
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


def paired_bootstrap_diff(a: list[float], b: list[float], n_resamples: int = 2000, seed: int = 0) -> dict:
    """Is router A really better than router B, or is it luck? Resample the SAME queries for both,
    so the comparison isn't drowned out by which queries happened to be easy or hard."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_resamples)], diffs[int(0.975 * n_resamples) - 1]
    return {"mean_diff": (sum(a) - sum(b)) / n, "ci_low": lo, "ci_high": hi,
            "significant": lo > 0 or hi < 0}  # interval excludes zero -> unlikely to be luck


def confusion_matrix(queries: list[LabelledQuery], decisions: list[RoutingDecision]) -> dict[str, dict[str, int]]:
    """Rows: labelled route. Columns: predicted route."""
    matrix: dict[str, dict[str, int]] = {}
    for q, d in zip(queries, decisions):
        row = matrix.setdefault(q.route, {})
        row[d.route] = row.get(d.route, 0) + 1
    return matrix
