"""takes a routing decision and produces the final answer.

missing arg [city: default_location; other tools: LLM]
tool failure: answer with LLM
LLM failure: network/quota error 
"""

import time
from dataclasses import asdict, dataclass, field

from .config import load_router_config
from .llm import LLMError
from .router.decision import CALCULATOR, CLARIFY, LLM, WEATHER, BaseRouter, RoutingDecision
from .tools.base import ToolResult
from .tools.registry import build_registry


@dataclass
class ExecutionResult:
    query: str
    answer: str
    decision: RoutingDecision
    tool_results: list[ToolResult] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)  # routing / tools / llm / total
    events: list[str] = field(default_factory=list)            # fallbacks and notes, in order

    @property
    def final_route(self) -> str:
        """What actually answered: may differ from decision.route if a fallback kicked in."""
        if self.decision.route == CLARIFY:
            return CLARIFY
        if any(e.startswith("fallback to llm") for e in self.events):
            return LLM
        return self.decision.route

    def to_dict(self) -> dict:
        return {
            "query": self.query, "answer": self.answer, "final_route": self.final_route,
            "decision": self.decision.to_dict(),
            "tool_results": [asdict(r) for r in self.tool_results],
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "events": self.events,
        }


class Executor:
    def __init__(self, router: BaseRouter, llm, tools=None, default_location: str | None = "use-config"):
        self.router = router
        self.llm = llm
        self.tools = tools or build_registry()
        self.default_location = (load_router_config().get("default_location")
                                 if default_location == "use-config" else default_location)

    def run(self, query: str) -> ExecutionResult:
        start = time.perf_counter()
        decision = self.router.route(query)
        result = ExecutionResult(query, "", decision, timings_ms={"routing": decision.latency_ms, "tools": 0.0, "llm": 0.0})

        if decision.route == CLARIFY:
            result.answer = decision.question
        elif decision.plan:
            result.answer = self._run_plan(query, decision, result)
        elif decision.route == LLM:
            result.answer = self._ask_llm(query, result)
        else:
            result.answer = self._run_tool(decision.route, query, result)

        result.timings_ms["total"] = (time.perf_counter() - start) * 1000
        return result

    # single tool
    def _run_tool(self, route: str, query: str, result: ExecutionResult) -> str:
        tool = self.tools[route]
        argument = tool.extract(query)
        if argument is None and route == WEATHER:
            if self.default_location is None:
                result.events.append("no location given; asked the user")
                return "Which city would you like the weather for?"
            argument = self.default_location
            result.events.append(f"no location given; used default location '{argument}'")
        if argument is None:
            result.events.append(f"fallback to llm: {route} could not extract an argument")
            return self._ask_llm(query, result)

        tool_result = self._call(tool, argument, result)
        if tool_result.ok:
            return tool_result.output
        if route == CALCULATOR and _is_math_error(tool_result.error):
            return f"I can't compute that: {tool_result.error}."  # a correct answer, not a failure
        result.events.append(f"fallback to llm: {route} failed ({tool_result.error})")
        return self._ask_llm(query, result, note=f"A live {route} lookup failed, so answer from general "
                                                  "knowledge and say that the information may be out of date.")

    # multi-step
    def _run_plan(self, query: str, decision: RoutingDecision, result: ExecutionResult) -> str:
        step_results: list[ToolResult] = []
        for i, step in enumerate(decision.plan):
            try:
                # "{0[value]} * 15 / 100" -> "409.81 * 15 / 100"
                argument = step["argument"].format(*[r.data for r in step_results])
            except (KeyError, IndexError):
                result.events.append(f"fallback to llm: plan step {i + 1} needed a value an earlier step didn't return")
                return self._ask_llm(query, result)
            tool_result = self._call(self.tools[step["tool"]], argument, result)
            if not tool_result.ok:
                result.events.append(f"fallback to llm: plan step {i + 1} ({step['tool']}) failed ({tool_result.error})")
                return self._ask_llm(query, result)
            step_results.append(tool_result)

        evidence = "; ".join(r.output for r in step_results[:-1])
        return f"{step_results[-1].output}   (using: {evidence})"

    # shared
    def _call(self, tool, argument: str, result: ExecutionResult) -> ToolResult:
        tool_result = tool.run(argument)
        result.tool_results.append(tool_result)
        result.timings_ms["tools"] += tool_result.latency_ms
        return tool_result

    def _ask_llm(self, query: str, result: ExecutionResult, note: str | None = None) -> str:
        prompt = f"{query}\n\n({note})" if note else query
        start = time.perf_counter()
        try:
            answer = self.llm.generate(prompt)
        except LLMError as e:
            result.events.append(f"llm unavailable: {e}")
            answer = "Sorry, I couldn't reach the language model right now. Please try again in a moment."
        result.timings_ms["llm"] += (time.perf_counter() - start) * 1000
        return answer


def _is_math_error(error: str | None) -> bool:
    return bool(error) and any(s in error for s in ("division by zero", "too large", "overflow"))
