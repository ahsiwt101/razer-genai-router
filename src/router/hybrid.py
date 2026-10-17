"""hybrid router 

  1. PLAN           for known multi-step patterns ("15% of Tesla's stock price")       
  2. OUT OF SCOPE   requests for actions no tool can do ("book me a flight")        
  3. RULES          high-precision patterns (accept or abstain)                       
  4. CLASSIFIER     probabilities per route + calibrated confidence p
  5. AMBIGUOUS      a. feasibility: tools that can't extract an argument are ruled out
                    b. (optional) reranker breaks the tie between the top two
                    c. choose the action with the lowest expected costs: a route, or ask the user
"""

import re

from ..config import load_router_config
from ..tools.extractors import extract_expression, extract_search_query
from .calibration import load_calibrators
from .classifier import ClassifierRouter
from .decision import CALCULATOR, CLARIFY, LLM, WEATHER, WEB_SEARCH, BaseRouter, RoutingDecision
from .embedding import EmbeddingRouter
from .planner import plan
from .reranker import Reranker
from .rules import RuleRouter

LIVE_TOOLS = {WEATHER, WEB_SEARCH}

# handling requests for unexpected actions
_ACTION_REQUEST = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+)?(book|order|buy|purchase|send|email|text|call|set|schedule|"
    r"remind|play|turn|open|cancel|reserve|pay|transfer)\b", re.I)


_EXTRACTORS = {CALCULATOR: extract_expression, WEB_SEARCH: extract_search_query}

_FRIENDLY = {CALCULATOR: "a calculation", WEATHER: "the current weather",
             WEB_SEARCH: "the latest information from the web", LLM: "a general answer"}


class HybridRouter(BaseRouter):
    name = "hybrid"

    def __init__(self, embedder: EmbeddingRouter | None = None, config: dict | None = None,
                 use_reranker: bool | None = None, accept_threshold: float | None = None):
        self.config = config or load_router_config()
        self.accept_threshold = (self.config["accept_threshold"] if accept_threshold is None
                                 else accept_threshold)
        self.rules = RuleRouter()
        self.embedder = embedder or EmbeddingRouter()
        self.classifier = ClassifierRouter(self.embedder)
        self.calibrator = load_calibrators()["classifier"]
        use_reranker = self.config["use_reranker"] if use_reranker is None else use_reranker
        self.reranker = Reranker(self.embedder) if use_reranker else None
        if use_reranker:
            self.name = "hybrid+rerank"

    #  main routing logic 
    def _route(self, query: str) -> RoutingDecision:
        multi_step = plan(query)
        if multi_step is not None:
            return multi_step

        if _ACTION_REQUEST.search(query.strip()):
            return RoutingDecision(LLM, 0.9, "Asks for an action no available tool can perform; "
                                   "the LLM will explain what it can do instead",
                                   stage="out_of_scope", fallback_used=True)

        decision, hints = self.rules.match(query)
        if decision is not None:
            return decision

        probs = self.classifier.score(query)
        p = self.calibrator.predict(probs)  # P(the classifier's top route is correct)
        top = max(probs, key=probs.get)
        feasible = {route: self._feasible(route, query) for route in probs}

        if p >= self.accept_threshold and feasible[top]:
            return RoutingDecision(top, p, f"Classifier chose {top} (probability {probs[top]:.2f}, "
                                   f"calibrated confidence {p:.2f})",
                                   stage="classifier", scores=probs, hints=hints)

        return self._resolve_ambiguous(query, probs, p, feasible, hints)

    def _resolve_ambiguous(self, query, probs, p, feasible, hints) -> RoutingDecision:
        notes, adjusted = [], dict(probs)
        top = max(probs, key=probs.get)
        why = (f"confidence {p:.2f} < {self.accept_threshold:.2f}" if p < self.accept_threshold
               else f"top route {top} can't extract an argument")

        for route, ok in feasible.items():
            if not ok and adjusted[route] > 0:
                notes.append(f"ruled out {route} (nothing to extract)")
                if route == CALCULATOR:
                    adjusted[LLM] += adjusted[route]
                adjusted[route] = 0.0
        total = sum(adjusted.values())  # since llm is always feasible, so total > 0
        adjusted = {route: v / total for route, v in adjusted.items()}

        # reranker
        if self.reranker is not None:
            first, second = sorted(adjusted, key=adjusted.get, reverse=True)[:2]
            rerank_scores, _ = self.reranker.rerank(query)
            if first in rerank_scores and second in rerank_scores and rerank_scores[second] > rerank_scores[first]:
                adjusted[first], adjusted[second] = adjusted[second], adjusted[first]
                notes.append(f"reranker preferred {second} over {first}")

        # lowest expected cost among
        costs = {action: self.expected_cost(action, adjusted) for action in [*adjusted, CLARIFY]}
        action = min(costs, key=costs.get)

        question = None
        if action == CLARIFY:
            a, b = sorted(adjusted, key=adjusted.get, reverse=True)[:2]
            question = f"Just to check: do you want {_FRIENDLY[a]} or {_FRIENDLY[b]}?"

        reason = (f"Unsure ({why}); lowest expected cost: {action} ({costs[action]:.2f})"
                  + "".join(f"; {n}" for n in notes))

        confidence = p if action == CLARIFY else adjusted[action]
        return RoutingDecision(action, confidence, reason, stage="fallback", scores=adjusted, hints=hints,
                               fallback_used=True, question=question)

    # helpers
    @staticmethod
    def _feasible(route: str, query: str) -> bool:
        extractor = _EXTRACTORS.get(route)
        return extractor is None or extractor(query) is not None

    def cost(self, chosen: str, true: str) -> float:
        return mistake_cost(chosen, true, self.config["costs"])

    def expected_cost(self, action: str, probs: dict[str, float]) -> float:
        if action == CLARIFY:
            return self.config["costs"]["clarify"]
        return sum(p * self.cost(action, true) for true, p in probs.items())


def mistake_cost(chosen: str, true: str, costs: dict) -> float:
    """Cost of answering with `chosen` when `true` was right. Shared by the router (to decide)
    and the evaluation (to score), so both use exactly the same definition."""
    if chosen == true:
        return 0.0
    if chosen == LLM:
        return costs["missed_live_tool"] if true in LIVE_TOOLS else costs["missed_calculator"]
    if true == LLM:
        return costs["unneeded_tool"]
    return costs["wrong_tool"]
