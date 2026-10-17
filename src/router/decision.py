"""
every router returns a RoutingDecision, which is a dataclass with the following fields:
- route: str, one of "llm", "calculator", "weather", "web_search"
- confidence: float, 0..1, how sure the router is
- reason: str, human-readable explanation
- stage: str, which part of the system decided, e.g. "rules", "embedding", "fallback"
- scores: dict[str, float], score per route (empty for rules)
- hints: list[str], weak signals noticed along the way
- fallback_used: bool, True if we were unsure and fell back
- latency_ms: float, time spent routing, filled in by BaseRouter

this allows us to compare by just swapping routers later on 
"""
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

# Route names. Kept as constants so a typo like "calulator" fails loudly instead of silently.
LLM = "llm"
CALCULATOR = "calculator"
WEATHER = "weather"
WEB_SEARCH = "web_search"


@dataclass
class RoutingDecision:
    route: str                  # where the query goes: "llm", "calculator", "weather", "web_search"
    confidence: float           # 0..1, how sure the router is
    reason: str                 # human-readable explanation
    stage: str                  # which part of the system decided, e.g. "rules", "embedding", "fallback"
    scores: dict[str, float] = field(default_factory=dict)  # score per route (empty for rules)
    hints: list[str] = field(default_factory=list)          # weak signals noticed along the way
    fallback_used: bool = False                             # True if we were unsure and fell back
    latency_ms: float = 0.0                                 # time spent routing, filled in by BaseRouter

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(d["confidence"], 3)
        d["latency_ms"] = round(d["latency_ms"], 3)
        d["scores"] = {k: round(v, 3) for k, v in d["scores"].items()}
        return d


class BaseRouter(ABC):
    """
    the base class for all routers. 
    route() written here once and handles timing for every router. 
    """

    name = "base"

    def route(self, query: str) -> RoutingDecision:
        start = time.perf_counter()
        decision = self._route(query)
        decision.latency_ms = (time.perf_counter() - start) * 1000
        return decision

    @abstractmethod
    def _route(self, query: str) -> RoutingDecision:
        ...
