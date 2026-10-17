"""multi-step planning 
for queries that require more than two tools 
"""

import re

from .decision import CALCULATOR, WEATHER, WEB_SEARCH, RoutingDecision

# suggests that the thing being computed on must be looked up first.
_LIVE_VALUE = re.compile(r"\b(price|cost|rate|value|worth|population|salary|market cap|revenue|gdp)\b", re.I)

_MATH_OVER_LOOKUP = re.compile(
    r"(?:(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)\s+of"
    r"|(?P<half>half)(?:\s+of)?"
    r"|(?P<double>double|twice)"
    r"|(?P<triple>triple)"
    r"|(?P<times>\d+(?:\.\d+)?)\s+times)"
    r"\s+(?P<entity>(?:the\s+)?[^\d?!.]+)",
    re.I,
)

_PLACE = r"[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*"
_WEATHER_COMPARE = [
    # "how much warmer is Singapore than London"
    re.compile(r"(?P<cmp>warmer|colder|hotter|cooler)\s+is\s+(?:it\s+in\s+)?(?P<a>" + _PLACE + r")"
               r"\s+than\s+(?:in\s+)?(?P<b>" + _PLACE + r")"),
    # "is London colder than Paris"
    re.compile(r"\bis\s+(?:it\s+)?(?:in\s+)?(?P<a>" + _PLACE + r")\s+(?P<cmp>warmer|colder|hotter|cooler)"
               r"\s+than\s+(?:in\s+)?(?P<b>" + _PLACE + r")"),
]


def plan(query: str) -> RoutingDecision | None:
    """return a multi-step decision, or None if the query doesn't match a known pattern."""
    m = _MATH_OVER_LOOKUP.search(query)
    if m and _LIVE_VALUE.search(m["entity"]):
        entity = re.sub(r"^the\s+", "", m["entity"].strip(), flags=re.I)
        if m["pct"]:
            expression = f"{m['pct']} / 100 * {{0[value]}}"
        elif m["half"]:
            expression = "{0[value]} / 2"
        elif m["double"]:
            expression = "{0[value]} * 2"
        elif m["triple"]:
            expression = "{0[value]} * 3"
        else:
            expression = f"{{0[value]}} * {m['times']}"
        steps = [{"tool": WEB_SEARCH, "argument": entity},
                 {"tool": CALCULATOR, "argument": expression}]
        return RoutingDecision(
            WEB_SEARCH, 0.9, f"Multi-step: look up '{entity}', then calculate on the result",
            stage="planner", plan=steps)

    m = next((m for p in _WEATHER_COMPARE if (m := p.search(query))), None)
    if m:
        # "warmer"/"hotter": a minus b. "colder"/"cooler": b minus a. A positive answer = "yes, by this much".
        first, second = (0, 1) if m["cmp"].lower() in ("warmer", "hotter") else (1, 0)
        steps = [{"tool": WEATHER, "argument": m["a"]},
                 {"tool": WEATHER, "argument": m["b"]},
                 {"tool": CALCULATOR, "argument": f"{{{first}[temp_c]}} - {{{second}[temp_c]}}"}]
        return RoutingDecision(
            WEATHER, 0.9, f"Multi-step: weather in {m['a']} and {m['b']}, then compare",
            stage="planner", plan=steps)
    return None
