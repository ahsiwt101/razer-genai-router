"""
Stage 1: Rules
rule fires only when almost certainly right. if not then we move ahead to embedding stage. 
other signals are recorded as hints, to be used later. 
"""

import re

from .decision import CALCULATOR, LLM, WEATHER, WEB_SEARCH, BaseRouter, RoutingDecision


def normalize(query: str) -> str:
    """Lowercase, unify maths symbols, and collapse whitespace so patterns stay simple."""
    q = query.lower().replace("×", "*").replace("÷", "/").replace("’", "'")
    return " ".join(q.split())


# CALCULATOR 
# two numbers joined by operator and that '-' and 'x' must have spaces around them 
# safety measure against words like 'covid-19' to not trigger calc
_ARITH = re.compile(r"\d\s*[+*/^]\s*\(?-?\d|\d\s+[-x]\s+\(?\d")

# handling case of maths written in words  
_WORD_ARITH = re.compile(
    r"\d\s*(plus|minus|times|multiplied by|divided by|to the power of|mod)\s*\d"
)

# handling case of function applied to numbers 
_MATH_FUNC = re.compile(r"\b(square root|sqrt|cube root|log|ln|factorial)\s*(of\s*)?\(?\d")

# handling case of percentages applied to numbers 
_PERCENT_OF = re.compile(r"\d+(\.\d+)?\s*(%|percent)\s+of\s+\d")

# WEATHER 

# words that almost always mean weather
_WEATHER_CORE = re.compile(
    r"\b(weather|forecast|rain(ing|y)?|snow(ing|y)?|umbrella|humid(ity)?|thunderstorms?|drizzl(e|ing))\b"
)
# words that cld mean weather in context 
_WEATHER_SOFT = re.compile(
    r"\b(temperature|degrees|hot|cold|warm|chilly|sunny|cloudy|windy|jacket|sunscreen)\b"
)

_PLACE = re.compile(r"\b(in|at)\s+[a-z]{3,}")
_TIME_NOW = re.compile(r"\b(today|tonight|tomorrow|now|right now|this (morning|afternoon|evening|week|weekend)|later)\b")

# WEB SEARCH
_RECENCY = re.compile(
    r"\b(latest|current(ly)?|today'?s|yesterday|this (week|month|year|season)|right now|"
    r"recent(ly)?|breaking|upcoming|news|20[2-9]\d)\b"
)
# things that changes across time - needs to be most likely looked up
_EVENT = re.compile(
    r"\b(news|won|wins?|winner|scores?|results?|prices?|stocks?|elections?|released?|launch(ed)?|"
    r"announced?|ceo|president|prime minister|champions?|match|race|standings)\b"
)
_EXPLICIT_SEARCH = re.compile(r"\b(search (for|the web|online)|google|look up|look it up)\b")

# HINTS ONLY
# questions asking for explaination is prob llm 
_EXPLANATORY = re.compile(
    r"^(explain|describe|define|why|how (does|do|did|can|would)|what (causes|is the difference|does .* mean))\b"
)


class RuleRouter(BaseRouter):
    name = "rules"

    def match(self, query: str) -> tuple[RoutingDecision | None, list[str]]:
        """ Return (decision, hints) 
        decision is None if ambiguity - moves on to embedding with hints 
        """

        q = normalize(query)
        hints = self._hints(q)

        # order: calculator first, then weather, then web search. [cal is most precise]
        for check in (self._calculator, self._weather, self._web_search):
            result = check(q, hints)
            if result is not None:
                route, confidence, reason = result
                return RoutingDecision(route, confidence, reason, stage="rules", hints=hints), hints
        return None, hints

    def _route(self, query: str) -> RoutingDecision:
        """rules-only baseline 
        """
        decision, hints = self.match(query)
        if decision is not None:
            return decision
        if "weather_word" in hints and "explanatory" not in hints:
            return RoutingDecision(WEATHER, 0.55, "weak signal: weather-related word", "rules_weak", hints=hints)
        if "recency" in hints:
            return RoutingDecision(WEB_SEARCH, 0.55, "weak signal: asks about recent information", "rules_weak", hints=hints)
        return RoutingDecision(LLM, 0.5, "no rule matched; defaulting to direct LLM", "rules_default", hints=hints)

    # individual checks
    # each returns (route, confidence, reason) or None if it doesnt mathc.
    # confidences are hand-set currently to reflect precision of each rule 

    @staticmethod
    def _calculator(q: str, hints: list[str]):
        if _ARITH.search(q):
            return CALCULATOR, 0.97, "Contains an arithmetic expression between numbers"
        if _WORD_ARITH.search(q):
            return CALCULATOR, 0.95, "Contains arithmetic written in words between numbers"
        if _MATH_FUNC.search(q):
            return CALCULATOR, 0.95, "Applies a maths function to a number"
        if _PERCENT_OF.search(q):
            return CALCULATOR, 0.95, "Asks for a percentage of a number"
        return None

    @staticmethod
    def _weather(q: str, hints: list[str]):
        if "explanatory" in hints:  # "how does rain form?" is a science question, not a forecast
            return None
        if _WEATHER_CORE.search(q):
            return WEATHER, 0.93, "Contains an unambiguous weather word"
        if _WEATHER_SOFT.search(q) and _PLACE.search(q) and _TIME_NOW.search(q):
            return WEATHER, 0.85, "Weather-like word together with a place and a current time"
        return None

    @staticmethod
    def _web_search(q: str, hints: list[str]):
        if _EXPLICIT_SEARCH.search(q):
            return WEB_SEARCH, 0.95, "Explicitly asks to search or look something up"
        if _RECENCY.search(q) and _EVENT.search(q):
            return WEB_SEARCH, 0.90, "Asks about a recent or changing real-world event"
        return None

    @staticmethod
    def _hints(q: str) -> list[str]:
        hints = []
        if _WEATHER_CORE.search(q) or _WEATHER_SOFT.search(q):
            hints.append("weather_word")
        if _RECENCY.search(q) or _TIME_NOW.search(q):
            hints.append("recency")
        if _PLACE.search(q):
            hints.append("has_place")
        if re.search(r"\d", q):
            hints.append("has_number")
        if _EXPLANATORY.search(q):
            hints.append("explanatory")
        return hints
