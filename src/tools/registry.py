"""one place that knows which tools exist.
"""
from functools import partial

from . import calculator, weather, web_search
from .base import Tool
from .extractors import extract_expression, extract_location, extract_search_query

SIMULATED_LATENCY_MS = {"calculator": 0, "weather": 150, "web_search": 400}

def build_registry(simulate_latency: bool = False) -> dict[str, Tool]:
    def latency(name: str) -> float:
        return SIMULATED_LATENCY_MS[name] if simulate_latency else 0

    tools = [
        Tool("calculator", "Safely evaluates arithmetic expressions.",
             extract_expression, calculator.run),
        Tool("weather", "Current weather for a location (mocked).",
             extract_location, partial(weather.run, simulated_latency_ms=latency("weather"))),
        Tool("web_search", "Searches the web for current information (mocked).",
             extract_search_query, partial(web_search.run, simulated_latency_ms=latency("web_search"))),
    ]
    return {tool.name: tool for tool in tools}
