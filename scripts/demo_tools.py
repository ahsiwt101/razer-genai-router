"""Extract the argument, then run the tool, for a few queries per tool, including the failures.

Run:  python scripts/demo_tools.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

from src.tools.registry import build_registry

tools = build_registry(simulate_latency=True)

CASES = [
    ("calculator", "What's two hundred times three?"),
    ("calculator", "convert 100 fahrenheit to celsius"),
    ("calculator", "if I split a 240 dollar bill among 6 people how much does each pay"),
    ("calculator", "What's 15% of Tesla's stock price?"),     # no number to compute with: needs a search first
    ("calculator", "what is 7 / 0"),                          # extractor rejects it before the tool runs
    ("weather", "Is it gonna pour in Tokyo later?"),
    ("weather", "weather check for Hong Kong"),
    ("weather", "do I need a coat tonight"),                  # no location: ask the user or use a default
    ("web_search", "Can you search for the best ramen in Tokyo?"),
    ("web_search", "what's the current price of gold"),       # returns a [mock] number for multi-step demos
]

for tool_name, query in CASES:
    tool = tools[tool_name]
    argument = tool.extract(query)
    print(f"\n[{tool_name}] {query}")
    if argument is None:
        print("   extract -> None   (feasibility check fails: this tool can't handle the query as written)")
        continue
    result = tool.run(argument)
    print(f"   extract -> {argument!r}")
    print(f"   run     -> ok={result.ok}  {result.output or result.error}   ({result.latency_ms:.1f} ms)")
