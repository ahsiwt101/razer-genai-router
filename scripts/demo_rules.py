"""
Run:  python scripts/demo_rules.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

from src.router.rules import RuleRouter

# (query, what a human would say is correct)
QUERIES = [
    # --- examples from assignment
    ("What is 245 * 18?", "calculator"),
    ("What is the weather in Singapore today?", "weather"),
    ("Who won the latest Formula 1 race?", "web_search"),
    ("Explain gradient descent.", "llm"),
    # --- rules shld handle these type of queries
    ("What's 15% of 80?", "calculator"),
    ("Will I need an umbrella tomorrow?", "weather"),
    ("How does rain form?", "llm"),
    ("Is it hot in Bangkok right now?", "weather"),
    # --- need embeddings 
    ("What's two hundred times three?", "calculator"),
    ("Is it gonna pour in Tokyo later?", "weather"),
    ("How's Nvidia doing on the market?", "web_search"),
    ("What is the temperature of the sun?", "llm"),
    ("What's 15% of Tesla's stock price?", "web_search -> calculator"),
]

router = RuleRouter()
print(f"{'QUERY':42s} {'ROUTE':11s} {'CONF':>5s}  {'STAGE':14s} {'EXPECTED':24s} HINTS")
print("-" * 125)
for query, expected in QUERIES:
    d = router.route(query)
    mark = "OK " if d.route == expected else "XX "
    print(f"{mark}{query:39s} {d.route:11s} {d.confidence:5.2f}  {d.stage:14s} {expected:24s} {d.hints}")

print(f"\nLatency of last call: {d.latency_ms:.3f} ms")
print("\nExample structured decision:")
print(router.route("What is 245 * 18?").to_dict())
