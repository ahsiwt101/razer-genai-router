"""Embedding vs reranker vs classifier on the hard queries.

Run:  python scripts/demo_reranker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

from src.router.classifier import ClassifierRouter
from src.router.embedding import EmbeddingRouter
from src.router.reranker import RerankRouter

embedder = EmbeddingRouter()
routers = [embedder, RerankRouter(embedder), ClassifierRouter(embedder)]

QUERIES = [
    ("How does rain form?", "llm"),
    ("What is the temperature of the sun?", "llm"),
    ("Is it gonna pour in Tokyo later?", "weather"),
    ("What's 15% of Tesla's stock price?", "web_search"),
    ("why does it rain more in the tropics", "llm"),
    ("convert 100 fahrenheit to celsius", "calculator"),
]

for query, expected in QUERIES:
    print(f"\n{query}   (expected: {expected})")
    for r in routers:
        d = r.route(query)
        mark = "OK" if d.route == expected else "XX"
        print(f"  {r.name:10s} {mark} {d.route:11s} {d.latency_ms:6.1f} ms   {d.reason}")
