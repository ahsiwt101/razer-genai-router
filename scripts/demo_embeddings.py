"""
Run:  python scripts/demo_embeddings.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

import numpy as np

from src.router.embedding import EmbeddingRouter
from src.router.rules import RuleRouter

print("Loading model (first run downloads ~90MB)...")
t = time.perf_counter()
router = EmbeddingRouter()
print(f"Startup: {time.perf_counter() - t:.1f}s  (load model + embed {len(router.texts)} examples, happens once)\n")

# what is an embedding?
print("=" * 100)
print("1. WHAT AN EMBEDDING LOOKS LIKE")
v = router.embed(["Is it gonna pour in Tokyo later?"])[0]
print(f"   'Is it gonna pour in Tokyo later?'  ->  {len(v)} numbers: [{', '.join(f'{x:.3f}' for x in v[:6])}, ...]")
print(f"   length of the vector = {np.linalg.norm(v):.3f}  (normalised to 1, so dot product = cosine similarity)\n")

#  similarity between sentences
print("=" * 100)
print("2. COSINE SIMILARITY BETWEEN PAIRS (1.0 = same meaning)")
pairs = [
    ("Is it gonna pour in Tokyo later?", "will it rain tomorrow"),          # same meaning, different words
    ("Is it gonna pour in Tokyo later?", "explain gradient descent"),       # unrelated
    ("What's two hundred times three?", "what is 37 * 412"),                # words vs digits
    ("How hot is the sun?", "how hot will it be in Dubai today"),           # surface trap: same words, different intent
]
for a, b in pairs:
    va, vb = router.embed([a, b])
    print(f"   {float(va @ vb):.2f}   '{a}'  vs  '{b}'")
print()

#  routing the demo queries
print("=" * 100)
print("3. ROUTING: rules vs embeddings  (none of these queries are in routes.yaml)")
QUERIES = [
    ("What is 245 * 18?", "calculator"),
    ("What is the weather in Singapore today?", "weather"),
    ("Who won the latest Formula 1 race?", "web_search"),
    ("Explain gradient descent.", "llm"),
    ("What's 15% of 80?", "calculator"),
    ("Will I need an umbrella tomorrow?", "weather"),
    ("How does rain form?", "llm"),
    ("Is it hot in Bangkok right now?", "weather"),
    ("What's two hundred times three?", "calculator"),
    ("Is it gonna pour in Tokyo later?", "weather"),
    ("How's Nvidia doing on the market?", "web_search"),
    ("What is the temperature of the sun?", "llm"),
    ("What's 15% of Tesla's stock price?", "web_search"),  # really web_search -> calculator (multi-step)
]
rules = RuleRouter()
latencies = []
for query, expected in QUERIES:
    r = rules.route(query)
    e = router.route(query)
    latencies.append(e.latency_ms)
    ranked = sorted(e.scores.items(), key=lambda kv: kv[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1]
    print(f"\n   {query}   (expected: {expected})")
    print(f"     rules:      {'OK' if r.route == expected else 'XX'}  {r.route}")
    print(f"     embeddings: {'OK' if e.route == expected else 'XX'}  {e.route}   margin {margin:.2f}   "
          + "  ".join(f"{k}={v:.2f}" for k, v in ranked))
    text, label, sim = router.nearest(query, n=1)[0]
    print(f"     closest example: '{text}' [{label}] {sim:.2f}")

print("\n" + "=" * 100)
print(f"Embedding routing latency: avg {np.mean(latencies):.1f} ms, max {np.max(latencies):.1f} ms per query "
      f"(the first call is slower while the model warms up)")
