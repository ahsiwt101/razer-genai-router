""" for choosingnGEMINI_MODEL in .env.

Run:  python scripts/benchmark_llm.py
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

from src.llm import GeminiClient, LLMError

CANDIDATES = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]
PROMPTS = ["Explain gradient descent in two sentences.", "What's the weather in Tokyo right now?"]

for model in CANDIDATES:
    try:
        llm = GeminiClient(model)
        times, answers = [], []
        for prompt in PROMPTS * 2:  # each prompt twice; the first call to a model can be slower
            start = time.perf_counter()
            answers.append(llm.generate(prompt))
            times.append((time.perf_counter() - start) * 1000)
        print(f"\n{model}: median {statistics.median(times):.0f} ms  (min {min(times):.0f}, max {max(times):.0f})")
        print(f"   gradient descent -> {answers[0][:150]!r}")
        print(f"   live weather     -> {answers[1][:150]!r}")
    except LLMError as e:
        print(f"\n{model}: FAILED  {str(e)[:150]}")
