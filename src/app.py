"""Command-line app: route a query, run it, show the decision, the answer and the timings.

  python -m src.app "What is 245 * 18?"          one query
  python -m src.app                              interactive mode (type queries, 'quit' to exit)

Options:
  --mock-llm           don't call Gemini (no key/network needed)
  --rerank             enable the cross-encoder tie-breaker for ambiguous queries
  --simulate-latency   make mock tools sleep like real APIs (150 ms weather, 400 ms search)
  --json               print the full result as JSON
"""
import argparse
import json
import time

from .executor import ExecutionResult, Executor
from .llm import GeminiClient, LLMError, MockLLM
from .router.hybrid import HybridRouter
from .tools.registry import build_registry


def build_executor(mock_llm: bool = False, rerank: bool = False, simulate_latency: bool = False) -> Executor:
    router = HybridRouter(use_reranker=rerank)
    llm = MockLLM() if mock_llm else GeminiClient()
    return Executor(router, llm, tools=build_registry(simulate_latency=simulate_latency))


def show(result: ExecutionResult) -> None:
    d = result.decision
    # The structured routing decision, in the format the assignment asks for.
    print(json.dumps({"route": d.route, "confidence": round(d.confidence, 2), "reason": d.reason}, indent=2))
    print(f"stage: {d.stage}" + ("   (fallback)" if d.fallback_used else ""))
    if d.plan:
        print("plan:  " + "  ->  ".join(f"{s['tool']}({s['argument']})" for s in d.plan))
    for event in result.events:
        print(f"note:  {event}")
    print(f"\nanswer: {result.answer}\n")
    t = result.timings_ms
    print(f"timing: routing {t['routing']:.1f} ms | tools {t['tools']:.1f} ms | llm {t['llm']:.0f} ms | "
          f"total {t['total']:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a query to the LLM or a tool, without native tool calling.")
    parser.add_argument("query", nargs="*", help="the query (omit for interactive mode)")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--simulate-latency", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print("Loading models...", end=" ", flush=True)
    start = time.perf_counter()
    try:
        executor = build_executor(args.mock_llm, args.rerank, args.simulate_latency)
    except LLMError as e:
        raise SystemExit(f"\n{e}\n(Or run with --mock-llm to skip the real LLM.)")
    print(f"ready in {time.perf_counter() - start:.1f}s (LLM: {executor.llm.model})\n")

    def handle(query: str) -> None:
        result = executor.run(query)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) if args.json else "", end="")
        if not args.json:
            show(result)

    if args.query:
        handle(" ".join(args.query))
        return
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"quit", "exit", "q"}:
            break
        if query:
            handle(query)
            print("-" * 80)


if __name__ == "__main__":
    main()
