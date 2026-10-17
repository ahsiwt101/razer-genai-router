"""Full evaluation on test set 

Run:  python eval/run_eval.py               (mock LLM: routing metrics are complete, LLM latency is 0)
      python eval/run_eval.py --real-llm    (Gemini: real end-to-end latency)

writes to eval/results/:
  report.md        all tables (the numbers in the README come from here)
  errors.md        every misrouted query with the stage and reason, for error analysis
  examples.md      full structured outputs for representative queries
  metrics.json     everything, machine-readable
  accuracy.png, latency.png, e2e_latency.png, reliability.png
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # so "src" can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so "plots" can be imported

from plots import grouped_hbar, reliability, stacked_hbar

from src.config import load_router_config
from src.data import EVAL_DIR, load_queries
from src.evaluation import bootstrap_ci, confusion_matrix, latency_summary, paired_bootstrap_diff, routing_metrics
from src.executor import Executor
from src.llm import GeminiClient, MockLLM
from src.router.calibration import auroc, expected_calibration_error, load_calibrators
from src.router.classifier import ClassifierRouter
from src.router.embedding import EmbeddingRouter, load_routes
from src.router.hybrid import HybridRouter
from src.router.rules import RuleRouter
from src.tools.registry import build_registry

RESULTS = EVAL_DIR / "results"
EXAMPLE_QUERIES = [  
    "What is 245 * 18?", "What is the weather in Singapore today?", "Who won the latest Formula 1 race?",
    "Explain gradient descent.", "What's 15% of Tesla's stock price?", "how much warmer is Singapore than London",
    "what about tomorrow", "how do you calculate compound interest", "do I need a coat tonight",
    "book me a flight to Bali", "what is 7 / 0",
]


def norm(text: str) -> str:
    return " ".join(text.lower().replace("?", "").split())


def check_no_leakage(test_sets: dict) -> None:
    """Fail loudly if any test query also appears in the training examples or the validation set."""
    seen = {norm(t) for spec in load_routes().values() for t in spec.get("examples", []) + spec.get("hard_examples", [])}
    seen |= {norm(q.query) for q in load_queries(EVAL_DIR / "validation.jsonl")}
    for name, queries in test_sets.items():
        leaked = [q.query for q in queries if norm(q.query) in seen]
        if leaked:
            raise SystemExit(f"LEAKAGE: {name} set contains training/validation queries: {leaked}")


def pct(x) -> str:
    return "–" if x is None else f"{x:.1%}"


def ms(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-llm", action="store_true", help="use Gemini for end-to-end latency")
    args = parser.parse_args()

    test = load_queries(EVAL_DIR / "test.jsonl")
    robust = load_queries(EVAL_DIR / "robustness.jsonl")
    check_no_leakage({"test": test, "robustness": robust})
    config = load_router_config()
    costs = config["costs"]

    print("Loading routers...")
    embedder = EmbeddingRouter()
    routers = {
        "rules": RuleRouter(),
        "embedding": embedder,
        "classifier": ClassifierRouter(embedder),
        "hybrid": HybridRouter(embedder),
        "hybrid+rerank": HybridRouter(embedder, use_reranker=True),
    }

    # 1. routing quality
    print("Routing test + robustness sets...")
    routing = {}
    for set_name, queries in [("test", test), ("robustness", robust)]:
        routing[set_name] = {}
        for name, router in routers.items():
            decisions = [router.route(q.query) for q in queries]
            routing[set_name][name] = (routing_metrics(queries, decisions, costs), decisions)

    flags = {name: [float(c) for c in m["correct_flags"]] for name, (m, _) in routing["test"].items()}
    ci = {name: bootstrap_ci(f) for name, f in flags.items()}
    comparisons = {f"hybrid vs {other}": paired_bootstrap_diff(flags["hybrid"], flags[other])
                   for other in ["rules", "embedding", "classifier", "hybrid+rerank"]}

    # 2. calibration on unseen data
    calibrators = load_calibrators()
    calibration = {}
    for name in ["embedding", "classifier"]:
        scores = [routers[name].score(q.query) for q in test]
        confidences = [calibrators[name].predict(s) for s in scores]
        correct = [q.is_correct(max(s, key=s.get)) for q, s in zip(test, scores)]
        calibration[name] = {"ece": expected_calibration_error(confidences, correct),
                             "auroc": auroc(confidences, correct), "confidences": confidences, "correct": correct}

    # 3. end to end
    print(f"End-to-end run ({'Gemini' if args.real_llm else 'mock LLM'}, simulated tool latency)...")
    llm = GeminiClient() if args.real_llm else MockLLM()
    executor = Executor(routers["hybrid"], llm, tools=build_registry(simulate_latency=True))
    runs = [executor.run(q.query) for q in test]
    e2e_correct = [q.is_correct(r.final_route) for q, r in zip(test, runs)]
    by_final = {}
    for r in runs:
        by_final.setdefault(r.final_route, []).append(r)
    e2e_by_route = {
        route: {"count": len(rs), **{f"total_{k}": v for k, v in latency_summary([r.timings_ms["total"] for r in rs]).items()},
                **{f"mean_{part}_ms": sum(r.timings_ms[part] for r in rs) / len(rs) for part in ("routing", "tools", "llm")}}
        for route, rs in sorted(by_final.items())
    }
    events = Counter(e.split(":")[0].split(";")[0] for r in runs for e in r.events)
    examples = [executor.run(q).to_dict() for q in EXAMPLE_QUERIES]

    # 4. write results
    RESULTS.mkdir(parents=True, exist_ok=True)
    names = list(routers)
    t = {name: routing["test"][name][0] for name in names}
    rb = {name: routing["robustness"][name][0] for name in names}
    llm_label = f"Gemini ({llm.model})" if args.real_llm else "mock (0 ms; run with --real-llm for real LLM latency)"

    report = [f"# Evaluation results\n\nTest set: {len(test)} held-out queries. Robustness set: {len(robust)} "
              f"typo/slang variants. Accept threshold: {config['accept_threshold']}. LLM: {llm_label}. "
              f"Tool latency: simulated (weather 150 ms, web search 400 ms).\n",
              "## Routing quality (test set)\n",
              table(["router", "accuracy [95% CI]", "strict acc.", "incorrect tool", "missed tool",
                     "unnecessary tool", "clarify", "fallback", "mean mistake cost", "p50 ms", "p95 ms"],
                    [[n, f"{pct(t[n]['accuracy'])} [{pct(ci[n][0])}, {pct(ci[n][1])}]", pct(t[n]["strict_accuracy"]),
                      pct(t[n]["incorrect_tool_selection_rate"]), pct(t[n]["missed_tool_rate"]),
                      pct(t[n]["unnecessary_tool_rate"]), pct(t[n]["clarify_rate"]), pct(t[n]["fallback_rate"]),
                      f"{t[n]['mean_mistake_cost']:.3f}", ms(t[n]["latency_ms"]["p50"]), ms(t[n]["latency_ms"]["p95"])]
                     for n in names]),
              "\n## Is the hybrid really better? (paired bootstrap, test set)\n",
              table(["comparison", "accuracy difference", "95% CI", "beyond luck?"],
                    [[k, f"{v['mean_diff']:+.1%}", f"[{v['ci_low']:+.1%}, {v['ci_high']:+.1%}]",
                      "yes" if v["significant"] else "no"] for k, v in comparisons.items()]),
              "\n## Robustness (typos and slang)\n",
              table(["router", "test accuracy", "robustness accuracy", "change"],
                    [[n, pct(t[n]["accuracy"]), pct(rb[n]["accuracy"]),
                      f"{rb[n]['accuracy'] - t[n]['accuracy']:+.1%}"] for n in names]),
              "\n## Accuracy by category (test set)\n",
              table(["category"] + names, [[cat] + [pct(t[n]["per_category_accuracy"].get(cat)) for n in names]
                                          for cat in t["hybrid"]["per_category_accuracy"]]),
              f"\nMulti-step plans fully correct (hybrid): {pct(t['hybrid']['multi_step_plan_accuracy'])}\n",
              "\n## Hybrid: which stage decided (test set)\n",
              table(["stage", "queries", "accuracy", "p50 routing ms"],
                    [[s, v["count"], pct(v["accuracy"]), ms(v["latency_p50_ms"])]
                     for s, v in sorted(t["hybrid"]["by_stage"].items(), key=lambda kv: -kv[1]["count"])]),
              "\n## Hybrid: routing latency by chosen route (test set)\n",
              table(["route", "p50 ms", "p95 ms"], [[r, ms(v["p50"]), ms(v["p95"])]
                                                    for r, v in sorted(t["hybrid"]["latency_by_route_ms"].items())]),
              "\n## End to end (hybrid + executor, test set)\n",
              f"Accuracy of the route that actually answered (after execution fallbacks): "
              f"{pct(sum(e2e_correct) / len(e2e_correct))}. Total latency p50 "
              f"{ms(latency_summary([r.timings_ms['total'] for r in runs])['p50'])} ms, p95 "
              f"{ms(latency_summary([r.timings_ms['total'] for r in runs])['p95'])} ms."
              + (f" Gemini rate-limit retries: {llm.retries}." if args.real_llm else "") + "\n",
              table(["answered by", "queries", "total p50 ms", "total p95 ms", "mean routing ms", "mean tools ms", "mean LLM ms"],
                    [[r, v["count"], ms(v["total_p50"]), ms(v["total_p95"]), ms(v["mean_routing_ms"]),
                      ms(v["mean_tools_ms"]), ms(v["mean_llm_ms"])] for r, v in e2e_by_route.items()]),
              "\nExecutor events: " + (", ".join(f"{k} ({v})" for k, v in events.most_common()) or "none") + "\n",
              "\n## Calibration on the test set (calibrators were fitted on validation only)\n",
              table(["router", "ECE (lower = more honest)", "AUROC (higher = better at spotting its own mistakes)"],
                    [[n, f"{v['ece']:.3f}", f"{v['auroc']:.3f}"] for n, v in calibration.items()]),
              "\n## Hybrid confusion matrix (rows = labelled route, columns = predicted)\n"]
    cm = confusion_matrix(test, routing["test"]["hybrid"][1])
    cols = sorted({c for row in cm.values() for c in row})
    report.append(table(["labelled \\ predicted"] + cols, [[r] + [cm[r].get(c, 0) for c in cols] for r in sorted(cm)]))
    (RESULTS / "report.md").write_text("\n".join(report) + "\n")

    errors = ["# Misrouted queries (hybrid)\n"]
    for set_name, queries in [("test", test), ("robustness", robust)]:
        rows = [[q.query, q.category, "/".join(q.acceptable or [q.route]), d.route, d.stage, d.reason.replace("|", "/")]
                for q, d in zip(queries, routing[set_name]["hybrid"][1]) if not q.is_correct(d.route)]
        errors += [f"\n## {set_name} ({len(rows)} errors)\n", table(["query", "category", "expected", "got", "stage", "reason"], rows)]
    (RESULTS / "errors.md").write_text("\n".join(errors) + "\n")

    (RESULTS / "examples.md").write_text(
        f"# Example outputs (LLM: {llm_label})\n\n" + "\n\n".join(
            f"## {ex['query']}\n\n```json\n{json.dumps(ex, indent=2, ensure_ascii=False)}\n```" for ex in examples) + "\n")

    def strip(m):
        return {k: v for k, v in m.items() if k != "correct_flags"}
    (RESULTS / "metrics.json").write_text(json.dumps({
        "test": {n: strip(t[n]) for n in names}, "robustness": {n: strip(rb[n]) for n in names},
        "accuracy_ci95": ci, "comparisons": comparisons,
        "calibration": {n: {"ece": v["ece"], "auroc": v["auroc"]} for n, v in calibration.items()},
        "end_to_end": {"llm": llm_label, "final_route_accuracy": sum(e2e_correct) / len(e2e_correct),
                       "by_final_route": e2e_by_route, "events": dict(events)},
        "config": config,
    }, indent=2))

    grouped_hbar(RESULTS / "accuracy.png", names,
                 {"test": [t[n]["accuracy"] for n in names], "robustness (typos, slang)": [rb[n]["accuracy"] for n in names]},
                 "Routing accuracy by approach", "accuracy (whiskers: 95% bootstrap CI on test)",
                 xlim=(0, 1.12), errors={"test": [ci[n] for n in names]})
    grouped_hbar(RESULTS / "latency.png", names,
                 {"p50": [t[n]["latency_ms"]["p50"] for n in names], "p95": [t[n]["latency_ms"]["p95"] for n in names]},
                 "Routing latency by approach (test set)", "milliseconds per query (routing only)", value_fmt="{:.1f} ms")
    stacked_hbar(RESULTS / "e2e_latency.png", list(e2e_by_route),
                 {part: [v[f"mean_{part}_ms"] for v in e2e_by_route.values()] for part in ("routing", "tools", "llm")},
                 f"End-to-end latency by answering route (LLM: {'Gemini' if args.real_llm else 'mock'})",
                 "mean milliseconds per query (tool latency simulated)")
    reliability(RESULTS / "reliability.png", calibration["classifier"]["confidences"], calibration["classifier"]["correct"],
                "Classifier calibration on unseen test queries")

    print("\n" + "\n".join(report[:3]))
    print(f"\nFull report: eval/results/report.md   errors: eval/results/errors.md   examples: eval/results/examples.md")


if __name__ == "__main__":
    main()
