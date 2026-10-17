"""Choose the hybrid router's accept_threshold on the VALIDATION set, and plot the trade-off.

Run:  python scripts/tune_thresholds.py

For each candidate threshold (0.50 to 0.99) it routes every validation query and records:
  accuracy          overall share routed correctly
  coverage          share decided confidently (rules / planner / classifier) instead of by the fallback
  accepted accuracy accuracy on just the confidently-decided queries
  unneeded tools    share of no-tool queries that were sent to a tool anyway

Choice rule (precision target): accept a confident decision only where confident decisions are
right at least 95% of the time; everything below goes to the cost-based fallback. We take the
LOWEST threshold from which that holds for EVERY higher threshold too, so a lucky isolated point
can't be picked.

Rejected rule: "maximise validation accuracy". It chose 0.52 (fallback almost switched off)
because of a ONE-query difference (44/48 vs 43/48). With 48 queries, one query is ~2 percentage
points of noise, too little to justify removing the safety net.

The test set is never touched here.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
import numpy as np

from src.config import ROUTER_CONFIG
from src.data import EVAL_DIR, load_queries
from src.router.embedding import EmbeddingRouter
from src.router.hybrid import HybridRouter

TOOLS = {"calculator", "weather", "web_search"}
CONFIDENT_STAGES = {"rules", "planner", "classifier", "out_of_scope"}
RESULTS = EVAL_DIR / "results"


def sweep(val, embedder, thresholds):
    rows = []
    no_tool = [q for q in val if not set(q.acceptable or [q.route]) & TOOLS]
    for t in thresholds:
        router = HybridRouter(embedder, accept_threshold=t)
        decisions = {q.query: router.route(q.query) for q in val}
        correct = [q.is_correct(decisions[q.query].route) for q in val]
        confident = [decisions[q.query].stage in CONFIDENT_STAGES for q in val]
        accepted = [c for c, conf in zip(correct, confident) if conf]
        rows.append({
            "threshold": round(float(t), 2),
            "accuracy": float(np.mean(correct)),
            "coverage": float(np.mean(confident)),
            "accepted_accuracy": float(np.mean(accepted)) if accepted else float("nan"),
            "unneeded_tools": float(np.mean([decisions[q.query].route in TOOLS for q in no_tool])),
        })
    return rows


PRECISION_TARGET = 0.95


def choose(rows) -> float:
    for i, r in enumerate(rows):
        if all(later["accepted_accuracy"] >= PRECISION_TARGET for later in rows[i:]):
            return r["threshold"]
    return rows[-1]["threshold"]  # target never reached: be as cautious as possible


def plot(rows, chosen, path):
    t = [r["threshold"] for r in rows]
    # categorical slots 1-4 of the validated palette, in fixed order
    series = [("accuracy", "Accuracy", "#2a78d6"), ("coverage", "Coverage (decided confidently)", "#eb6834"),
              ("accepted_accuracy", "Accuracy when confident", "#1baf7a"),
              ("unneeded_tools", "Unneeded tool calls (no-tool queries)", "#eda100")]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for key, label, color in series:
        y = [r[key] for r in rows]
        ax.plot(t, y, color=color, linewidth=2, label=label)
        ax.annotate(label, (t[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#52514e")  # direct label in text ink
    ax.axvline(chosen, color="#52514e", linewidth=1, linestyle="--")
    ax.annotate(f"chosen: {chosen:.2f}", (chosen, 1.03), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color="#0b0b0b")
    ax.set_xlabel("accept_threshold (calibrated confidence)", color="#52514e")
    ax.set_ylabel("share of validation queries", color="#52514e")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(t[0], t[-1])
    ax.set_title("Hybrid router: threshold trade-off on the validation set", loc="left", fontsize=11, color="#0b0b0b")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.8, 1))  # leave room on the right for direct labels
    fig.savefig(path)
    plt.close(fig)


def write_threshold(value: float) -> None:
    """Replace only the number on the accept_threshold line, keeping every comment in the file."""
    text = ROUTER_CONFIG.read_text()
    new_text, n = re.subn(r"^(accept_threshold:\s*)[0-9.]+", rf"\g<1>{value:.2f}", text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("Couldn't find the accept_threshold line in config/router.yaml")
    ROUTER_CONFIG.write_text(new_text)


if __name__ == "__main__":
    val = load_queries(EVAL_DIR / "validation.jsonl")
    embedder = EmbeddingRouter()
    rows = sweep(val, embedder, np.arange(0.50, 0.995, 0.01))
    chosen = choose(rows)

    print(f"{'threshold':>9} {'accuracy':>9} {'coverage':>9} {'acc|conf':>9} {'unneeded':>9}")
    for r in rows[::5] + [r for r in rows if r["threshold"] == chosen]:
        mark = "  <- chosen" if r["threshold"] == chosen else ""
        print(f"{r['threshold']:>9.2f} {r['accuracy']:>9.3f} {r['coverage']:>9.3f} "
              f"{r['accepted_accuracy']:>9.3f} {r['unneeded_tools']:>9.3f}{mark}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    plot(rows, chosen, RESULTS / "threshold_sweep.png")
    write_threshold(chosen)
    print(f"\nChosen accept_threshold = {chosen:.2f} (written to config/router.yaml)")
    print(f"Plot: eval/results/threshold_sweep.png")
