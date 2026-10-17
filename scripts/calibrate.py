"""Fit confidence calibrators on the validation set and save them to config/calibration.json.

Run:  python scripts/calibrate.py

For each router (embedding, classifier, reranker) this:
  1. routes every validation query and records its scores + whether it was correct
  2. measures how honest the RAW confidence is (ECE, lower = better)
  3. measures how honest the CALIBRATED confidence is, using 5-fold cross-validation
     (fit on 4/5 of the queries, test on the other 1/5, rotate) so we never grade the
     calibrator on queries it was fitted to
  4. fits the final calibrator on all validation queries and saves it
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data import EVAL_DIR, load_queries
from src.router.calibration import (CALIBRATION_FILE, ConfidenceCalibrator, auroc, expected_calibration_error,
                                    save_calibrators)
from src.router.classifier import ClassifierRouter
from src.router.embedding import EmbeddingRouter
from src.router.reranker import RerankRouter


def raw_confidence(scores: dict[str, float]) -> float:
    """What the router would naively report as confidence, squashed into 0..1 if it isn't already."""
    top = max(scores.values())
    return top if 0.0 <= top <= 1.0 else float(1 / (1 + np.exp(-top)))


def cross_validated(score_dicts, correct, folds=5) -> list[float]:
    preds = np.zeros(len(correct))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    for train, test in splitter.split(np.zeros(len(correct)), correct):
        cal = ConfidenceCalibrator().fit([score_dicts[i] for i in train], [correct[i] for i in train])
        for i in test:
            preds[i] = cal.predict(score_dicts[i])
    return preds.tolist()


val = load_queries(EVAL_DIR / "validation.jsonl")
embedder = EmbeddingRouter()
routers = {
    "embedding": embedder,
    "classifier": ClassifierRouter(embedder),
    "reranker": RerankRouter(embedder),
}

print(f"Validation set: {len(val)} queries\n")
print(f"{'':22s} {'---- ECE (lower=better) ----':>30s}   {'--- AUROC (higher=better) ---':>30s}")
print(f"{'router':12s} {'accuracy':>9s} {'raw':>14s} {'calibrated':>14s}   {'raw':>14s} {'calibrated':>14s}")
calibrators = {}
for name, router in routers.items():
    score_dicts = [router.score(q.query) for q in val]
    correct = [q.is_correct(max(s, key=s.get)) for q, s in zip(val, score_dicts)]

    raw = [raw_confidence(s) for s in score_dicts]
    cal = cross_validated(score_dicts, correct)
    print(f"{name:12s} {sum(correct):>4d}/{len(val):<4d} "
          f"{expected_calibration_error(raw, correct):14.3f} {expected_calibration_error(cal, correct):14.3f}   "
          f"{auroc(raw, correct):14.3f} {auroc(cal, correct):14.3f}")

    calibrators[name] = ConfidenceCalibrator().fit(score_dicts, correct)

save_calibrators(calibrators)
print(f"\nSaved to {CALIBRATION_FILE.relative_to(Path.cwd()) if CALIBRATION_FILE.is_relative_to(Path.cwd()) else CALIBRATION_FILE}")

# Show what calibration does to a few concrete queries.
print("\nWhat the embedding router's confidence means before vs after calibration:")
emb_cal = calibrators["embedding"]
for query in ["What is the temperature of the sun?", "Will I need an umbrella tomorrow?",
              "What's 15% of Tesla's stock price?", "Explain gradient descent."]:
    s = embedder.score(query)
    best = max(s, key=s.get)
    print(f"  {query:40s} {best:11s} raw {raw_confidence(s):.2f}  ->  calibrated {emb_cal.predict(s):.2f}")
