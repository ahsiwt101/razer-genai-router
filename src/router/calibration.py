"""Calibration: to turn raw score into probabilities using platt scaling 

record score of winning route + margin 
then logistic regression to map that to P(correct) 

"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CALIBRATION_FILE = CONFIG_DIR / "calibration.json"

def features(scores: dict[str, float]) -> list[float]:
    top, second = sorted(scores.values(), reverse=True)[:2]
    return [top, top - second]


class ConfidenceCalibrator:
    """maps (top, margin) -> P(correct) 
    features are standardised first --> mean 0, var 1 
    """

    def __init__(self, coef=None, intercept: float = 0.0, mean=None, scale=None):
        self.coef = np.array(coef) if coef is not None else None
        self.intercept = intercept
        self.mean = np.array(mean) if mean is not None else None
        self.scale = np.array(scale) if scale is not None else None

    def fit(self, score_dicts: list[dict[str, float]], correct: list[bool]) -> "ConfidenceCalibrator":
        X = np.array([features(s) for s in score_dicts])
        self.mean, self.scale = X.mean(axis=0), X.std(axis=0) + 1e-9  # +tiny avoids dividing by zero
        model = LogisticRegression(C=1.0).fit((X - self.mean) / self.scale, np.array(correct, dtype=int))
        self.coef, self.intercept = model.coef_[0], float(model.intercept_[0])
        return self

    def predict(self, scores: dict[str, float]) -> float:
        """P(the top-scoring route is the correct one)."""
        x = (np.array(features(scores)) - self.mean) / self.scale
        z = float(np.dot(self.coef, x) + self.intercept)
        return 1.0 / (1.0 + np.exp(-z))  # squashes any number into 0..1

    def to_dict(self) -> dict:
        return {"coef": self.coef.tolist(), "intercept": self.intercept,
                "mean": self.mean.tolist(), "scale": self.scale.tolist()}


def expected_calibration_error(confidences: list[float], correct: list[bool], bins: int = 5) -> float: 
    # group decisions by confidence intervals. comparing average confidence with fraction actually correct. 
    conf, corr = np.array(confidences), np.array(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(conf[in_bin].mean() - corr[in_bin].mean())
    return float(ece)


def auroc(confidences: list[float], correct: list[bool]) -> float:
    # ece not always reliable, hence auroc to measure how well confidence separates correct vs wrong decisions.
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.array(correct, dtype=int), confidences))


def save_calibrators(calibrators: dict[str, ConfidenceCalibrator], path: Path = CALIBRATION_FILE) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({name: c.to_dict() for name, c in calibrators.items()}, indent=2))


def load_calibrators(path: Path = CALIBRATION_FILE) -> dict[str, ConfidenceCalibrator]:
    data = json.loads(path.read_text())
    return {name: ConfidenceCalibrator(d["coef"], d["intercept"], d["mean"], d["scale"])
            for name, d in data.items()}
