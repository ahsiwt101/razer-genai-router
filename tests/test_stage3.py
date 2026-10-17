"""Tests for the reranker, classifier and calibration. Run:  pytest -q"""
import pytest

from src.router.calibration import ConfidenceCalibrator, auroc, expected_calibration_error
from src.router.classifier import ClassifierRouter
from src.router.embedding import EmbeddingRouter
from src.router.reranker import RerankRouter


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingRouter()


def test_classifier_probabilities_sum_to_one(embedder):
    scores = ClassifierRouter(embedder).score("is it raining in Oslo")
    assert sum(scores.values()) == pytest.approx(1.0)
    assert max(scores, key=scores.get) == "weather"


def test_reranker_scores_only_top_two_routes(embedder):
    scores = RerankRouter(embedder).score("Is it gonna pour in Tokyo later?")
    assert len(scores) == 2 and "weather" in scores


def test_ece_is_zero_when_confidence_matches_reality():
    # 10 decisions at 0.8 confidence, 8 of them correct -> perfectly honest
    assert expected_calibration_error([0.8] * 10, [True] * 8 + [False] * 2) == pytest.approx(0.0)


def test_constant_confidence_has_useless_auroc():
    # the trap we hit: the same confidence for everything can't separate right from wrong
    assert auroc([0.77] * 4, [True, True, False, True]) == pytest.approx(0.5)


def test_calibrator_gives_low_margin_lower_confidence():
    # decisions with big margins were right, small margins were wrong
    scores = [{"a": 0.6, "b": 0.2}] * 10 + [{"a": 0.45, "b": 0.44}] * 10
    correct = [True] * 10 + [False] * 10
    cal = ConfidenceCalibrator().fit(scores, correct)
    assert cal.predict({"a": 0.6, "b": 0.2}) > 0.8
    assert cal.predict({"a": 0.45, "b": 0.44}) < 0.2
