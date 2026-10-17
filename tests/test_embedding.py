"""Unit tests for the embedding router. Run:  pytest -q   (loads the model once, ~5s)"""
import pytest

from src.router.embedding import EmbeddingRouter


@pytest.fixture(scope="module")  # load the model once for all tests in this file, not once per test
def router():
    return EmbeddingRouter()


def test_identical_text_has_similarity_one(router):
    a, b = router.embed(["will it rain tomorrow", "will it rain tomorrow"])
    assert float(a @ b) == pytest.approx(1.0, abs=1e-5)


def test_every_route_gets_a_score(router):
    assert set(router.score("hello")) == {"calculator", "weather", "web_search", "llm"}


@pytest.mark.parametrize("query, route", [
    ("What's two hundred times three?", "calculator"),   # rules miss this (numbers as words)
    ("Is it gonna pour in Tokyo later?", "weather"),      # rules miss this ("pour" not in word list)
    ("How's Nvidia doing on the market?", "web_search"),  # rules miss this (no keywords)
    ("Explain gradient descent.", "llm"),
])
def test_routes_paraphrases(router, query, route):
    assert router.route(query).route == route
