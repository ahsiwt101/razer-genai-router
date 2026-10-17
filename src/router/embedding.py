"""
Stage 2: Embeddings (semantic similarity) 

- a small neural network turns every example query in routes.yaml into a vector (an "embedding") 
- similar meaning -> vectors pointing in similar directions
- a new query is embedded the same way
- cosine similarity compares the query vector with every example vector (1.0 = same meaning)
- each route's score = the average of its top-k most similar examples. highest score wins
"""

from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from .decision import BaseRouter, RoutingDecision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTES = PROJECT_ROOT / "routes.yaml"
# general-purpose sentence embedding model 
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_routes(path: Path = DEFAULT_ROUTES) -> dict[str, dict]:
    with open(path) as f:
        return yaml.safe_load(f)["routes"]


class EmbeddingRouter(BaseRouter):
    name = "embedding"

    def __init__(self, routes_path: Path = DEFAULT_ROUTES, model_name: str = DEFAULT_MODEL,
                 top_k: int = 3, model: SentenceTransformer | None = None):
        # top_k: how many nearest examples per route to average
        self.top_k = top_k
        self.model = model or SentenceTransformer(model_name)

        routes = load_routes(routes_path)
        self.route_names = list(routes)
        self.texts: list[str] = []
        labels: list[str] = []
        for route, spec in routes.items():
            for text in spec.get("examples", []) + spec.get("hard_examples", []):
                self.texts.append(text)
                labels.append(route)
        self.labels = np.array(labels)

        # embed every example once. 
        # shape: (number_of_examples, 384).
        # hence routing query is just one embedding now
        self.example_vectors = self.embed(self.texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        # scales every vector to length 1
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def similarities(self, query: str) -> np.ndarray:
        #cosine similarity between the query and every example
        query_vector = self.embed([query])[0]
        return self.example_vectors @ query_vector 

    def score(self, query: str) -> dict[str, float]:
        #one score per route.
        return self._aggregate(self.similarities(query))

    def nearest(self, query: str, n: int = 5) -> list[tuple[str, str, float]]:
        # the n most similar examples as (example text, its route, similarity). 
        sims = self.similarities(query)
        top = np.argsort(sims)[::-1][:n]
        return [(self.texts[i], str(self.labels[i]), float(sims[i])) for i in top]

    def _aggregate(self, sims: np.ndarray) -> dict[str, float]:
        scores = {}
        for route in self.route_names:
            route_sims = np.sort(sims[self.labels == route])[::-1] 
            scores[route] = float(route_sims[: self.top_k].mean())
        return scores

    def _route(self, query: str) -> RoutingDecision:
        sims = self.similarities(query)
        scores = self._aggregate(sims)
        best, second = sorted(scores, key=scores.get, reverse=True)[:2]
        margin = scores[best] - scores[second]

        # the single closest example from the winning route
        best_idx = max(np.flatnonzero(self.labels == best), key=lambda i: sims[i])
        reason = (f"Most similar to {best} examples, e.g. '{self.texts[best_idx]}' "
                  f"({sims[best_idx]:.2f}); margin over {second}: {margin:.2f}")

        # calibration to convert cosine score to real probability later 
        return RoutingDecision(best, max(0.0, scores[best]), reason, stage="embedding", scores=scores)
