"""Stage 4: Cross-Encoder Reranker
used only when the embedding stage is unsure.

bi-encoder: query, examples embedded seperately. faster 
cross-encoder: query + example as one input. output relevance score. 

model used: ms-marco-MiniLM-L6-v2. 
compared three cross-encoders: ms-marco-MiniLM-L6-v2, Quora-duplicate, STS-B 
ms-macro was the fastest + most accurate on our validation set.
"""

import numpy as np
from sentence_transformers import CrossEncoder

from .decision import BaseRouter, RoutingDecision
from .embedding import EmbeddingRouter

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"

class Reranker:
    def __init__(self, embedder: EmbeddingRouter, model_name: str = DEFAULT_RERANKER,
                 per_route: int = 3, top_routes: int = 2):
        # recheck embedding router top 2 + each routes top 3 examples. 
        # route's score = its best-scoring example (max) 
        self.embedder = embedder
        self.per_route = per_route
        self.top_routes = top_routes
        self.model = CrossEncoder(model_name, device="cpu")
        self.model.predict([("warm up", "warm up")] * 3, show_progress_bar=False)  

    def rerank(self, query: str, sims: np.ndarray | None = None) -> tuple[dict[str, float], dict[str, str]]:
        """re-score the top routes. Returns ({route: score}, {route: best-matching example}).

        sims: the embedding stage's similarities, if already computed (avoids embedding twice).
        """
        if sims is None:
            sims = self.embedder.similarities(query)
        coarse = self.embedder._aggregate(sims)
        contenders = sorted(coarse, key=coarse.get, reverse=True)[: self.top_routes]

        pairs, owners = [], []
        for route in contenders:
            idx = np.flatnonzero(self.embedder.labels == route)
            for i in idx[np.argsort(sims[idx])[::-1][: self.per_route]]:  
                pairs.append((query, self.embedder.texts[i]))
                owners.append(route)

        pair_scores = self.model.predict(pairs, show_progress_bar=False)
        scores, best_example = {}, {}
        for (_, example), route, s in zip(pairs, owners, pair_scores):
            if route not in scores or s > scores[route]:
                scores[route], best_example[route] = float(s), example
        return scores, best_example


class RerankRouter(BaseRouter):
    """embedding + reranking on every query."""

    name = "rerank"

    def __init__(self, embedder: EmbeddingRouter | None = None, reranker: Reranker | None = None):
        self.embedder = embedder or EmbeddingRouter()
        self.reranker = reranker or Reranker(self.embedder)

    def score(self, query: str) -> dict[str, float]:
        return self.reranker.rerank(query)[0]

    def _route(self, query: str) -> RoutingDecision:
        scores, examples = self.reranker.rerank(query)
        best, second = sorted(scores, key=scores.get, reverse=True)[:2]
        reason = (f"Reranker prefers {best} (closest: '{examples[best]}', {scores[best]:.2f}) "
                  f"over {second} ({scores[second]:.2f})")
        # calibration converts raw reranker scores 
        return RoutingDecision(best, scores[best], reason, stage="reranker", scores=scores)
