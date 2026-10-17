"""Comparison router: trained classifier on top of the same embeddings 

embedding does nearest neighbours 
classifier learns during training and gets weight for each one  

"""

from sklearn.linear_model import LogisticRegression

from .decision import BaseRouter, RoutingDecision
from .embedding import EmbeddingRouter

class ClassifierRouter(BaseRouter):
    name = "classifier"

    def __init__(self, embedder: EmbeddingRouter | None = None, C: float = 10.0):
        # reusing embedding router's model
        self.embedder = embedder or EmbeddingRouter()
        self.clf = LogisticRegression(C=C, max_iter=5000)
        self.clf.fit(self.embedder.example_vectors, self.embedder.labels)

    def score(self, query: str) -> dict[str, float]:
        #one probability per route; they sum to 1.
        probs = self.clf.predict_proba(self.embedder.embed([query]))[0]
        return {str(route): float(p) for route, p in zip(self.clf.classes_, probs)}

    def _route(self, query: str) -> RoutingDecision:
        scores = self.score(query)
        best, second = sorted(scores, key=scores.get, reverse=True)[:2]
        reason = f"Classifier probability {scores[best]:.2f} for {best} (next: {second} {scores[second]:.2f})"
        return RoutingDecision(best, scores[best], reason, stage="classifier", scores=scores)
