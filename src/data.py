"""Loading labelled query datasets (validation / test / robustness).

Each line of a .jsonl file is one JSON object:
  query       the user's text
  route       the single best route
  category    clear | overlap | ambiguous | no_tool | multi_step | out_of_scope
  acceptable  (optional) every route that would also be a reasonable answer; for genuinely
              ambiguous queries there isn't one right answer, and grading as if there were is unfair
  plan        (optional) the full tool sequence for multi-step queries
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


@dataclass
class LabelledQuery:
    query: str
    route: str
    category: str
    acceptable: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)

    def is_correct(self, predicted_route: str) -> bool:
        return predicted_route in (self.acceptable or [self.route])


def load_queries(path: Path) -> list[LabelledQuery]:
    with open(path) as f:
        return [LabelledQuery(**json.loads(line)) for line in f if line.strip()]
