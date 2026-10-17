"""load config/router.yaml (thresholds, costs, defaults)."""
from pathlib import Path

import yaml

ROUTER_CONFIG = Path(__file__).resolve().parents[1] / "config" / "router.yaml"


def load_router_config(path: Path = ROUTER_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
