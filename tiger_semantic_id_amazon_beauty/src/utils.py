from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility (numpy, random, torch if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dirs(*paths: str | os.PathLike[str]) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def try_orjson_dumps(obj: Any) -> str:
    try:
        import orjson  # type: ignore

        return orjson.dumps(obj).decode("utf-8")
    except Exception:
        import json

        return json.dumps(obj)


def try_orjson_loads(s: str) -> Any:
    try:
        import orjson  # type: ignore

        return orjson.loads(s)
    except Exception:
        import json

        return json.loads(s)


@dataclass
class Paths:
    data_dir: str = "/content/data"
    artifacts_dir: str = "/content/artifacts"

