from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from .utils import ensure_dirs


def assign_semantic_ids(
    codes: torch.Tensor,  # [num_items, 3]
    artifacts_dir: str,
    codebook_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, List[int]], Dict[int, List[int]]]:
    """Assign 4-token Semantic IDs (c1,c2,c3,c4) with collision resolution.

    Returns:
      - semantic_ids: [num_items, 4] int16
      - sid_to_items: map from 'c1-c2-c3-c4' to list of item_idx
      - prefix_to_items: map from 'c1-c2-c3' (no c4) to item_idx list
    """
    ensure_dirs(artifacts_dir)
    codes_np = codes.detach().cpu().numpy().astype(np.int32)
    num_items = codes_np.shape[0]
    sid = np.zeros((num_items, 4), dtype=np.int16)
    slot_to_items: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i, (c1, c2, c3) in enumerate(codes_np):
        slot_to_items[(int(c1), int(c2), int(c3))].append(i)
    sid_to_items: Dict[str, List[int]] = {}
    prefix_to_items: Dict[str, List[int]] = {}
    max_c4 = 0
    for (c1, c2, c3), items in slot_to_items.items():
        items = sorted(items)
        for c4, item_idx in enumerate(items):
            sid[item_idx] = [c1, c2, c3, c4]
            key = f"{c1}-{c2}-{c3}-{c4}"
            sid_to_items.setdefault(key, []).append(item_idx)
            prefix_key = f"{c1}-{c2}-{c3}"
            prefix_to_items.setdefault(prefix_key, []).append(item_idx)
            max_c4 = max(max_c4, c4)
    print(f"Assigned semantic IDs. Collisions: {sum(len(v)>1 for v in slot_to_items.values())}; max c4={max_c4}")
    # Persist
    np.save(os.path.join(artifacts_dir, "semantic_ids.npy"), sid)
    with open(os.path.join(artifacts_dir, "sid_to_items.json"), "w") as f:
        json.dump(sid_to_items, f)
    with open(os.path.join(artifacts_dir, "prefix_to_items.json"), "w") as f:
        json.dump(prefix_to_items, f)
    return sid, sid_to_items, {tuple(map(int, k.split("-"))): v for k, v in prefix_to_items.items()}  # type: ignore


def build_semantic_vocab(levels: int = 4, codebook_size: int = 256) -> int:
    """Flat token space size for semantic tokens: levels * codebook_size."""
    return levels * codebook_size

