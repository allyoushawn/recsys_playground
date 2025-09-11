import json
import numpy as np
import torch

from tiger_semantic_id_amazon_beauty.src.semantic_id import assign_semantic_ids


def test_assign_semantic_ids_collision(tmp_path):
    # Two items share same (c1,c2,c3) -> expect c4=0,1
    codes = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=torch.long)
    sid, sid_to_items, prefix = assign_semantic_ids(codes, str(tmp_path), codebook_size=256)
    assert sid.shape == (3, 4)
    # Check c4 assignments for the first two
    assert tuple(sid[0].tolist()[:3]) == (1, 2, 3)
    assert tuple(sid[1].tolist()[:3]) == (1, 2, 3)
    assert {int(sid[0, 3]), int(sid[1, 3])} == {0, 1}
    # Files persisted
    assert (tmp_path / "semantic_ids.npy").exists()
    assert (tmp_path / "sid_to_items.json").exists()
    # Mapping keys present
    loaded = json.loads((tmp_path / "sid_to_items.json").read_text())
    assert "1-2-3-0" in loaded or "1-2-3-1" in loaded
