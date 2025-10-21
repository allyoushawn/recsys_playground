import pandas as pd

from tiger_semantic_id_amazon_beauty.src.data import build_id_maps, apply_id_maps


def test_build_and_apply_id_maps():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u1", "u3"],
            "item_id": ["i1", "i2", "i3", "i2"],
            "ts": [1, 2, 3, 4],
        }
    )
    u2i, it2i = build_id_maps([df])
    assert set(u2i.keys()) == {"u1", "u2", "u3"}
    assert set(it2i.keys()) == {"i1", "i2", "i3"}
    mapped = apply_id_maps(df, u2i, it2i)
    assert {"user_idx", "item_idx"}.issubset(mapped.columns)
    # Round trip
    inv_u = {v: k for k, v in u2i.items()}
    inv_i = {v: k for k, v in it2i.items()}
    assert inv_u[mapped.loc[0, "user_idx"]] == "u1"
    assert inv_i[mapped.loc[1, "item_idx"]] == "i2"

