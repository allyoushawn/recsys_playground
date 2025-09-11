import json

import pandas as pd

from tiger_semantic_id_amazon_beauty.src.data import load_meta_df


def test_load_meta_df_item_id_from_asin(tmp_path):
    # Write a tiny JSONL meta file with 'asin'
    p = tmp_path / "meta.json"
    rows = [
        {"asin": "A1", "title": "T", "brand": "B", "category": [["Beauty", "Hair"]], "price": 9.99},
        {"asin": "A2", "title": "U", "brand": "", "category": [["Beauty", "Makeup"]]},
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    df = load_meta_df(str(p))
    assert "item_id" in df.columns
    assert set(df["item_id"]) == {"A1", "A2"}
    assert "category_leaf" in df.columns

