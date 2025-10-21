import pandas as pd

from tiger_semantic_id_amazon_beauty.src.data import filter_and_split, BeautyConfig


def test_filter_and_split_leave_one_out():
    # Two users; u1 has 5 interactions, u2 has 4; with min=5 expect only u1
    rows = []
    ts = 1
    for i in range(5):
        rows.append({"user_id": "u1", "item_id": f"i{i}", "ts": ts}); ts += 1
    for i in range(4):
        rows.append({"user_id": "u2", "item_id": f"j{i}", "ts": ts}); ts += 1
    df = pd.DataFrame(rows)
    tr, va, te = filter_and_split(df, BeautyConfig(min_user_interactions=5, max_hist_len=20))
    # Only u1 remains; train should have len-2, val len=1, test len=1
    assert tr["user_id"].nunique() == 1
    assert len(tr) == 3 and len(va) == 1 and len(te) == 1
    # Chronology: last -> test, second last -> val
    assert te.iloc[0]["item_id"] == "i4"
    assert va.iloc[0]["item_id"] == "i3"
