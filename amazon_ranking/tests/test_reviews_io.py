import gzip
import json

import pytest

from amazon_ranking.src.reviews_io import load_reviews_streaming


def _write_gz(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_streaming_legacy_projects_three_columns(tmp_path):
    p = tmp_path / "reviews_Beauty_5.json.gz"
    # include a heavy text field that must NOT be retained / must not appear
    rows = [
        {"reviewerID": f"u{u}", "asin": f"i{u % 3}", "unixReviewTime": 100 + u, "reviewText": "x" * 5000}
        for u in range(10)
    ]
    _write_gz(p, rows)
    df = load_reviews_streaming(str(p), "legacy")
    assert list(df.columns) == ["user_id", "item_id", "ts"]
    assert len(df) == 10
    assert df["ts"].dtype.kind == "i"
    assert df["user_id"].iloc[0] == "u0" and df["item_id"].iloc[0] == "i0"


def test_streaming_2023_and_max_users(tmp_path):
    p = tmp_path / "Video_Games.jsonl.gz"
    rows = [
        {"user_id": f"u{u}", "parent_asin": f"i{u}", "timestamp": 1000 + u, "title": "t" * 2000}
        for u in range(20)
        for _ in range(2)  # 2 interactions per user
    ]
    _write_gz(p, rows)
    df = load_reviews_streaming(str(p), "2023", max_users=5)
    assert list(df.columns) == ["user_id", "item_id", "ts"]
    assert df["user_id"].nunique() == 5  # subsample honored
    assert len(df) == 10  # 5 users x 2 interactions


def test_streaming_skips_malformed_and_missing_fields(tmp_path):
    p = tmp_path / "r.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps({"reviewerID": "u1", "asin": "i1", "unixReviewTime": 1}) + "\n")
        f.write("{not valid json\n")  # malformed -> skipped
        f.write(json.dumps({"reviewerID": "u2", "asin": "i2"}) + "\n")  # missing ts -> skipped
    df = load_reviews_streaming(str(p), "legacy")
    assert len(df) == 1 and df["user_id"].iloc[0] == "u1"


def test_streaming_unknown_format_raises(tmp_path):
    p = tmp_path / "r.jsonl.gz"
    _write_gz(p, [{"reviewerID": "u1", "asin": "i1", "unixReviewTime": 1}])
    with pytest.raises(ValueError):
        load_reviews_streaming(str(p), "nope")
