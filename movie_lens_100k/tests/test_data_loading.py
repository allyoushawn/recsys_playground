import os
import sys

import pandas as pd


# Ensure src/ is importable when running tests directly
CURRENT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from data.movielens import load_movielens_100k  # noqa: E402


def test_load_movielens_100k_returns_dataframe():
    df = load_movielens_100k()
    assert isinstance(df, pd.DataFrame)


def test_load_movielens_100k_has_expected_columns():
    df = load_movielens_100k()
    expected = ["user_id", "movie_id", "rating", "timestamp"]
    assert list(df.columns) == expected
    assert not df.empty

