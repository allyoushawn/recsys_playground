import os
import sys

import pandas as pd


# Ensure src/ is importable when running tests directly
CURRENT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from models.popularity import get_top_n  # noqa: E402


def test_get_top_n_by_count_with_tiebreaks():
    # Build a small, deterministic ratings DataFrame
    # movie 10: 3 ratings, mean 3.0
    # movie 20: 3 ratings, mean 4.0 (should rank before 10 due to higher mean)
    # movie 30: 2 ratings
    # movie 40: 1 rating
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "movie_id": [10, 10, 10, 20, 20, 20, 30, 30, 40],
            "rating": [3, 3, 3, 5, 4, 3, 2, 4, 5],
            "timestamp": [0] * 9,
        }
    )

    top = get_top_n(df, n=3)
    assert top == [20, 10, 30]


def test_get_top_n_clips_to_available_movies():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "movie_id": [1, 2, 3],
            "rating": [4, 5, 3],
            "timestamp": [0, 0, 0],
        }
    )

    top = get_top_n(df, n=10)
    assert len(top) == 3
    assert set(top) == {1, 2, 3}

