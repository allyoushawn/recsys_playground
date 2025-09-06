from typing import List

import pandas as pd


def get_top_n(train_ratings: pd.DataFrame, n: int = 10) -> List[int]:
    """
    Return the top-N popular movies by interaction count.

    Popularity is defined as the number of ratings per `movie_id` in the
    provided training ratings. Ties are broken by higher mean rating and then
    by ascending `movie_id` to ensure deterministic output.

    Parameters
    ----------
    train_ratings : pd.DataFrame
        DataFrame containing at least the columns: `movie_id` and `rating`.
    n : int
        Number of top movie IDs to return. Defaults to 10.

    Returns
    -------
    List[int]
        A list of movie IDs ordered from most to least popular.
    """
    required_cols = {"movie_id", "rating"}
    missing = required_cols - set(train_ratings.columns)
    if missing:
        raise ValueError(f"train_ratings missing required columns: {sorted(missing)}")

    # Aggregate counts and mean ratings to provide deterministic tie-breaking
    agg = (
        train_ratings.groupby("movie_id")["rating"]
        .agg(count="count", mean="mean")
        .reset_index()
    )

    # Sort by count desc, mean desc, movie_id asc for stability
    agg_sorted = agg.sort_values(
        by=["count", "mean", "movie_id"], ascending=[False, False, True]
    )

    top_ids = agg_sorted["movie_id"].head(n).tolist()
    return top_ids

