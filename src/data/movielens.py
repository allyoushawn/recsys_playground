import os
import zipfile
from urllib.request import urlretrieve
from typing import Optional

import pandas as pd


ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def _default_data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "movielens")


def load_movielens_100k(dest_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Download (if needed) and load the MovieLens-100K dataset.

    - Downloads the archive from GroupLens if not already present
    - Extracts the archive to a cache directory
    - Loads `u.data` into a pandas DataFrame with columns: user_id, movie_id, rating, timestamp

    Parameters
    ----------
    dest_dir : Optional[str]
        Directory to store the dataset. Defaults to `<repo_root>/data/movielens`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: user_id, movie_id, rating, timestamp
    """
    base_dir = dest_dir or _default_data_dir()
    os.makedirs(base_dir, exist_ok=True)

    zip_path = os.path.join(base_dir, "ml-100k.zip")
    extract_dir = os.path.join(base_dir, "ml-100k")

    # Download if the archive doesn't exist
    if not os.path.exists(zip_path) and not os.path.exists(extract_dir):
        urlretrieve(ML_100K_URL, zip_path)

    # Extract if not already extracted
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_dir)

    data_file = os.path.join(extract_dir, "u.data")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Expected file not found: {data_file}")

    # Load the data
    df = pd.read_csv(
        data_file,
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )

    return df

