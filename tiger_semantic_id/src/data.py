from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dirs


# Dataset URLs
# - Beauty: Legacy 2014 SNAP dataset (5-core)
# - Video_Games: Amazon Reviews 2023 dataset (full)
DATASET_URLS = {
    "Beauty": {
        "reviews": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
        "meta": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
        "format": "legacy",  # 2014 SNAP format
    },
    "Video_Games": {
        "reviews": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz",
        "meta": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz",
        "format": "2023",  # Amazon Reviews 2023 format
    },
}

# Legacy constants for backward compatibility
SNAP_REVIEWS = DATASET_URLS["Beauty"]["reviews"]
SNAP_META = DATASET_URLS["Beauty"]["meta"]


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    dataset_name: str = "Beauty"  # Options: "Beauty", "Video_Games"
    min_user_interactions: int = 5
    max_hist_len: int = 20

    def get_urls(self) -> Tuple[str, str]:
        """Get the review and metadata URLs for the configured dataset."""
        if self.dataset_name not in DATASET_URLS:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}. "
                f"Available options: {list(DATASET_URLS.keys())}"
            )
        urls = DATASET_URLS[self.dataset_name]
        return urls["reviews"], urls["meta"]

    def get_filenames(self) -> Tuple[str, str]:
        """Get expected local filenames for reviews and metadata."""
        if self.dataset_name == "Beauty":
            return "reviews_Beauty_5.json.gz", "meta_Beauty.json.gz"
        elif self.dataset_name == "Video_Games":
            # 2023 format uses .jsonl.gz
            return "Video_Games.jsonl.gz", "meta_Video_Games.jsonl.gz"
        else:
            # Generic fallback
            return (
                f"reviews_{self.dataset_name}_5.json.gz",
                f"meta_{self.dataset_name}.json.gz",
            )

    def get_format(self) -> str:
        """Get the format type for the dataset (legacy or 2023)."""
        if self.dataset_name not in DATASET_URLS:
            return "legacy"  # Default to legacy format
        return DATASET_URLS[self.dataset_name].get("format", "legacy")


# Legacy alias for backward compatibility
BeautyConfig = DatasetConfig


def download_dataset(data_dir: str, dataset_name: str = "Beauty") -> Tuple[str, str]:
    """Return expected filepaths for the specified dataset.

    Actual download is done in the notebook via wget.

    Args:
        data_dir: Directory to store data files
        dataset_name: Name of the dataset (e.g., "Beauty", "Video_Games")

    Returns:
        Tuple of (reviews_path, meta_path)
    """
    ensure_dirs(data_dir)
    cfg = DatasetConfig(dataset_name=dataset_name)
    reviews_file, meta_file = cfg.get_filenames()
    reviews_gz = os.path.join(data_dir, reviews_file)
    meta_gz = os.path.join(data_dir, meta_file)
    return reviews_gz, meta_gz


# Legacy function for backward compatibility
def download_beauty(data_dir: str) -> Tuple[str, str]:
    """Return expected filepaths; actual download is done in the notebook via wget."""
    return download_dataset(data_dir, dataset_name="Beauty")


def _parse_json_lines(path: str) -> List[dict]:
    """Parse JSON lines from a (possibly gzipped) file into a list of dicts.

    Uses stdlib json; if orjson is installed in the environment it will be faster,
    but we keep this simple and robust.
    """
    import json

    opener = gzip.open if path.endswith(".gz") else open
    rows: List[dict] = []
    with opener(path, "rb") as f:
        for raw in f:
            try:
                s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                rows.append(json.loads(s))
            except Exception:
                # Some lines may contain trailing commas or encoding issues; skip them.
                continue
    return rows


def load_reviews_df(reviews_path: str, dataset_format: str = "legacy") -> pd.DataFrame:
    """Load reviews DataFrame from file.

    Args:
        reviews_path: Path to reviews file
        dataset_format: "legacy" for 2014 SNAP format, "2023" for Amazon Reviews 2023

    Returns:
        DataFrame with columns [user_id, item_id, ts]
    """
    rows = _parse_json_lines(reviews_path)
    df = pd.DataFrame(rows)

    # Debug: Print available columns
    print(f"DEBUG: Available columns in {dataset_format} format: {df.columns.tolist()}")

    # Handle different formats
    if dataset_format == "2023":
        # Amazon Reviews 2023 format
        # user_id, parent_asin, timestamp
        required_cols = ["user_id", "parent_asin", "timestamp"]
        df = df[required_cols]
        df = df.rename(columns={"parent_asin": "item_id", "timestamp": "ts"})
    else:
        # Legacy 2014 SNAP format
        # reviewerID, asin, unixReviewTime
        required_cols = ["reviewerID", "asin", "unixReviewTime"]
        df = df[required_cols]
        df = df.rename(
            columns={"reviewerID": "user_id", "asin": "item_id", "unixReviewTime": "ts"}
        )

    df = df.dropna()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df = df.dropna()
    df["ts"] = df["ts"].astype(int)
    return df


def load_meta_df(meta_path: str, dataset_format: str = "legacy") -> pd.DataFrame:
    """Load metadata DataFrame from file.

    Args:
        meta_path: Path to metadata file
        dataset_format: "legacy" for 2014 SNAP format, "2023" for Amazon Reviews 2023

    Returns:
        DataFrame with columns [item_id, title, brand, category_leaf, price, description, features]
    """
    rows = _parse_json_lines(meta_path)
    df = pd.DataFrame(rows)

    if dataset_format == "2023":
        # Amazon Reviews 2023 format
        # Required columns: parent_asin, title
        # Optional: features (list), description (list), details (dict with brand), main_category, price

        # Rename parent_asin to item_id
        if "parent_asin" in df.columns:
            df = df.rename(columns={"parent_asin": "item_id"})

        # Extract brand from details dict if present
        if "details" in df.columns:
            def extract_brand(details):
                if isinstance(details, dict):
                    # Try common brand keys
                    for key in ["Brand", "brand", "Manufacturer", "manufacturer"]:
                        if key in details:
                            return details[key]
                return None
            df["brand"] = df["details"].apply(extract_brand)

        # Use main_category as category_leaf
        if "main_category" in df.columns:
            df["category_leaf"] = df["main_category"]

        # Join description list into single string
        if "description" in df.columns:
            def join_description(desc):
                if isinstance(desc, list):
                    return " ".join(str(d) for d in desc if d)
                return str(desc) if desc else ""
            df["description"] = df["description"].apply(join_description)

        # Join features list into single string
        if "features" in df.columns:
            def join_features(feats):
                if isinstance(feats, list):
                    return " ".join(str(f) for f in feats if f)
                return str(feats) if feats else ""
            df["features"] = df["features"].apply(join_features)

    else:
        # Legacy 2014 SNAP format
        cols = ["asin", "title", "brand", "category", "categories", "price"]
        df = df[[c for c in cols if c in df.columns]].copy()

        # Ensure an 'item_id' column exists
        if "asin" in df.columns and "item_id" not in df.columns:
            df = df.rename(columns={"asin": "item_id"})
        if "item_id" not in df.columns:
            if "id" in df.columns:
                df["item_id"] = df["id"].astype(str)
            else:
                df["item_id"] = pd.Series([None] * len(df))

        # Normalize category: keep last leaf
        def leaf_any(x):
            if isinstance(x, list) and x:
                last = x[-1]
                if isinstance(last, list):
                    last = last[-1] if last else None
                return last
            return None

        if "category" in df.columns and df["category"].notna().any():
            df["category_leaf"] = df["category"].apply(leaf_any)
        elif "categories" in df.columns:
            df["category_leaf"] = df["categories"].apply(leaf_any)
        else:
            df["category_leaf"] = None

    # Normalize string columns
    for c in ("title", "brand", "category_leaf"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    # Normalize price
    if "price" in df.columns:
        def norm_price(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        df["price"] = df["price"].apply(norm_price)

    return df


def filter_and_split(
    reviews: pd.DataFrame, cfg: DatasetConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter users with >= min interactions; split by leave-one-out per user.

    Returns train_df, val_df, test_df with columns [user_id, item_id, ts].
    """
    # Filter by user count
    counts = reviews["user_id"].value_counts()
    keep_users = set(counts[counts >= cfg.min_user_interactions].index)
    df = reviews[reviews["user_id"].isin(keep_users)].copy()
    # Sort histories
    df = df.sort_values(["user_id", "ts"])  # ascending time
    # Leave-one-out split
    def split_user(g: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if len(g) < 3:
            # Should not happen after filtering; fallback: last->test, rest->train
            test = g.tail(1)
            val = g.tail(2).head(1)
            train = g.head(len(g) - 2)
        else:
            test = g.tail(1)
            val = g.tail(2).head(1)
            train = g.head(len(g) - 2)
        # Cap train history length per user
        if len(train) > cfg.max_hist_len:
            # Keep most recent max_hist_len for training sequences
            train = train.tail(cfg.max_hist_len)
        return train, val, test

    trains: List[pd.DataFrame] = []
    vals: List[pd.DataFrame] = []
    tests: List[pd.DataFrame] = []
    for _, g in df.groupby("user_id", sort=False):
        tr, va, te = split_user(g)
        trains.append(tr)
        vals.append(va)
        tests.append(te)
    train_df = pd.concat(trains).reset_index(drop=True)
    val_df = pd.concat(vals).reset_index(drop=True)
    test_df = pd.concat(tests).reset_index(drop=True)
    return train_df, val_df, test_df


def build_id_maps(df_list: List[pd.DataFrame]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build contiguous integer ids for users and items across splits."""
    uid_set = set()
    iid_set = set()
    for df in df_list:
        uid_set.update(df["user_id"].unique().tolist())
        iid_set.update(df["item_id"].unique().tolist())
    user2id = {u: i for i, u in enumerate(sorted(uid_set))}
    item2id = {it: i for i, it in enumerate(sorted(iid_set))}
    return user2id, item2id


def apply_id_maps(df: pd.DataFrame, user2id: Dict[str, int], item2id: Dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["user_idx"] = out["user_id"].map(user2id)
    out["item_idx"] = out["item_id"].map(item2id)
    return out


def save_mappings(
    artifacts_dir: str, user2id: Dict[str, int], item2id: Dict[str, int]
) -> None:
    ensure_dirs(artifacts_dir)
    import json

    with open(os.path.join(artifacts_dir, "user2id.json"), "w") as f:
        json.dump(user2id, f)
    with open(os.path.join(artifacts_dir, "item2id.json"), "w") as f:
        json.dump(item2id, f)
