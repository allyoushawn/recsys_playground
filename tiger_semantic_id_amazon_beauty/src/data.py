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


SNAP_REVIEWS = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
)
SNAP_META = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"
)


@dataclass
class BeautyConfig:
    min_user_interactions: int = 5
    max_hist_len: int = 20


def download_beauty(data_dir: str) -> Tuple[str, str]:
    """Return expected filepaths; actual download is done in the notebook via wget."""
    ensure_dirs(data_dir)
    reviews_gz = os.path.join(data_dir, "reviews_Beauty_5.json.gz")
    meta_gz = os.path.join(data_dir, "meta_Beauty.json.gz")
    return reviews_gz, meta_gz


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


def load_reviews_df(reviews_path: str) -> pd.DataFrame:
    rows = _parse_json_lines(reviews_path)
    df = pd.DataFrame(rows)[["reviewerID", "asin", "unixReviewTime"]]
    df = df.rename(
        columns={"reviewerID": "user_id", "asin": "item_id", "unixReviewTime": "ts"}
    )
    df = df.dropna()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df = df.dropna()
    df["ts"] = df["ts"].astype(int)
    return df


def load_meta_df(meta_path: str) -> pd.DataFrame:
    rows = _parse_json_lines(meta_path)
    cols = ["asin", "title", "brand", "category", "price"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df.rename(columns={"asin": "item_id"})
    # Normalize category: keep last leaf where possible
    def leaf(cat):
        if isinstance(cat, list) and cat:
            # Snap formats can be like [["Beauty", "Hair Care"]] or nested
            last = cat[-1]
            if isinstance(last, list):
                last = last[-1] if last else None
            return last
        return None

    if "category" in df.columns:
        df["category_leaf"] = df["category"].apply(leaf)
    else:
        df["category_leaf"] = None
    for c in ("title", "brand", "category_leaf"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
    if "price" in df.columns:
        # Normalize price if present
        def norm_price(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        df["price"] = df["price"].apply(norm_price)
    return df


def filter_and_split(
    reviews: pd.DataFrame, cfg: BeautyConfig
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

