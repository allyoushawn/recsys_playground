"""Memory-light streaming loader for Amazon review files.

``tiger_semantic_id.src.data.load_reviews_df`` parses every line into a full
dict (retaining review text and all other fields) and only projects afterward,
which OOMs on large 2023 dumps (e.g. Video_Games ~4.6M reviews). This streams
the (possibly gzipped) JSON-lines file and keeps ONLY the three columns the
ranking datamodule needs — ``[user_id, item_id, ts]`` — projecting each record
immediately so the heavy text fields are never retained in memory.

This is the streaming-based "better practice" alternative to simply throwing a
high-RAM Colab runtime at the problem.
"""

from __future__ import annotations

import gzip
import json

import pandas as pd

# (user, item, timestamp) source field names per Amazon dump format.
_FIELDS = {
    "legacy": ("reviewerID", "asin", "unixReviewTime"),  # 2014 SNAP 5-core (Beauty)
    "2023": ("user_id", "parent_asin", "timestamp"),  # McAuley 2023 (Video_Games)
}


def load_reviews_streaming(path: str, dataset_format: str = "legacy", max_users: int = 0) -> pd.DataFrame:
    """Stream a (gzipped) reviews file into a ``[user_id, item_id, ts]`` frame.

    Only the three needed fields are kept per record — review text and other
    columns are dropped at parse time, so peak memory is ~the 3 columns rather
    than the full deserialized corpus.

    Args:
        path: path to the reviews file (``.json.gz`` / ``.jsonl.gz`` / plain).
        dataset_format: ``"legacy"`` (2014 SNAP) or ``"2023"`` (McAuley 2023).
        max_users: if > 0, keep only the first ``max_users`` distinct users
            (by sorted id) — a deterministic subsample for quick runs.
    """
    if dataset_format not in _FIELDS:
        raise ValueError(f"unknown dataset_format {dataset_format!r}; options: {sorted(_FIELDS)}")
    u_key, i_key, t_key = _FIELDS[dataset_format]
    opener = gzip.open if path.endswith(".gz") else open

    users, items, tss = [], [], []
    with opener(path, "rb") as f:
        for raw in f:
            try:
                d = json.loads(raw)
            except Exception:
                continue  # skip malformed lines (encoding, trailing commas, ...)
            u, i, t = d.get(u_key), d.get(i_key), d.get(t_key)
            if u is None or i is None or t is None:
                continue
            users.append(u)
            items.append(str(i))
            tss.append(t)

    df = pd.DataFrame({"user_id": users, "item_id": items, "ts": tss})
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["ts"] = df["ts"].astype("int64")
    if max_users:
        keep = set(sorted(df["user_id"].unique())[:max_users])
        df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    return df
