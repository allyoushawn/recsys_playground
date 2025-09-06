from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_name(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name).strip().lower())
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[.]+$", "", s)
    return s


def find_income_column(df: pd.DataFrame) -> str:
    candidates = []
    for c in df.columns:
        n = normalize_name(c)
        if any(k in n for k in ["income", "class", ">50k", "<=50k"]):
            candidates.append(c)
    if not candidates:
        raise ValueError("Could not find income/class target column in dataset")
    # Prefer exact 'income' or 'class' style names
    for key in ["income", "class"]:
        for c in candidates:
            if normalize_name(c) == key:
                return c
    return candidates[0]


def find_marital_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        n = normalize_name(c)
        if "marital" in n:
            return c
    raise ValueError("Could not find marital-status column in dataset")


def map_income_to_binary(series: pd.Series) -> np.ndarray:
    vals = series.astype(str).str.strip().str.lower().str.replace(".", "", regex=False)
    # True if contains '>50k' token (with or without period)
    y = vals.str.contains("50k") & vals.str.contains(">")
    return y.astype(np.uint8).to_numpy()


def map_never_married_to_binary(series: pd.Series) -> np.ndarray:
    vals = series.astype(str).str.strip().str.lower()
    # Normalize hyphen/period variants
    vals = vals.str.replace("-", " ", regex=False).str.replace(".", "", regex=False)
    y = (vals == "never married")
    return y.astype(np.uint8).to_numpy()


def try_fetch_openml() -> pd.DataFrame:
    names = [
        "Census-Income (KDD)",
        "census-income (kdd)",
        "census income (kdd)",
        "census-income-kdd",
        "census income kdd",
    ]
    last_err: Optional[Exception] = None
    for nm in names:
        try:
            bunch = fetch_openml(name=nm, version="active", as_frame=True)
            if hasattr(bunch, "frame") and bunch.frame is not None:
                return bunch.frame.copy()
            # Fallback to join data/target
            df = bunch.data
            if hasattr(bunch, "target") and bunch.target is not None:
                df = pd.concat([df, bunch.target], axis=1)
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to fetch dataset from OpenML")


def fetch_uci_census_kdd() -> pd.DataFrame:
    # Known UCI mirrors/paths for Census-Income (KDD)
    base_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-kdd/",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/",
    ]
    files = [
        ("census-income.data.gz", True),
        ("census-income.data", False),
    ]
    test_files = [
        ("census-income.test.gz", True),
        ("census-income.test", False),
    ]

    # Column names from UCI schema (order matters)
    columns = [
        "age",
        "class-of-worker",
        "detailed-industry-recode",
        "detailed-occupation-recode",
        "education",
        "wage-per-hour",
        "enroll-in-edu-inst-last-wk",
        "marital-status",
        "major-industry-code",
        "major-occupation-code",
        "race",
        "hispanic-origin",
        "sex",
        "member-of-a-labor-union",
        "reason-for-unemployment",
        "full-or-part-time-employment-status",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "tax-filer-status",
        "region-of-previous-residence",
        "state-of-previous-residence",
        "detailed-household-and-family-status",
        "detailed-household-summary-in-household",
        "instance-weight",
        "migration-code-change-in-msa",
        "migration-code-change-in-reg",
        "migration-code-move-within-reg",
        "live-in-this-house-1-year-ago",
        "migration-prev-res-in-sunbelt",
        "num-persons-worked-for-employer",
        "family-members-under-18",
        "country-of-birth-father",
        "country-of-birth-mother",
        "country-of-birth-self",
        "citizenship",
        "own-business-or-self-employed",
        "fill-inc-questionnaire-for-veterans-admin",
        "veterans-benefits",
        "weeks-worked-in-year",
        "year",
        "income",
    ]

    def _read_url(url: str, compressed: bool) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(
                url,
                header=None,
                names=columns,
                na_values=["?", " ?"],
                sep=",",
                skipinitialspace=True,
                compression="gzip" if compressed else None,
                engine="python",
                dtype="object",
            )
            return df
        except Exception:
            return None

    data_df = None
    test_df = None
    for base in base_urls:
        if data_df is None:
            for fname, gz in files:
                data_df = _read_url(base + fname, gz)
                if data_df is not None:
                    break
        if test_df is None:
            for fname, gz in test_files:
                test_df = _read_url(base + fname, gz)
                if test_df is not None:
                    break
    if data_df is None:
        raise RuntimeError("Failed to download UCI Census-Income (KDD) data file")
    if test_df is None:
        # If test missing, proceed with just data
        df = data_df
    else:
        df = pd.concat([data_df, test_df], axis=0, ignore_index=True)
    # Strip whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Heuristically coerce numeric-looking columns to numeric
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            non_na_ratio = 1.0 - (conv.isna().mean())
            if non_na_ratio > 0.98:  # mostly numeric
                df[c] = conv
    return df


@dataclass
class PrepConfig:
    output_dir: str
    test_size: float
    val_size: float
    batch_size: int
    num_workers: int
    onehot_min_freq: int
    seed: int


def build_pipeline(numeric_cols: List[str], cat_cols: List[str], min_freq: int) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_freq,
                    sparse_output=False,
                ),
            ),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return pre


def pipeline_metadata(pre: ColumnTransformer, numeric_cols: List[str], cat_cols: List[str]) -> Dict:
    meta: Dict = {
        "original_columns": numeric_cols + cat_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "sklearn_version": sklearn.__version__,
    }
    # Numeric stats
    scaler_meta = {"means": None, "scales": None}
    if len(numeric_cols) > 0:
        scaler = pre.named_transformers_["num"].named_steps["scaler"]
        if hasattr(scaler, "mean_"):
            scaler_meta["means"] = scaler.mean_.tolist()
        if hasattr(scaler, "scale_"):
            scaler_meta["scales"] = scaler.scale_.tolist()
    meta["scaler"] = scaler_meta
    # Categorical categories
    cats: Dict[str, List[str]] = {}
    if len(cat_cols) > 0:
        onehot = pre.named_transformers_["cat"].named_steps["onehot"]
        if hasattr(onehot, "categories_"):
            for c, arr in zip(cat_cols, onehot.categories_):
                cats[c] = arr.astype(str).tolist()
    meta["onehot"] = {"categories": cats}
    # Hash params
    params_json = json.dumps(pre.get_params(deep=True), sort_keys=True, default=str)
    meta["pipeline_hash"] = hashlib.sha256(params_json.encode("utf-8")).hexdigest()
    return meta


def summarize_split(name: str, y_income: np.ndarray, y_never: np.ndarray) -> str:
    def dist(arr: np.ndarray) -> Dict[int, int]:
        unique, counts = np.unique(arr, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    return (
        f"{name}: n={len(y_income)} | income>50K {dist(y_income)} | never-married {dist(y_never)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Census-Income (KDD) for PLE-style MTL")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--onehot_min_freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PrepConfig(
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        onehot_min_freq=args.onehot_min_freq,
        seed=args.seed,
    )

    set_seeds(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Fetching Census-Income (KDD) from OpenML…")
    try:
        df = try_fetch_openml()
        source = "openml"
    except Exception as e:  # noqa: BLE001
        print(f"OpenML fetch failed: {e}\nFalling back to UCI…")
        df = fetch_uci_census_kdd()
        source = "uci"

    print(f"Loaded dataset from {source}. Shape: {df.shape}")

    income_col = find_income_column(df)
    marital_col = find_marital_column(df)
    print(f"Targets detected: income='{income_col}', marital='{marital_col}'")

    y_income = map_income_to_binary(df[income_col])
    y_never = map_never_married_to_binary(df[marital_col])

    # Drop targets from features
    X_df = df.drop(columns=[income_col, marital_col], errors="ignore")

    # Auto infer column types
    cat_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    print(f"Columns: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    # Split data
    X_trainval, X_test, y_inc_trainval, y_inc_test, y_nv_trainval, y_nv_test = train_test_split(
        X_df, y_income, y_never, test_size=cfg.test_size, random_state=cfg.seed, stratify=y_income
    )
    rel_val = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_inc_train, y_inc_val, y_nv_train, y_nv_val = train_test_split(
        X_trainval,
        y_inc_trainval,
        y_nv_trainval,
        test_size=rel_val,
        random_state=cfg.seed,
        stratify=y_inc_trainval,
    )

    print(summarize_split("Train", y_inc_train, y_nv_train))
    print(summarize_split("Val", y_inc_val, y_nv_val))
    print(summarize_split("Test", y_inc_test, y_nv_test))

    # Build and fit pipeline on train only
    pre = build_pipeline(num_cols, cat_cols, min_freq=cfg.onehot_min_freq)
    X_tr = pre.fit_transform(X_train)
    X_va = pre.transform(X_val)
    X_te = pre.transform(X_test)

    # Ensure dense float32
    X_tr = np.asarray(X_tr, dtype=np.float32)
    X_va = np.asarray(X_va, dtype=np.float32)
    X_te = np.asarray(X_te, dtype=np.float32)

    dim = X_tr.shape[1]
    print(f"Final feature dimensionality: {dim}")

    # Validations
    for name, arr in [("X_train", X_tr), ("X_val", X_va), ("X_test", X_te)]:
        if arr.dtype != np.float32:
            raise TypeError(f"{name} must be float32")
        if np.isnan(arr).any():
            raise ValueError(f"NaNs found in {name}")
    for name, arr in [
        ("y_income_train", y_inc_train),
        ("y_income_val", y_inc_val),
        ("y_income_test", y_inc_test),
        ("y_nevermarried_train", y_nv_train),
        ("y_nevermarried_val", y_nv_val),
        ("y_nevermarried_test", y_nv_test),
    ]:
        if not set(np.unique(arr)).issubset({0, 1}):
            raise ValueError(f"Labels for {name} must be in {{0,1}}")

    # Save arrays
    np.save(os.path.join(cfg.output_dir, "X_train.npy"), X_tr)
    np.save(os.path.join(cfg.output_dir, "X_val.npy"), X_va)
    np.save(os.path.join(cfg.output_dir, "X_test.npy"), X_te)
    np.save(os.path.join(cfg.output_dir, "y_income_train.npy"), y_inc_train.astype(np.uint8))
    np.save(os.path.join(cfg.output_dir, "y_income_val.npy"), y_inc_val.astype(np.uint8))
    np.save(os.path.join(cfg.output_dir, "y_income_test.npy"), y_inc_test.astype(np.uint8))
    np.save(os.path.join(cfg.output_dir, "y_nevermarried_train.npy"), y_nv_train.astype(np.uint8))
    np.save(os.path.join(cfg.output_dir, "y_nevermarried_val.npy"), y_nv_val.astype(np.uint8))
    np.save(os.path.join(cfg.output_dir, "y_nevermarried_test.npy"), y_nv_test.astype(np.uint8))

    # Save feature metadata
    meta = pipeline_metadata(pre, num_cols, cat_cols)
    with open(os.path.join(cfg.output_dir, "feature_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Import dataset utilities and do a one-batch sanity check
    from dataset import CensusKDDDataset, make_dataloaders

    loaders = make_dataloaders(
        data_dir=cfg.output_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    batch = next(iter(loaders["train"]))
    x, yi, yn = batch["x"], batch["y_income"], batch["y_never_married"]
    print(f"Batch shapes: x={tuple(x.shape)}, y_income={tuple(yi.shape)}, y_never={tuple(yn.shape)}")

    # Compact report
    def few_rows(df_: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        return df_.head(n)

    print("\n=== Summary ===")
    print(f"Source: {source}")
    print(
        f"Split sizes: train={len(y_inc_train)}, val={len(y_inc_val)}, test={len(y_inc_test)} | X_dim={dim}"
    )
    print(summarize_split("Train", y_inc_train, y_nv_train))
    print(summarize_split("Val", y_inc_val, y_nv_val))
    print(summarize_split("Test", y_inc_test, y_nv_test))
    # Show first 3 categorical feature names (if available)
    if len(cat_cols) > 0:
        print("First 3 categorical features:", cat_cols[:3])
    # Show a few raw rows for sanity
    print("Sample rows:")
    print(few_rows(df[[c for c in df.columns if c != income_col]][:3]))


if __name__ == "__main__":
    main()
