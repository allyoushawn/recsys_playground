from __future__ import annotations

from typing import Dict, Iterable

import matplotlib
matplotlib.use("Agg")  # ensure non-interactive backend for headless test environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_c1_category_distribution(codes: np.ndarray, items_df: pd.DataFrame, top_n: int = 10):
    """Bar chart showing category distribution per c1 over top-N categories."""
    assert codes.shape[1] >= 1
    df = items_df.copy()
    df = df.set_index("item_idx")
    c1 = codes[:, 0]
    cats = df["category_leaf"].fillna("unknown")
    data = pd.DataFrame({"c1": c1, "category": cats.values})
    top_cats = data["category"].value_counts().head(top_n).index
    data = data[data["category"].isin(top_cats)]
    pivot = data.pivot_table(index="c1", columns="category", aggfunc=len, fill_value=0)
    pivot.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.ylabel("Count")
    plt.title("Category distribution per c1 (top categories)")
    plt.tight_layout()
    return plt.gcf()


def plot_hierarchy_c1_c2(codes: np.ndarray, items_df: pd.DataFrame, c1_values: Iterable[int]):
    """For chosen c1 values, stacked bars over categories by c2 to show refinement."""
    assert codes.shape[1] >= 2
    df = items_df.copy().set_index("item_idx")
    cats = df["category_leaf"].fillna("unknown")
    c1 = codes[:, 0]
    c2 = codes[:, 1]
    data = pd.DataFrame({"c1": c1, "c2": c2, "category": cats.values})
    fig, axes = plt.subplots(nrows=1, ncols=len(list(c1_values)), figsize=(5 * len(list(c1_values)), 4))
    if len(list(c1_values)) == 1:
        axes = [axes]
    for ax, c1v in zip(axes, c1_values):
        sub = data[data["c1"] == c1v]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="c2", columns="category", aggfunc=len, fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"c1={c1v}")
        ax.set_ylabel("Count")
    plt.tight_layout()
    return fig
