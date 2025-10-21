import numpy as np
import pandas as pd

from tiger_semantic_id_amazon_beauty.src.visualize import (
    plot_c1_category_distribution,
    plot_hierarchy_c1_c2,
)


def test_plots_run_without_error(tmp_path):
    codes = np.array([[0, 0, 1], [0, 1, 2], [1, 0, 1], [1, 1, 1]], dtype=np.int64)
    items_df = pd.DataFrame(
        {
            "item_idx": [0, 1, 2, 3],
            "category_leaf": ["Makeup", "Makeup", "Skin", "Hair"],
        }
    )
    fig1 = plot_c1_category_distribution(codes, items_df, top_n=3)
    fig2 = plot_hierarchy_c1_c2(codes, items_df, c1_values=[0, 1])
    # Save to ensure the figures can be written
    fig1.savefig(tmp_path / "c1_cat.png")
    fig2.savefig(tmp_path / "hier.png")
