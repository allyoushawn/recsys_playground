from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn
from sklearn.metrics import roc_auc_score


_bce = nn.BCEWithLogitsLoss(reduction="mean")


def bin_metrics(
    logits: Tensor,
    labels: Tensor,
    pos_weight: Optional[float] = None,
) -> Dict[str, float]:
    """Compute loss, accuracy, and AUROC for binary logits and 0/1 labels.

    Args:
        logits: Tensor of shape (N,)
        labels: Tensor of shape (N,) with values in {0,1}
        pos_weight: Optional scalar multiplier for positive class in BCEWithLogitsLoss

    Returns:
        dict with keys: loss, acc, auc (auc may be NaN if only one class present)
    """
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=logits.device))
    else:
        criterion = _bce

    labels_f = labels.to(dtype=logits.dtype)
    loss = criterion(logits.view(-1), labels_f.view(-1))

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(dtype=labels.dtype)
        acc = (preds == labels).float().mean().item()
        # AUC
        auc = float("nan")
        try:
            y_true = labels.detach().cpu().numpy().astype(np.int32)
            y_score = probs.detach().cpu().numpy().astype(np.float64)
            # roc_auc_score requires both classes present
            if len(np.unique(y_true)) == 2:
                auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass

    return {"loss": float(loss.item()), "acc": float(acc), "auc": auc}

