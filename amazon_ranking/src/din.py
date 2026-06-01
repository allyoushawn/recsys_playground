"""Deep Interest Network (DIN, Zhou et al. 2018) for the amazon_ranking layer.

A compact, self-contained DIN for implicit-feedback CTR/ranking on top of
:class:`amazon_ranking.src.datamodule.SequenceRankingDataModule` examples
(``{user_idx, history, target_idx, label}``). Trains on CPU or GPU.

Design notes / deviations from the paper, kept deliberate for a teaching baseline:
- The local activation unit scores each history item against the target, and the
  history is pooled with **masked softmax** weights (the paper uses un-normalized
  weights to preserve intensity; softmax is used here for numerical stability).
- A single item embedding table is shared by history and target; index
  ``num_items`` is the padding id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import metrics as M


def pad_histories(
    histories: Sequence[Sequence[int]], max_hist_len: int, pad_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad histories to ``max_hist_len``.

    Returns ``(ids [B, L] long, mask [B, L] float)`` where mask is 1 for real
    history positions and 0 for padding. The most recent ``max_hist_len`` items
    are kept.
    """
    batch = len(histories)
    ids = np.full((batch, max_hist_len), pad_idx, dtype=np.int64)
    mask = np.zeros((batch, max_hist_len), dtype=np.float32)
    for i, hist in enumerate(histories):
        h = list(hist)[-max_hist_len:]
        if h:
            ids[i, : len(h)] = h
            mask[i, : len(h)] = 1.0
    return torch.from_numpy(ids), torch.from_numpy(mask)


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class LocalActivationUnit(nn.Module):
    """Scores each history item against the target item."""

    def __init__(self, embed_dim: int, hidden: Sequence[int] = (36,)) -> None:
        super().__init__()
        # Input per history slot: [hist, target, hist-target, hist*target].
        self.mlp = _mlp(embed_dim * 4, hidden, 1)

    def forward(self, target: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        # target [B, E], hist [B, L, E] -> scores [B, L]
        length = hist.size(1)
        tgt = target.unsqueeze(1).expand(-1, length, -1)
        feats = torch.cat([hist, tgt, hist - tgt, hist * tgt], dim=-1)
        return self.mlp(feats).squeeze(-1)


class DIN(nn.Module):
    """Deep Interest Network for binary CTR over (history, target) pairs."""

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 32,
        attn_hidden: Sequence[int] = (36,),
        mlp_hidden: Sequence[int] = (80, 40),
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.pad_idx = int(num_items)
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=self.pad_idx)
        self.attn = LocalActivationUnit(embed_dim, attn_hidden)
        # MLP over [interest, target, interest*target].
        self.mlp = _mlp(embed_dim * 3, mlp_hidden, 1)

    def forward(
        self, hist_ids: torch.Tensor, hist_mask: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return logits [B]. ``hist_ids`` [B, L], ``hist_mask`` [B, L], ``target_ids`` [B]."""
        hist = self.item_emb(hist_ids)  # [B, L, E]
        target = self.item_emb(target_ids)  # [B, E]
        scores = self.attn(target, hist)  # [B, L]
        scores = scores.masked_fill(hist_mask == 0, float("-inf"))
        has_hist = hist_mask.sum(dim=1, keepdim=True) > 0  # [B, 1]
        weights = torch.softmax(scores, dim=1)
        # Fully-masked (empty-history) rows are all -inf -> softmax = NaN. Zero
        # ONLY those rows. A NaN in a row that *has* history signals real
        # numerical instability, so it is left to propagate rather than be
        # silently swallowed (the prior `nan_to_num` hid both cases alike).
        weights = torch.where(has_hist, weights, torch.zeros_like(weights))
        weights = weights.unsqueeze(-1)  # [B, L, 1]
        interest = (weights * hist).sum(dim=1)  # [B, E]; empty-history rows -> 0
        feats = torch.cat([interest, target, interest * target], dim=-1)
        return self.mlp(feats).squeeze(-1)


@dataclass
class DINTrainConfig:
    embed_dim: int = 32
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 0


def _iter_batches(
    examples: Sequence[Dict[str, object]], batch_size: int, max_hist_len: int, pad_idx: int, device
):
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        hist_ids, hist_mask = pad_histories(
            [ex["history"] for ex in chunk], max_hist_len, pad_idx
        )
        target = torch.tensor([int(ex["target_idx"]) for ex in chunk], dtype=torch.long)
        label = torch.tensor([float(ex["label"]) for ex in chunk], dtype=torch.float32)
        yield (
            hist_ids.to(device),
            hist_mask.to(device),
            target.to(device),
            label.to(device),
        )


def train_din(
    model: DIN,
    examples: Sequence[Dict[str, object]],
    max_hist_len: int,
    cfg: DINTrainConfig,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train ``model`` in place with BCE loss. Returns first/last epoch mean loss.

    Note: ``cfg.seed`` controls only the *batch shuffle order* and torch's
    optimizer/dropout RNG from this point on. It does **not** make parameter
    initialization reproducible, because ``model`` is already constructed by the
    caller. For end-to-end reproducibility seed torch *before* building the
    ``DIN`` (e.g. ``torch.manual_seed(s); model = DIN(...)``).
    """
    torch.manual_seed(cfg.seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(cfg.seed)
    order = list(range(len(examples)))
    epoch_losses: List[float] = []
    for _ in range(cfg.epochs):
        rng.shuffle(order)
        shuffled = [examples[i] for i in order]
        model.train()
        total, n_batches = 0.0, 0
        for hist_ids, hist_mask, target, label in _iter_batches(
            shuffled, cfg.batch_size, max_hist_len, model.pad_idx, device
        ):
            opt.zero_grad()
            logits = model(hist_ids, hist_mask, target)
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batches += 1
        epoch_losses.append(total / max(n_batches, 1))
    return {
        "first_epoch_loss": epoch_losses[0] if epoch_losses else float("nan"),
        "last_epoch_loss": epoch_losses[-1] if epoch_losses else float("nan"),
    }


def _resolve_device(model: DIN, device):
    """Return ``device`` if given, else the device the model's params live on.

    Defaulting to the model's own device avoids a silent CPU/GPU mismatch when a
    model trained on CUDA is scored with the old hard-coded ``device="cpu"``.
    """
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:  # parameter-less model (shouldn't happen for DIN)
        return "cpu"


@torch.no_grad()
def score_candidates(
    model: DIN,
    history: Sequence[int],
    candidates: Sequence[int],
    max_hist_len: int,
    device=None,
) -> np.ndarray:
    """Return the model logit for each candidate under ``history`` as a 1-D array.

    The single scoring primitive: ``rank_candidates`` and ``evaluate_ranking``
    both derive their outputs from one call to this, so the model runs exactly
    once per query.
    """
    model.eval()
    device = _resolve_device(model, device)
    cand = [int(c) for c in candidates]
    hist_ids, hist_mask = pad_histories([list(history)] * len(cand), max_hist_len, model.pad_idx)
    target = torch.tensor(cand, dtype=torch.long)
    logits = model(hist_ids.to(device), hist_mask.to(device), target.to(device))
    return logits.cpu().numpy()


@torch.no_grad()
def rank_candidates(
    model: DIN, history: Sequence[int], candidates: Sequence[int], max_hist_len: int, device=None
) -> List[int]:
    """Return ``candidates`` sorted by descending model score (best first)."""
    cand = [int(c) for c in candidates]
    scores = score_candidates(model, history, cand, max_hist_len, device)
    order = np.argsort(-scores, kind="stable")
    return [cand[i] for i in order]


@torch.no_grad()
def evaluate_ranking(
    model: DIN,
    eval_examples: Dict[int, Dict[str, object]],
    max_hist_len: int,
    ks: Sequence[int] = (5, 10),
    device=None,
) -> Dict[str, float]:
    """Mean Recall@k / NDCG@k / MRR@k + sampled AUC over the eval candidate sets."""
    model.eval()
    per_query: List[Dict[str, float]] = []
    for entry in eval_examples.values():
        positive = int(entry["positive"])
        candidates = [int(c) for c in entry["candidates"]]
        history = [int(i) for i in entry.get("history", [])]
        # One forward pass per query; ranking AND sampled AUC come from it.
        scores = score_candidates(model, history, candidates, max_hist_len, device)
        order = np.argsort(-scores, kind="stable")
        ranked = [candidates[i] for i in order]
        pos_score = float(scores[0])  # candidates[0] is the positive
        neg_scores = scores[1:].tolist()
        row: Dict[str, float] = {"sampled_auc": M.sampled_auc(pos_score, neg_scores)}
        for k in ks:
            row[f"recall@{k}"] = M.recall_at_k(ranked, positive, k)
            row[f"ndcg@{k}"] = M.ndcg_at_k(ranked, positive, k)
            row[f"mrr@{k}"] = M.mrr_at_k(ranked, positive, k)
        per_query.append(row)
    return M.mean_metrics(per_query)
