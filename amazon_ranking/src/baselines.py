"""Non-attention ranking baselines for a fair comparison against DIN (backlog E5).

Each model consumes the **same** ``SequenceRankingDataModule`` examples as DIN and
exposes the identical forward signature

    forward(hist_ids [B,L], hist_mask [B,L], target_ids [B]) -> logits [B]

plus a ``pad_idx`` attribute, so they reuse DIN's training/eval harness
(``train_din`` / ``score_candidates`` / ``rank_candidates`` / ``evaluate_ranking``)
unchanged. This guarantees identical data, splits, negatives, and metrics across
all models — the whole point of a fair comparison.

Sequence handling (the one design choice): DIN pools history with *target-aware
attention*; these baselines pool history by a **mask-aware mean** into a single
"interest" vector, then model feature interactions over the two fields
``[interest, target]``. So the comparison isolates "attention pooling +
interaction MLP" (DIN) vs "mean pooling + {FM, cross network, plain MLP}".
Empty histories pool to a zero interest vector, matching DIN.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .din import _mlp  # shared MLP builder (Linear/ReLU stack, no final activation)


class _SeqRankModel(nn.Module):
    """Shared embedding table + mask-aware mean pooling of the history."""

    def __init__(self, num_items: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.pad_idx = int(num_items)  # same convention as DIN: index num_items is padding
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=self.pad_idx)
        self.embed_dim = int(embed_dim)

    def _pool(self, hist_ids: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        """Mask-aware mean of history item embeddings. Empty rows -> zeros."""
        hist = self.item_emb(hist_ids)  # [B, L, E]
        w = hist_mask.unsqueeze(-1)  # [B, L, 1]
        summed = (hist * w).sum(dim=1)  # [B, E]
        cnt = w.sum(dim=1).clamp(min=1.0)  # [B, 1]; empty history -> divide by 1 -> 0
        return summed / cnt


class MeanPoolMLP(_SeqRankModel):
    """DIN minus attention: mean-pooled interest + target through the same-shape MLP.

    The cleanest ablation — isolates the value of target-aware attention by keeping
    everything else (embedding, ``[interest, target, interest*target]`` features, MLP)
    identical to DIN and swapping only attention pooling for mean pooling.
    """

    def __init__(
        self, num_items: int, embed_dim: int = 32, mlp_hidden: Sequence[int] = (80, 40)
    ) -> None:
        super().__init__(num_items, embed_dim)
        self.mlp = _mlp(embed_dim * 3, mlp_hidden, 1)

    def forward(
        self, hist_ids: torch.Tensor, hist_mask: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        interest = self._pool(hist_ids, hist_mask)  # [B, E]
        target = self.item_emb(target_ids)  # [B, E]
        feats = torch.cat([interest, target, interest * target], dim=-1)
        return self.mlp(feats).squeeze(-1)


class DeepFM(_SeqRankModel):
    """DeepFM over the two fields ``[interest, target]``.

    - First-order: a linear term over the concatenated field vectors.
    - Second-order (FM): ``0.5 * ((sum_i v_i)^2 - sum_i v_i^2)`` summed over the
      embedding dims — with two fields this is exactly the cross term
      ``<interest, target>``, computed via the standard FM identity.
    - Deep: an MLP over the concatenated fields.
    Output logit = linear + fm + deep.
    """

    def __init__(
        self, num_items: int, embed_dim: int = 32, mlp_hidden: Sequence[int] = (80, 40)
    ) -> None:
        super().__init__(num_items, embed_dim)
        self.linear = nn.Linear(embed_dim * 2, 1)  # first-order term
        self.dnn = _mlp(embed_dim * 2, mlp_hidden, 1)  # deep component

    def forward(
        self, hist_ids: torch.Tensor, hist_mask: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        interest = self._pool(hist_ids, hist_mask)  # [B, E]
        target = self.item_emb(target_ids)  # [B, E]
        fields = torch.stack([interest, target], dim=1)  # [B, 2, E]
        sum_sq = fields.sum(dim=1) ** 2  # [B, E]
        sq_sum = (fields ** 2).sum(dim=1)  # [B, E]
        fm = 0.5 * (sum_sq - sq_sum).sum(dim=1)  # [B]
        x = torch.cat([interest, target], dim=-1)  # [B, 2E]
        lin = self.linear(x).squeeze(-1)  # [B]
        deep = self.dnn(x).squeeze(-1)  # [B]
        return lin + fm + deep


class _CrossNet(nn.Module):
    """DCN cross network: ``x_{l+1} = x0 * (x_l . w_l) + b_l + x_l``."""

    def __init__(self, in_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.w = nn.ParameterList(
            [nn.Parameter(torch.randn(in_dim) * 0.01) for _ in range(num_layers)]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(in_dim)) for _ in range(num_layers)]
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for w, b in zip(self.w, self.b):
            xw = (x * w).sum(dim=1, keepdim=True)  # [B, 1] = x_l . w_l
            x = x0 * xw + b + x  # [B, in_dim]
        return x


class DCN(_SeqRankModel):
    """Deep & Cross Network over the concatenated ``[interest, target]`` features.

    A cross network models explicit bounded-degree feature interactions; a parallel
    deep MLP models implicit ones; their outputs are concatenated into a final head.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 32,
        cross_layers: int = 2,
        mlp_hidden: Sequence[int] = (80, 40),
    ) -> None:
        super().__init__(num_items, embed_dim)
        in_dim = embed_dim * 2
        self.cross = _CrossNet(in_dim, cross_layers)
        deep_out = mlp_hidden[-1] if mlp_hidden else in_dim
        self.deep = _mlp(in_dim, mlp_hidden[:-1], deep_out)  # deep tower -> deep_out dims
        self.head = nn.Linear(in_dim + deep_out, 1)  # combine cross + deep

    def forward(
        self, hist_ids: torch.Tensor, hist_mask: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        interest = self._pool(hist_ids, hist_mask)  # [B, E]
        target = self.item_emb(target_ids)  # [B, E]
        x0 = torch.cat([interest, target], dim=-1)  # [B, 2E]
        cross_out = self.cross(x0)  # [B, 2E]
        deep_out = self.deep(x0)  # [B, deep_out]
        return self.head(torch.cat([cross_out, deep_out], dim=-1)).squeeze(-1)


# Registry so a runner can build models by name with a shared signature.
MODEL_REGISTRY = {
    "meanpool": MeanPoolMLP,
    "deepfm": DeepFM,
    "dcn": DCN,
}


def build_baseline(name: str, num_items: int, embed_dim: int = 32) -> _SeqRankModel:
    """Construct a baseline by name (``meanpool`` | ``deepfm`` | ``dcn``)."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown baseline {name!r}; options: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_items=num_items, embed_dim=embed_dim)
