"""Self-Attentive Sequential Recommendation (SASRec, Kang & McAuley 2018) adapted
for (history, target) binary scoring.

The original SASRec produces an ordered list of next-item probabilities by running
causal (left-to-right) self-attention over the user's interaction sequence and
scoring every item against the representation at each position.  Here we adapt it
to the shared (history, target) binary-scoring interface used by DIN and the
baselines: the user representation is the hidden state at the *last valid history
position*, and the score is the dot product of that representation with the target
item embedding.

Design notes
------------
- Causal (upper-triangular) attention mask is the defining feature of SASRec and
  is combined with the key-padding mask so that padded positions contribute
  neither as queries nor as keys.
- Both masks use **bool** dtype (True = disallowed) to avoid PyTorch deprecation
  warnings about mismatched mask dtypes.
- Empty-history rows (all-padding) safely produce a zero user representation
  without NaNs: the causal mask always allows self-attention on the diagonal
  (each position can attend to itself) so no query row is ever fully masked,
  preventing softmax from producing NaN.  The user representation for
  empty-history rows is then explicitly zeroed out.
- ``pad_idx = num_items`` follows the shared convention; ``item_emb`` is inherited
  from :class:`~amazon_ranking.src.baselines._SeqRankModel`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from amazon_ranking.src.baselines import _SeqRankModel


class _SASRecBlock(nn.Module):
    """One causal Transformer block for SASRec (pre-norm).

    Applies masked (causal) multi-head self-attention followed by a position-wise
    FFN with residual connections and LayerNorm.  The causal mask prevents each
    position from attending to future positions — SASRec's defining property.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [B, L, E].

        Parameters
        ----------
        attn_mask:         [L, L] bool  — causal upper-triangular mask; True = disallowed.
                           The diagonal is always False so every query can self-attend,
                           preventing all-masked softmax rows (NaN guard).
        key_padding_mask:  [B, L] bool  — True = padding (ignore)
        """
        # Self-attention sub-layer (pre-norm).
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.drop(x) + residual
        # FFN sub-layer (pre-norm).
        residual = x
        x = self.norm2(x)
        x = self.drop(self.ff(x)) + residual
        return x


class SASRec(_SeqRankModel):
    """Self-Attentive Sequential Recommendation adapted for binary CTR scoring.

    Inherits the shared item embedding table from
    :class:`~amazon_ranking.src.baselines._SeqRankModel`.

    Forward signature (same as DIN / all baselines)::

        forward(hist_ids [B, L], hist_mask [B, L], target_ids [B]) -> logits [B]

    The user representation is the hidden state at the **last valid history
    position** (determined by ``hist_mask``).  The logit is the dot product of
    this representation with the target item embedding.

    Parameters
    ----------
    num_items:
        Vocabulary size.  Pad index is ``num_items`` (inherited).
    embed_dim:
        Item embedding dimension (= Transformer hidden size).
    n_layers:
        Number of stacked causal Transformer blocks.
    n_heads:
        Number of attention heads.  Must divide ``embed_dim``.
    dropout:
        Dropout probability inside MHA and FFN.
    max_len:
        Maximum history length.  Positional embedding table size.
        Sequences longer than ``max_len`` are silently truncated from the left.
    **kwargs:
        Extra keyword arguments are accepted and silently ignored to allow the
        shared ``build_model`` registry to forward unknown kwargs.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 1,
        dropout: float = 0.0,
        max_len: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(num_items, embed_dim)
        self.max_len = int(max_len)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList(
            [_SASRecBlock(embed_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # Optional bias term for the dot-product score (as in the original paper).
        self.score_bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Return a bool causal mask of shape [seq_len, seq_len].

        True means *disallowed* (future positions).  Specifically, position (i, j)
        is True when j > i (upper-triangular, diagonal=1).

        NaN-safety guarantee: the diagonal is always False, so every query
        position can at minimum self-attend, ensuring softmax never sees an
        all-masked row (which would produce NaN outputs in nn.MultiheadAttention).

        Using bool dtype matches ``key_padding_mask`` and avoids PyTorch's
        UserWarning about mismatched mask dtypes.
        """
        # True in upper triangle (j > i) = disallowed; diagonal (j == i) = False = allowed.
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def _last_valid_repr(
        self, hidden: torch.Tensor, hist_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract the hidden state at the last valid (non-padding) position.

        Parameters
        ----------
        hidden:    [B, L, E]
        hist_mask: [B, L] float (1 = valid, 0 = padding)

        Returns
        -------
        u: [B, E]  — zero vector for all-padding rows (empty history).
        """
        B, L, E = hidden.shape
        # last_idx[b] = index of the last 1 in hist_mask[b], or -1 if none.
        # (hist_mask * torch.arange(L)) gives position index for valid slots, 0 elsewhere.
        # Taking the max gives the last valid position.  For all-zero rows the max
        # is also 0, so we use a has_hist guard to zero those rows out.
        positions = torch.arange(L, device=hidden.device).unsqueeze(0)  # [1, L]
        has_hist = hist_mask.sum(dim=1) > 0  # [B]
        # Replace padding positions with -1 so argmax lands on a valid position.
        pos_or_neg = torch.where(hist_mask > 0, positions, torch.full_like(positions, -1))
        last_idx = pos_or_neg.max(dim=1).values.clamp(min=0)  # [B]
        # Gather [B, E] from [B, L, E].
        idx = last_idx.view(B, 1, 1).expand(B, 1, E)
        u = hidden.gather(1, idx).squeeze(1)  # [B, E]
        # Zero out rows with no valid history.
        u = u * has_hist.unsqueeze(-1).to(u.dtype)
        return u

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hist_ids: torch.Tensor,
        hist_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ranking logits for a batch of (history, target) pairs.

        Parameters
        ----------
        hist_ids:   [B, L] long  — padded history item indices
        hist_mask:  [B, L] float — 1 for real positions, 0 for padding
        target_ids: [B] long     — target item indices

        Returns
        -------
        logits: [B] float  (raw, no sigmoid)
        """
        B, L = hist_ids.shape

        # Truncate to max_len from the left (keep most recent).
        if L > self.max_len:
            hist_ids = hist_ids[:, -self.max_len :]
            hist_mask = hist_mask[:, -self.max_len :]
            L = self.max_len

        # Item + positional embeddings.
        pos_idx = torch.arange(L, device=hist_ids.device)  # [L]
        x = self.item_emb(hist_ids) + self.pos_emb(pos_idx)  # [B, L, E]

        # Build a combined [B*H, L, L] bool mask so we can guarantee the
        # diagonal is never masked — this prevents all-masked softmax rows
        # (which would produce NaN) for empty-history batches.
        #
        # Strategy: start with the per-batch key-padding mask, broadcast it
        # to [B, L, L] (True = key is padding), then OR with the causal mask
        # [L, L] (True = future key).  Finally, force the diagonal to False
        # so every query can always attend to at least itself.
        #
        # nn.MultiheadAttention accepts [B*n_heads, L, L] or [L, L] as
        # attn_mask; passing [B, L, L] works for batch_first=True when
        # expanded to [B*n_heads, L, L].
        causal = self._causal_mask(L, hist_ids.device)          # [L, L] bool
        pad_mask = (hist_mask == 0)                             # [B, L] bool — True = ignore key
        # Expand pad_mask to [B, L, L]: True when the KEY position is padding.
        combined = causal.unsqueeze(0) | pad_mask.unsqueeze(1)  # [B, L, L]
        # Force diagonal to always-allowed (False) — NaN-safety guarantee.
        diag_idx = torch.arange(L, device=hist_ids.device)
        combined[:, diag_idx, diag_idx] = False

        # Expand for multi-head: [B*n_heads, L, L].
        # _SASRecBlock.attn is nn.MultiheadAttention with batch_first=True.
        n_heads = self.blocks[0].attn.num_heads
        # Repeat each batch element n_heads times: [B, L, L] -> [B*H, L, L].
        combined_mh = combined.repeat_interleave(n_heads, dim=0)  # [B*H, L, L]

        for block in self.blocks:
            x = block(x, attn_mask=combined_mh, key_padding_mask=None)
        x = self.norm(x)  # [B, L, E]

        # User representation: hidden state at the last valid position.
        u = self._last_valid_repr(x, hist_mask)  # [B, E]

        # Score = <u, target_emb> + bias.
        target_emb = self.item_emb(target_ids)  # [B, E]
        logits = (u * target_emb).sum(dim=-1) + self.score_bias  # [B]
        return logits
