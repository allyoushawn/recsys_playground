"""OneTrans: One Transformer for All Features (CTR ranking adaptation).

Implements the core mechanisms from the OneTrans paper:

- **Auto-Split tokenizer**: the target item embedding is projected into
  ``n_ns_tokens`` learned NS (non-sequential) feature tokens instead of being
  pooled into a single vector.  This preserves item-side feature richness in the
  same sequence that carries history.

- **S-token sequence**: history item embeddings are kept UNCOMPRESSED as a
  sequence S of length L_S with learned positional encodings (chronological
  ordering is the timestamp-aware proxy).  Mean-pooling before the Transformer
  would lose the sequential signal — keeping the sequence is OneTrans's key
  distinction vs pool-then-cross baselines.

- **Mixed parameterization**: S-tokens share one set of (W_q, W_k, W_v) + FFN
  weights; each NS-token j gets its OWN (W_q^j, W_k^j, W_v^j) and FFN.  This
  lets NS-tokens learn dedicated feature-interaction patterns while S-tokens
  share weights for efficiency.  Toggle: ``use_mixed_param`` (default True).

- **Causal block mask**: S-token i attends to S-tokens <= i (causal within S);
  NS-token j attends to ALL valid S-tokens + NS-tokens <= j.  Padded S positions
  are never attended (key-padding mask from ``hist_mask``).

- **Pyramid schedule**: at layers > 0, only the last ``pyramid_keep`` S-tokens
  ISSUE queries (older ones carry residual but don't recompute attention).
  Toggle: ``use_pyramid`` (default False).

- **RMSNorm** instead of LayerNorm (pre-norm around attention + FFN blocks).

- **CTR head**: mean-pool the final NS hidden states -> MLP -> scalar logit.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from amazon_ranking.src.baselines import _SeqRankModel
from amazon_ranking.src.din import _mlp


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019).

    A simpler alternative to LayerNorm that omits the mean-centering step.
    Pre-norm use with residuals maintains training stability without the extra
    cost of computing the mean.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# Per-token projection for NS tokens (mixed parameterization)
# ---------------------------------------------------------------------------

class _PerTokenProj(nn.Module):
    """n_tokens independent linear projections (in_dim -> out_dim).

    Implements as a grouped linear via weight tensor [n_tokens, in_dim, out_dim].
    Avoids a Python loop at forward time.

    This implements the per-NS-token W_q^j / W_k^j / W_v^j projections from the
    OneTrans mixed parameterization design.
    """

    def __init__(self, n_tokens: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(n_tokens, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_tokens, out_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_tokens, in_dim]
        returns: [B, n_tokens, out_dim]
        """
        return torch.einsum("bti,tio->bto", x, self.weight) + self.bias


# ---------------------------------------------------------------------------
# OneTrans transformer block
# ---------------------------------------------------------------------------

class _OneTrans_Block(nn.Module):
    """One OneTrans block with mixed parameterization + causal block attention.

    Mixed parameterization (``use_mixed_param=True``):
    - S-tokens: shared (W_q, W_k, W_v) and shared FFN (paper: all S-tokens are
      homogeneous sequential features sharing one set of parameters).
    - NS-tokens j=0..n_ns-1: each has its own (W_q^j, W_k^j, W_v^j) and its
      own FFN (paper: different non-sequential features may have different
      interaction patterns).

    Simplified unified (``use_mixed_param=False``):
    - All tokens reuse the S-shared weights (ablation).

    The block accepts the full sequence X = [S ; NS] of length L = L_S + n_ns
    and the pre-built additive attention bias + per-batch key-padding mask.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_ns_tokens: int,
        ff_dim: int,
        dropout: float,
        use_mixed_param: bool,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_ns = n_ns_tokens
        self.use_mixed_param = use_mixed_param
        self.scale = math.sqrt(self.head_dim)

        # Pre-norms for S block (attention and FFN).
        self.norm1_s = RMSNorm(embed_dim)
        self.norm2_s = RMSNorm(embed_dim)

        # Shared S-token QKV projections (also fallback when use_mixed_param=False).
        self.Wq_s = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk_s = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_s = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)

        # Shared FFN for S-tokens.
        self.ff_s = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        if use_mixed_param:
            # Per-NS-token QKV (Auto-Split / mixed param: each NS feature token
            # has its own projection weights).
            self.Wq_ns = _PerTokenProj(n_ns_tokens, embed_dim, embed_dim)
            self.Wk_ns = _PerTokenProj(n_ns_tokens, embed_dim, embed_dim)
            self.Wv_ns = _PerTokenProj(n_ns_tokens, embed_dim, embed_dim)
            # Per-NS-token FFN.
            self.ff_ns: nn.Module = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, embed_dim),
                )
                for _ in range(n_ns_tokens)
            ])
            # Per-NS-token pre-norms.
            self.norm1_ns: nn.Module = nn.ModuleList(
                [RMSNorm(embed_dim) for _ in range(n_ns_tokens)]
            )
            self.norm2_ns: nn.Module = nn.ModuleList(
                [RMSNorm(embed_dim) for _ in range(n_ns_tokens)]
            )

        self.drop = nn.Dropout(dropout)

    def _project_qkv(
        self,
        x: torch.Tensor,
        L_S: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build Q, K, V for the full sequence X = [S ; NS].

        Returns Q, K, V each [B, L, E].
        """
        B, L, E = x.shape

        x_s = x[:, :L_S, :]         # [B, L_S, E]
        x_ns = x[:, L_S:, :]        # [B, n_ns, E]

        normed_s = self.norm1_s(x_s)  # [B, L_S, E]

        # Shared projections for S block.
        Q_s = self.Wq_s(normed_s)    # [B, L_S, E]
        K_s = self.Wk_s(normed_s)    # [B, L_S, E]
        V_s = self.Wv_s(normed_s)    # [B, L_S, E]

        if self.use_mixed_param:
            # Per-NS-token norms.
            norm1_ns_list: nn.ModuleList = self.norm1_ns  # type: ignore[assignment]
            normed_ns_parts = [norm1_ns_list[j](x_ns[:, j:j+1, :]) for j in range(self.n_ns)]
            normed_ns = torch.cat(normed_ns_parts, dim=1)  # [B, n_ns, E]
            Q_ns = self.Wq_ns(normed_ns)
            K_ns = self.Wk_ns(normed_ns)
            V_ns = self.Wv_ns(normed_ns)
        else:
            normed_ns = self.norm1_s(x_ns)
            Q_ns = self.Wq_s(normed_ns)
            K_ns = self.Wk_s(normed_ns)
            V_ns = self.Wv_s(normed_ns)

        Q = torch.cat([Q_s, Q_ns], dim=1)  # [B, L, E]
        K = torch.cat([K_s, K_ns], dim=1)  # [B, L, E]
        V = torch.cat([V_s, V_ns], dim=1)  # [B, L, E]
        return Q, K, V

    def _apply_ffn(
        self,
        x: torch.Tensor,
        L_S: int,
    ) -> torch.Tensor:
        """Apply FFN sub-layer (pre-norm + residual) to the full sequence."""
        x_s = x[:, :L_S, :]   # [B, L_S, E]
        x_ns = x[:, L_S:, :]  # [B, n_ns, E]

        # S FFN: shared norm + shared ff_s.
        s_out = self.drop(self.ff_s(self.norm2_s(x_s))) + x_s

        if self.use_mixed_param:
            norm2_ns_list: nn.ModuleList = self.norm2_ns  # type: ignore[assignment]
            ff_ns_list: nn.ModuleList = self.ff_ns        # type: ignore[assignment]
            ns_chunks = []
            for j in range(self.n_ns):
                tok = x_ns[:, j:j+1, :]
                ns_chunks.append(self.drop(ff_ns_list[j](norm2_ns_list[j](tok))) + tok)
            ns_out = torch.cat(ns_chunks, dim=1)
        else:
            ns_out = self.drop(self.ff_s(self.norm2_s(x_ns))) + x_ns

        return torch.cat([s_out, ns_out], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        L_S: int,
        attn_bias: torch.Tensor,
        key_pad_mask: Optional[torch.Tensor],
        pyramid_active_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:              [B, L, E]  full sequence [S ; NS]
        L_S:            number of S-tokens
        attn_bias:      [L, L] additive -inf mask (causal block structure)
        key_pad_mask:   [B, L] bool, True = this position is a padding KEY
        pyramid_active_s: [L_S] bool; if not None, only True S-rows issue new attention
                           (pyramid schedule — older S-tokens carry residual only)
        """
        B, L, E = x.shape
        H = self.n_heads
        D = self.head_dim

        Q, K, V = self._project_qkv(x, L_S)

        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, H, D).transpose(1, 2)  # [B, H, L, D]

        Q_ = _split_heads(Q)  # [B, H, L, D]
        K_ = _split_heads(K)
        V_ = _split_heads(V)

        # Scaled dot-product: [B, H, L, L].
        scores = torch.matmul(Q_, K_.transpose(-2, -1)) / self.scale

        # Add causal block structural bias [1, 1, L, L].
        scores = scores + attn_bias.unsqueeze(0).unsqueeze(0)

        # Key-padding: mask positions that are padding S-tokens as keys.
        if key_pad_mask is not None:
            # key_pad_mask [B, L] bool -> [B, 1, 1, L] for broadcast.
            scores = scores.masked_fill(
                key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Guard all-inf rows to prevent NaN in softmax.
        # An NS-token whose query row is entirely -inf (impossible by construction
        # since NS attends to itself and all S, but we guard defensively anyway).
        row_max = scores.amax(dim=-1, keepdim=True)       # [B, H, L, 1]
        all_neg_inf = (row_max == float("-inf"))
        # Replace -inf entries in fully-masked rows with 0 before softmax so that
        # exp(0)=1 -> uniform softmax weights (then zeroed out below).
        scores = scores.masked_fill(
            all_neg_inf.expand_as(scores) & scores.isinf(), 0.0
        )

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, L, L]
        # Zero weights for rows that were all-inf so their output is 0.
        attn_weights = attn_weights.masked_fill(
            all_neg_inf.expand_as(attn_weights), 0.0
        )

        attn_out_ = torch.matmul(attn_weights, V_)          # [B, H, L, D]
        attn_out = attn_out_.transpose(1, 2).contiguous().view(B, L, E)
        attn_out = self.drop(self.Wo(attn_out))

        # Attention residual.
        # Pyramid schedule: for inactive S-positions, zero the attention delta
        # so they carry their residual through unchanged (no update from attention).
        if pyramid_active_s is not None:
            # active [L_S] bool -> [1, L, 1] gate for broadcast over B.
            active_f = pyramid_active_s.to(dtype=attn_out.dtype, device=attn_out.device)
            # Build a gate of shape [1, L, 1]: 1.0 for active positions, 0.0 for inactive.
            gate = torch.ones(1, L, 1, device=attn_out.device, dtype=attn_out.dtype)
            gate[0, :L_S, 0] = active_f
            attn_out = attn_out * gate

        x = x + attn_out

        # FFN sub-layer — but inactive S rows must also skip the FFN update
        # (true pyramid freeze: they carry residual only, no attention OR FFN delta).
        if pyramid_active_s is not None:
            # Save the frozen S rows before FFN so we can restore them.
            inactive_mask = ~pyramid_active_s.to(device=x.device)  # [L_S] bool
            x_s_frozen = x[:, :L_S, :].clone()    # [B, L_S, E]
            x = self._apply_ffn(x, L_S)
            # Restore inactive S positions to their pre-FFN values.
            if inactive_mask.any():
                x[:, :L_S, :] = torch.where(
                    inactive_mask.unsqueeze(0).unsqueeze(-1),  # [1, L_S, 1]
                    x_s_frozen,
                    x[:, :L_S, :],
                )
        else:
            x = self._apply_ffn(x, L_S)

        return x


# ---------------------------------------------------------------------------
# OneTrans model
# ---------------------------------------------------------------------------

class OneTrans(_SeqRankModel):
    """OneTrans: sequence-to-CTR model with Auto-Split tokenizer + mixed parameterization.

    Key mechanisms (see module docstring for paper citations):

    1. **S-tokens** — history items as an uncompressed sequence with learned
       positional embeddings (chronological ordering = timestamp proxy).
    2. **NS-tokens** (Auto-Split tokenizer) — target item projected into
       ``n_ns_tokens`` feature tokens via a learned linear split.
    3. **Mixed parameterization** — S-tokens share QKV + FFN; each NS-token has
       its own QKV + FFN (``use_mixed_param=True``).
    4. **Causal block mask** — S-i attends S-j<=i; NS-j attends all valid S + NS-k<=j.
    5. **Pyramid schedule** — layers > 0 prune older S queries (``use_pyramid=True``).
    6. **RMSNorm** pre-norm.
    7. **CTR head** — mean-pool final NS hidden states -> MLP(d, (80,40), 1) -> logit.

    Parameters
    ----------
    num_items:
        Vocabulary size; pad index = num_items (inherited from _SeqRankModel).
    embed_dim:
        Embedding / hidden dimension.  Must be divisible by n_heads.
    n_layers:
        Number of OneTrans blocks.
    n_heads:
        Multi-head attention heads.
    n_ns_tokens:
        Number of NS (Auto-Split) feature tokens for the target item.
    ff_dim:
        FFN inner dimension; defaults to 4 * embed_dim.
    dropout:
        Dropout probability.
    max_len:
        Maximum supported history length; positional embedding table size.
    use_mixed_param:
        Enable per-NS-token QKV + FFN (True = paper's mixed parameterization).
    use_pyramid:
        Enable pyramid query pruning at Transformer layers > 0.
    pyramid_keep:
        Number of latest S-positions that still issue queries under pyramid.
        When None, defaults to max(1, L_S // 2) computed at forward time.
    item_category:
        Optional LongTensor of shape ``[num_items]`` mapping item idx -> category
        id.  When provided, a category embedding is appended as an extra NS-token
        (total NS count becomes ``n_ns_tokens + 1``).  Registered as a buffer so
        it moves with the model across devices.
    num_categories:
        Required when ``item_category`` is provided; size of the category
        vocabulary (number of distinct category ids).
    **kwargs:
        Extra kwargs silently ignored (registry / build_model compatibility).
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 2,
        n_ns_tokens: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
        max_len: int = 50,
        use_mixed_param: bool = True,
        use_pyramid: bool = False,
        pyramid_keep: Optional[int] = None,
        item_category: Optional[torch.Tensor] = None,
        num_categories: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(num_items, embed_dim)
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads}); "
                f"got embed_dim={embed_dim}, n_heads={n_heads}"
            )
        if item_category is not None and num_categories is None:
            raise ValueError(
                "num_categories is required when item_category is provided"
            )
        ff_dim = ff_dim if ff_dim is not None else 4 * embed_dim
        self.max_len = int(max_len)
        self.n_ns = int(n_ns_tokens)
        self.n_layers = int(n_layers)
        self.use_pyramid = bool(use_pyramid)
        self.pyramid_keep = pyramid_keep  # None -> computed dynamically at forward

        # Category NS-token: heterogeneous non-sequential feature ablation.
        # When present, n_ns_tokens_total = n_ns_tokens + 1 for the transformer blocks.
        self._has_category = item_category is not None
        if self._has_category:
            # Validate the category map (v2 cli-review hardening): it is indexed by
            # item id in forward, so a wrong shape/dtype/range would index-error or
            # silently mis-map at run time.
            if item_category.shape != (num_items,):
                raise ValueError(
                    f"item_category must have shape ({num_items},), got "
                    f"{tuple(item_category.shape)}"
                )
            item_category = item_category.long()
            if int(item_category.min()) < 0 or int(item_category.max()) >= int(num_categories):
                raise ValueError(
                    "item_category ids must lie in [0, num_categories)"
                )
            self.register_buffer("item_category", item_category)
            self.category_emb = nn.Embedding(int(num_categories), embed_dim)
        n_ns_total = self.n_ns + (1 if self._has_category else 0)

        # Positional embedding for S-tokens (history sequence, chronological proxy).
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        # Auto-Split tokenizer: single target embedding d -> n_ns_tokens * d, then reshape.
        self.ns_split = nn.Linear(embed_dim, n_ns_tokens * embed_dim)

        # OneTrans blocks — use n_ns_total so per-NS-token params account for the
        # optional category token.
        self.blocks = nn.ModuleList([
            _OneTrans_Block(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_ns_tokens=n_ns_total,
                ff_dim=ff_dim,
                dropout=dropout,
                use_mixed_param=use_mixed_param,
            )
            for _ in range(n_layers)
        ])

        # Final RMSNorm over the full sequence.
        self.final_norm = RMSNorm(embed_dim)

        # CTR head: mean-pool NS token states -> MLP -> scalar logit.
        self.head = _mlp(embed_dim, (80, 40), 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_causal_block_mask(
        self, L_S: int, L_NS: int, device: torch.device
    ) -> torch.Tensor:
        """Build additive attention structural bias [L, L].

        Causal block mask rules (0-indexed positions):
        - S-position i (0 <= i < L_S): may attend to S-positions j <= i only
          (causal within S; NS positions are never keys from S rows).
        - NS-position n = L_S + j: may attend to ALL S-positions (0..L_S-1)
          AND NS-positions L_S..L_S+j (causal within NS).

        Padded S-positions are blocked separately via the key_padding_mask;
        structurally they appear allowed here (mask handles them).

        Returns [L, L] float with 0.0 (allowed) or -inf (forbidden).
        """
        L = L_S + L_NS
        bias = torch.full((L, L), float("-inf"), device=device)

        # S-token rows: causal within S only.
        for i in range(L_S):
            bias[i, :i + 1] = 0.0

        # NS-token rows: all S positions + causal within NS.
        for j in range(L_NS):
            row = L_S + j
            bias[row, :L_S] = 0.0            # all S keys allowed
            bias[row, L_S:L_S + j + 1] = 0.0  # NS[0..j] allowed

        return bias

    def _pyramid_active(self, L_S: int, layer_idx: int) -> Optional[torch.Tensor]:
        """Return [L_S] bool mask of S-positions that issue queries at this layer.

        Layer 0: all positions active (returns None -> skip masking for efficiency).
        Layers >= 1 with use_pyramid=True: only the last ``pyramid_keep`` positions.

        ``pyramid_keep`` defaults to max(1, L_S // 2) when not set at init.
        """
        if not self.use_pyramid or layer_idx == 0:
            return None
        keep = self.pyramid_keep if self.pyramid_keep is not None else max(1, L_S // 2)
        keep = min(int(keep), L_S)
        active = torch.zeros(L_S, dtype=torch.bool)
        if keep > 0:
            active[-keep:] = True
        return active

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
        hist_ids:   [B, L] long  -- padded history item indices (pad_idx = num_items)
        hist_mask:  [B, L] float -- 1.0 for real positions, 0.0 for padding
        target_ids: [B] long     -- target item indices

        Returns
        -------
        logits: [B] float (raw scalar logit, no sigmoid)
        """
        # Truncate to max_len (keep most recent items).
        if hist_ids.shape[1] > self.max_len:
            hist_ids = hist_ids[:, -self.max_len:]
            hist_mask = hist_mask[:, -self.max_len:]

        B, L_S = hist_ids.shape
        device = hist_ids.device

        # --- S-tokens: item embeddings + positional embeddings ---
        # Right-aligned positional encoding: most recent item -> position max_len-1,
        # second-most-recent -> max_len-2, ..., so the positional encoding is
        # invariant to left-padding.
        cum_valid = hist_mask.cumsum(dim=1)               # [B, L_S]
        num_valid = hist_mask.sum(dim=1, keepdim=True)    # [B, 1]
        rank_from_right = num_valid - cum_valid            # [B, L_S]; 0 = most recent valid
        pos_idx = (self.max_len - 1 - rank_from_right).long().clamp(min=0)  # [B, L_S]
        # Zero out positions for padded slots (all get pos 0 as a dummy).
        pos_idx = pos_idx * hist_mask.long()

        hist_emb = self.item_emb(hist_ids)  # [B, L_S, E]
        pos = self.pos_emb(pos_idx)          # [B, L_S, E]
        S = hist_emb + pos                   # [B, L_S, E]

        # --- NS-tokens: Auto-Split target item projection ---
        # Project single target embedding d -> n_ns * d, then reshape.
        tgt_emb = self.item_emb(target_ids)             # [B, E]
        ns_flat = self.ns_split(tgt_emb)               # [B, n_ns * E]
        NS = ns_flat.view(B, self.n_ns, self.embed_dim)  # [B, n_ns, E]

        # Category NS-token: append category embedding as one extra NS-token so the
        # transformer sees a heterogeneous non-sequential feature (item_category ablation).
        if self._has_category:
            cat_ids = self.item_category[target_ids]         # [B]
            cat_tok = self.category_emb(cat_ids).unsqueeze(1)  # [B, 1, E]
            NS = torch.cat([NS, cat_tok], dim=1)             # [B, n_ns+1, E]

        n_ns_total = NS.shape[1]  # n_ns or n_ns+1

        # Concatenate [S ; NS] -> [B, L_S + n_ns_total, E].
        x = torch.cat([S, NS], dim=1)

        # --- Structural causal block attention bias (shared across the batch) ---
        attn_bias = self._build_causal_block_mask(L_S, n_ns_total, device)  # [L, L]

        # --- Key-padding mask: padded S-positions are never keys ---
        # [B, L_S] bool; True = this S-position is padding.
        s_pad = (hist_mask == 0)
        # NS-positions are always valid as keys.
        ns_valid = torch.zeros(B, n_ns_total, dtype=torch.bool, device=device)
        key_pad_mask = torch.cat([s_pad, ns_valid], dim=1)  # [B, L_S + n_ns_total]

        # --- Transformer blocks ---
        for layer_idx, block in enumerate(self.blocks):
            pyr = self._pyramid_active(L_S, layer_idx)
            x = block(
                x=x,
                L_S=L_S,
                attn_bias=attn_bias,
                key_pad_mask=key_pad_mask,
                pyramid_active_s=pyr,
            )

        x = self.final_norm(x)  # [B, L, E]

        # --- CTR head: mean-pool NS hidden states -> MLP -> scalar logit ---
        ns_out = x[:, L_S:, :]         # [B, n_ns_total, E]
        ns_pooled = ns_out.mean(dim=1)  # [B, E]
        return self.head(ns_pooled).squeeze(-1)  # [B]
