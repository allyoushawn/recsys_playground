from __future__ import annotations

from typing import Tuple

import torch
from torch import nn, Tensor


def init_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ExpertMLP(nn.Module):
    """Single-hidden-layer MLP expert returning d_model-sized output."""

    def __init__(self, d_in: int, hidden: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, d_model)
        self.ln = nn.LayerNorm(d_model)
        init_linear(self.fc1)
        init_linear(self.fc2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ln(x)
        return x


class Gate(nn.Module):
    """Linear gate mapping selector to logits over experts; outputs softmax weights."""

    def __init__(self, d_selector: int, num_experts: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_selector, num_experts)
        init_linear(self.linear)

    def forward(self, selector: Tensor) -> Tensor:
        logits = self.linear(selector)
        weights = torch.softmax(logits, dim=-1)
        return weights


class MMoELayer(nn.Module):
    """One MMoE layer with shared experts and per-task gates.

    Experts are shared across tasks. Each task has a gate that mixes expert outputs
    using a task-specific selector.
    """

    def __init__(
        self,
        d_expert_in: int,
        d_model: int,
        expert_hidden: int,
        num_experts: int,
        d_selector_t1: int,
        d_selector_t2: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [ExpertMLP(d_expert_in, expert_hidden, d_model, dropout) for _ in range(num_experts)]
        )
        self.gate_t1 = Gate(d_selector_t1, num_experts)
        self.gate_t2 = Gate(d_selector_t2, num_experts)

    def forward(self, x_for_experts: Tensor, sel_t1: Tensor, sel_t2: Tensor) -> Tuple[Tensor, Tensor]:
        # Expert outputs: (B, d_model) each
        outs = [e(x_for_experts) for e in self.experts]
        if len(outs) == 0:
            raise RuntimeError("MMoELayer must have at least one expert")
        stacked = torch.stack(outs, dim=1)  # (B, E, d_model)

        # Gates produce weights (B, E)
        w_t1 = self.gate_t1(sel_t1)
        w_t2 = self.gate_t2(sel_t2)

        # Mix experts with gate weights -> (B, d_model)
        g_t1 = (w_t1.unsqueeze(-1) * stacked).sum(dim=1)
        g_t2 = (w_t2.unsqueeze(-1) * stacked).sum(dim=1)
        return g_t1, g_t2


class Tower(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = max(1, d_model // 2)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        init_linear(self.fc1)
        init_linear(self.fc2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MMoEModel(nn.Module):
    """Multi-gate Mixture-of-Experts model with optional stacking (ML-MMoE).

    - num_levels=1: classic single-level MMoE (experts receive input x; gates select with x)
    - num_levels>1: stacked MMoE; experts remain shared per level. Gates at level>1 use the
      previous level's task-fused output as the selector. Experts by default still consume x.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        expert_hidden: int = 256,
        num_experts: int = 4,
        num_levels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_levels >= 1
        self.d_in = d_in
        self.d_model = d_model
        self.num_levels = num_levels

        layers = []
        for level in range(num_levels):
            if level == 0:
                d_sel = d_in
            else:
                d_sel = d_model
            # Experts consume x for all levels (keeps experts shared to the base feature space)
            layers.append(
                MMoELayer(
                    d_expert_in=d_in,
                    d_model=d_model,
                    expert_hidden=expert_hidden,
                    num_experts=num_experts,
                    d_selector_t1=d_sel,
                    d_selector_t2=d_sel,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

        self.tower_income = Tower(d_model)
        self.tower_never = Tower(d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        sel_t1 = x
        sel_t2 = x
        g_t1 = None
        g_t2 = None
        for i, layer in enumerate(self.layers):
            g_t1, g_t2 = layer(x_for_experts=x, sel_t1=sel_t1, sel_t2=sel_t2)
            # Next-level selectors are previous fused outputs
            sel_t1, sel_t2 = g_t1, g_t2

        assert g_t1 is not None and g_t2 is not None
        logit_income = self.tower_income(g_t1).squeeze(-1)
        logit_never = self.tower_never(g_t2).squeeze(-1)
        return logit_income, logit_never

