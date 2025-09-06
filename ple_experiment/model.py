from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, Tensor


def init_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ExpertMLP(nn.Module):
    """Single hidden-layer MLP expert returning d_model-sized output.

    Architecture: Linear(d_in->hidden) -> ReLU -> Dropout -> Linear(hidden->d_model) -> LayerNorm(d_model)
    """

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
        # selector: (B, d_sel) -> logits: (B, E) -> weights: (B, E)
        logits = self.linear(selector)
        weights = torch.softmax(logits, dim=-1)
        return weights


class PLELevel(nn.Module):
    """A single PLE extraction level with shared and task-specific experts and gates.

    All three gates (task1, task2, shared) consume the concatenation of expert outputs:
    [shared_experts, task1_experts, task2_experts]. Each gate outputs a fused vector (B, d_model).
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        expert_hidden: int,
        num_shared_experts: int,
        num_task_experts: int,
        dropout: float = 0.0,
        d_selector_t1: int | None = None,
        d_selector_t2: int | None = None,
        d_selector_shared: int | None = None,
    ) -> None:
        super().__init__()
        E_s = max(0, int(num_shared_experts))
        E_t = max(0, int(num_task_experts))

        # Experts
        self.shared_experts = nn.ModuleList(
            [ExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E_s)]
        )
        self.t1_experts = nn.ModuleList(
            [ExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E_t)]
        )
        self.t2_experts = nn.ModuleList(
            [ExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E_t)]
        )

        # Gates
        total_experts = E_s + E_t + E_t
        d_sel_t1 = d_selector_t1 if d_selector_t1 is not None else d_in
        d_sel_t2 = d_selector_t2 if d_selector_t2 is not None else d_in
        d_sel_sh = d_selector_shared if d_selector_shared is not None else d_in
        self.gate_t1 = Gate(d_sel_t1, total_experts)
        self.gate_t2 = Gate(d_sel_t2, total_experts)
        self.gate_shared = Gate(d_sel_sh, total_experts)

    def _stack_experts(self, outs: List[Tensor]) -> Tensor:
        # outs: list of (B, d_model) -> (B, E, d_model)
        return torch.stack(outs, dim=1)

    def _mix(self, weights: Tensor, expert_outs: Tensor) -> Tensor:
        # weights: (B, E), expert_outs: (B, E, d_model) -> fused: (B, d_model)
        fused = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        return fused

    def forward(
        self,
        x_for_experts: Tensor,
        sel_t1: Tensor,
        sel_t2: Tensor,
        sel_shared: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Compute expert outputs
        outs: List[Tensor] = []
        outs += [e(x_for_experts) for e in self.shared_experts]
        outs += [e(x_for_experts) for e in self.t1_experts]
        outs += [e(x_for_experts) for e in self.t2_experts]
        if len(outs) == 0:
            raise RuntimeError("PLELevel must have at least one expert")
        stacked = self._stack_experts(outs)  # (B, E, d_model)

        # Gates produce weights
        w_t1 = self.gate_t1(sel_t1)  # (B, E)
        w_t2 = self.gate_t2(sel_t2)
        w_sh = self.gate_shared(sel_shared)

        # Mix to fused outputs
        g_t1 = self._mix(w_t1, stacked)
        g_t2 = self._mix(w_t2, stacked)
        g_sh = self._mix(w_sh, stacked)
        return g_t1, g_t2, g_sh


class Tower(nn.Module):
    def __init__(self, d_model: int, out_dim: int = 1) -> None:
        super().__init__()
        hidden = max(1, d_model // 2)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)
        init_linear(self.fc1)
        init_linear(self.fc2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PLEModel(nn.Module):
    """Two-level PLE model for two binary tasks (income, never_married)."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        expert_hidden: int = 256,
        num_levels: int = 2,
        num_shared_experts: int = 2,
        num_task_experts: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_levels == 2, "This implementation targets exactly 2 levels as specified"
        self.d_in = d_in
        self.d_model = d_model
        self.num_levels = num_levels

        # Level 1: selector = x for all gates
        self.level1 = PLELevel(
            d_in=d_in,
            d_model=d_model,
            expert_hidden=expert_hidden,
            num_shared_experts=num_shared_experts,
            num_task_experts=num_task_experts,
            dropout=dropout,
            d_selector_t1=d_in,
            d_selector_t2=d_in,
            d_selector_shared=d_in,
        )

        # Level 2: selector for each gate = corresponding fused output from level1
        self.level2 = PLELevel(
            d_in=d_in,
            d_model=d_model,
            expert_hidden=expert_hidden,
            num_shared_experts=num_shared_experts,
            num_task_experts=num_task_experts,
            dropout=dropout,
            d_selector_t1=d_model,
            d_selector_t2=d_model,
            d_selector_shared=d_model,
        )

        # Task towers
        self.tower_income = Tower(d_model, out_dim=1)
        self.tower_never = Tower(d_model, out_dim=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Level 1
        g1_t1, g1_t2, g1_sh = self.level1(x_for_experts=x, sel_t1=x, sel_t2=x, sel_shared=x)
        # Level 2 selectors are g1 outputs
        g2_t1, g2_t2, g2_sh = self.level2(
            x_for_experts=x, sel_t1=g1_t1, sel_t2=g1_t2, sel_shared=g1_sh
        )
        # Towers (heads) use task fused from level 2
        logit_income = self.tower_income(g2_t1).squeeze(-1)
        logit_never = self.tower_never(g2_t2).squeeze(-1)
        return logit_income, logit_never

