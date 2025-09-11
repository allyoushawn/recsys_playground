from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RQCodebook(nn.Module):
    def __init__(self, levels: int, codebook_size: int, dim: int):
        super().__init__()
        self.levels = levels
        self.codebook_size = codebook_size
        self.dim = dim
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook_size, dim) * 0.1) for _ in range(levels)]
        )

    @torch.no_grad()
    def kmeans_init(self, x: torch.Tensor, iters: int = 10) -> None:
        """K-means init per level on a sample batch x: [B, D]."""
        B = x.shape[0]
        for l in range(self.levels):
            # random sample as initial centers
            idx = torch.randperm(B, device=x.device)[: self.codebook_size]
            centers = x[idx].clone()
            for _ in range(iters):
                dist = torch.cdist(x, centers)  # [B, K]
                assign = dist.argmin(dim=1)
                for k in range(self.codebook_size):
                    mask = assign == k
                    if mask.any():
                        centers[k] = x[mask].mean(dim=0)
            self.codebooks[l].copy_(centers)

    def forward(self, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize residual w.r.t. nearest codeword at each level sequentially.

        Returns: (quantized, codes) where
          - quantized: [B, D] sum of per-level quantized vectors
          - codes: [B, L] indices chosen per level
        """
        B, D = residual.shape
        device = residual.device
        codes = []
        quantized_sum = torch.zeros_like(residual)
        res = residual
        for l in range(self.levels):
            emb = self.codebooks[l]  # [K, D]
            # find nearest neighbor
            # dist(x, e)^2 = |x|^2 + |e|^2 - 2 x.e
            x2 = (res**2).sum(dim=1, keepdim=True)  # [B,1]
            e2 = (emb**2).sum(dim=1)  # [K]
            scores = x2 + e2 - 2 * res @ emb.t()  # [B,K]
            idx = scores.argmin(dim=1)
            codes.append(idx)
            q = F.embedding(idx, emb)
            quantized_sum = quantized_sum + q
            res = res - q
        codes = torch.stack(codes, dim=1)  # [B,L]
        return quantized_sum, codes


class MLP(nn.Module):
    def __init__(self, dims: List[int], activation=nn.ReLU, out_activation=None):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        if out_activation is not None:
            layers.append(out_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RQVAEConfig:
    input_dim: int = 768
    latent_dim: int = 32
    levels: int = 3
    codebook_size: int = 256
    beta: float = 0.25


class RQVAE(nn.Module):
    def __init__(self, cfg: RQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MLP([cfg.input_dim, 512, 256, 128, cfg.latent_dim])
        self.decoder = MLP([cfg.latent_dim, 128, 256, 512, cfg.input_dim])
        self.codebook = RQCodebook(cfg.levels, cfg.codebook_size, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        q, codes = self.codebook(z)
        x_hat = self.decoder(q)
        # Losses
        recon = F.mse_loss(x_hat, x)
        # VQ losses: commit + codebook (stop-grad on one side)
        commit = F.mse_loss(z.detach(), q)
        code = F.mse_loss(z, q.detach())
        loss = recon + self.cfg.beta * (commit + code)
        return x_hat, loss, recon, codes


def train_rqvae(
    model: RQVAE,
    data: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 4e-1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> RQVAE:
    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    # K-means init on a sample batch
    with torch.no_grad():
        sample = data[torch.randperm(data.shape[0])[: min(batch_size, data.shape[0])]].to(device)
        model.codebook.kmeans_init(sample)
    N = data.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        total = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            xb = data[idx]
            x_hat, loss, recon, _ = model(xb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)
        if ep % 5 == 0 or ep == 1:
            print(f"[RQVAE] epoch {ep}/{epochs} loss={total/N:.4f}")
    return model


@torch.no_grad()
def encode_codes(model: RQVAE, data: torch.Tensor, device: str | None = None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    z = model.encoder(data)
    _, codes = model.codebook(z)
    return codes.cpu()

