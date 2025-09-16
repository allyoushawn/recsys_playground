from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def _kmeans(x: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    """L2 k-means centers; x: [B,D]."""
    B = x.shape[0]
    n_centers = min(k, B)
    # init by sampling without replacement, then repeat (with noise) if needed
    idx = torch.randperm(B, device=x.device)[:n_centers]
    centers = x[idx].clone()
    if n_centers < k:
        rem = k - n_centers
        noise = 0.05 * x.std(dim=0, keepdim=True).clamp_min(1e-6)
        add = centers[:rem] + torch.randn_like(centers[:rem]) * noise
        centers = torch.cat([centers, add], dim=0)

    for _ in range(iters):
        # assign
        dist = torch.cdist(x, centers)  # [B, k]
        assign = dist.argmin(dim=1)
        # update
        for j in range(k):
            m = (assign == j)
            if m.any():
                centers[j] = x[m].mean(dim=0)
    return centers


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
        # x: [B,D] = encoder(normalized(x_raw))
        res = x.clone()
        for l in range(self.levels):
            centers = _kmeans(res, self.codebook_size, iters)  # your current routine
            self.codebooks[l].copy_(centers)
            # quantize current residual to subtract
            dist = torch.cdist(res, centers); idx = dist.argmin(dim=1)
            q = F.embedding(idx, centers)
            res = res - q


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

    def forward_with_losses(self, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize residual and compute per-level VQ losses for training.

        Returns: (quantized, codes, commit_loss, codebook_loss) where
          - quantized: [B, D] sum of per-level quantized vectors
          - codes: [B, L] indices chosen per level
          - commit_loss: sum of ||sg[r_l] - e_c_l||² over all levels
          - codebook_loss: sum of ||r_l - sg[e_c_l]||² over all levels
        """
        B, D = residual.shape
        device = residual.device
        codes = []
        quantized_sum = torch.zeros_like(residual)
        res = residual
        commit_loss = 0.0
        codebook_loss = 0.0

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

            # Per-level VQ losses as defined in the paper
            commit_loss += F.mse_loss(res.detach(), q)  # ||sg[r_l] - e_c_l||²
            codebook_loss += F.mse_loss(res, q.detach())  # ||r_l - sg[e_c_l]||²

            quantized_sum = quantized_sum + q
            res = res - q

        codes = torch.stack(codes, dim=1)  # [B,L]
        return quantized_sum, codes, commit_loss, codebook_loss


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

@torch.no_grad()
def code_usage(model: RQVAE, data: torch.Tensor, max_batches: int = 4, bs: int = 4096):
    model.eval()
    L, K = model.cfg.levels, model.cfg.codebook_size
    counts = [torch.zeros(K, device=data.device) for _ in range(L)]
    n = min(max_batches*bs, data.shape[0])
    for i in range(0, n, bs):
        xb = data[i:i+bs]
        z = model.encoder(model.normalize(xb))
        _, codes = model.codebook(z)  # [B,L]
        for l in range(L):
            binc = torch.bincount(codes[:, l], minlength=K).float()
            counts[l][:binc.numel()] += binc
    return [c / c.sum().clamp_min(1.0) for c in counts]  # distributions


# in rqvae.py
class RQVAE(nn.Module):
    def __init__(self, cfg: RQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("x_mean", torch.zeros(1, cfg.input_dim))
        self.register_buffer("x_std", torch.ones(1, cfg.input_dim))
        
        # Improved shallower encoder with dropout (like ImprovedRQVAE)
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(128, cfg.latent_dim)
        )
        
        # Improved shallower decoder
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, cfg.input_dim)
        )
        
        self.codebook = RQCodebook(cfg.levels, cfg.codebook_size, cfg.latent_dim)
        
        # Apply improved initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Improved weight initialization like ImprovedRQVAE"""
        if isinstance(m, nn.Linear):
            # Use He/Kaiming initialization for better diversity preservation
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def normalize(self, x):
        return (x - self.x_mean) / (self.x_std + 1e-8)

    def forward(self, x):
        x_n = self.normalize(x)
        z = self.encoder(x_n)
        q, codes, commit_loss, codebook_loss = self.codebook.forward_with_losses(z)
        x_hat = self.decoder(q)
        recon = F.mse_loss(x_hat, x_n)   # reconstruct normalized space (simplest)
        loss = recon + codebook_loss + self.cfg.beta * commit_loss
        return x_hat, loss, recon, codes

@torch.no_grad()
def fit_normalizer(model: RQVAE, data: torch.Tensor):
    model.x_mean.copy_(data.mean(dim=0, keepdim=True))
    model.x_std.copy_(data.std(dim=0, keepdim=True))



def train_rqvae(
    model: RQVAE,
    data: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,  # safer default
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> RQVAE:
    model = model.to(device)
    data = data.to(device)

    # Fit normalizer once on full data
    with torch.no_grad():
        fit_normalizer(model, data)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # K-means init on encoded, normalized residuals
    with torch.no_grad():
        # draw up to batch_size samples
        ridx = torch.randperm(data.shape[0], device=device)[:min(batch_size, data.shape[0])]
        sample = data[ridx]
        sample_n = model.normalize(sample)
        encoded_sample = model.encoder(sample_n)
        model.codebook.kmeans_init(encoded_sample)

    N = data.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        total = 0.0
        for i in range(0, N, batch_size):
            xb = data[perm[i:i+batch_size]]
            _, loss, _, _ = model(xb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)

        if ep % 5 == 0 or ep == 1:
            print(f"[RQVAE] epoch {ep}/{epochs} loss={total/N:.6f}")
            dists = code_usage(model, data)
            perplexities = [torch.exp(-(d * (d.clamp_min(1e-12).log())).sum()).item() for d in dists]
            print(f"   usage perplexity per level: {perplexities}")
    return model



@torch.no_grad()
def encode_codes(model: RQVAE, data: torch.Tensor, device: str | None = None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device); data = data.to(device)
    z = model.encoder(model.normalize(data))
    _, codes = model.codebook(z)
    return codes.cpu()


