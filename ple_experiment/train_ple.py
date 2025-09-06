from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn, optim, Tensor
# AMP: prefer torch.amp with device_type, fallback to torch.cuda.amp
try:
    from torch import amp as _amp  # PyTorch >= 2.0 recommended API
    _USE_NEW_AMP = True
except Exception:  # pragma: no cover
    from torch.cuda import amp as _amp  # type: ignore
    _USE_NEW_AMP = False
from torch.utils.data import DataLoader

# Ensure local imports work when running via `python ple_experiment/train_ple.py`
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in os.sys.path:
    os.sys.path.insert(0, SCRIPT_DIR)

from dataset import make_dataloaders  # noqa: E402
from metrics import bin_metrics  # noqa: E402
from model import PLEModel  # noqa: E402


@dataclass
class Config:
    data_dir: str
    out_dir: str = "./runs/ple_census"
    epochs: int = 15
    batch_size: int = 4096
    num_workers: int = 4
    lr: float = 2e-3
    weight_decay: float = 1e-4
    d_model: int = 128
    expert_hidden: int = 256
    num_levels: int = 2
    num_shared_experts: int = 2
    num_task_experts: int = 2
    dropout: float = 0.1
    w_income: float = 1.0
    w_never_married: float = 1.0
    use_pos_weight: bool = True
    grad_clip: float = 1.0
    mixed_precision: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    d_in: int | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train PLE on Census-Income (KDD)")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./runs/ple_census")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--expert_hidden", type=int, default=256)
    p.add_argument("--num_levels", type=int, default=2)
    p.add_argument("--num_shared_experts", type=int, default=2)
    p.add_argument("--num_task_experts", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--w_income", type=float, default=1.0)
    p.add_argument("--w_never_married", type=float, default=1.0)
    p.add_argument("--use_pos_weight", type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y"}, default=True)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y"}, default=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        expert_hidden=args.expert_hidden,
        num_levels=args.num_levels,
        num_shared_experts=args.num_shared_experts,
        num_task_experts=args.num_task_experts,
        dropout=args.dropout,
        w_income=args.w_income,
        w_never_married=args.w_never_married,
        use_pos_weight=args.use_pos_weight,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
    )
    return cfg


def compute_pos_weight(loader: DataLoader) -> Tuple[float, float]:
    # pos_weight = neg/pos for BCEWithLogitsLoss
    inc_pos = 0
    inc_neg = 0
    nm_pos = 0
    nm_neg = 0
    for batch in loader:
        yi = batch["y_income"].view(-1).to(torch.int)
        yn = batch["y_never_married"].view(-1).to(torch.int)
        inc_pos += int((yi == 1).sum())
        inc_neg += int((yi == 0).sum())
        nm_pos += int((yn == 1).sum())
        nm_neg += int((yn == 0).sum())
    inc_pw = float(inc_neg / max(1, inc_pos))
    nm_pw = float(nm_neg / max(1, nm_pos))
    return inc_pw, nm_pw


def evaluate(model: nn.Module, loader: DataLoader, device: str, pos_weights: Tuple[float | None, float | None]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    inc_metrics = {"loss": 0.0, "acc": 0.0, "auc": 0.0}
    nm_metrics = {"loss": 0.0, "acc": 0.0, "auc": 0.0}
    inc_vals = []
    nm_vals = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            yi = batch["y_income"].to(device, non_blocking=True).float()
            yn = batch["y_never_married"].to(device, non_blocking=True).float()

            logit_i, logit_n = model(x)
            bm_i = bin_metrics(logit_i, yi, pos_weight=pos_weights[0])
            bm_n = bin_metrics(logit_n, yn, pos_weight=pos_weights[1])

            bs = x.size(0)
            n_samples += bs
            inc_vals.append((bm_i["loss"], bm_i["acc"], bm_i["auc"]))
            nm_vals.append((bm_n["loss"], bm_n["acc"], bm_n["auc"]))

    # Average metrics (loss/acc), and compute mean AUC ignoring NaNs
    def agg(vals):
        if not vals:
            return {"loss": 0.0, "acc": 0.0, "auc": float("nan")}
        loss = float(np.mean([v[0] for v in vals]))
        acc = float(np.mean([v[1] for v in vals]))
        aucs = [v[2] for v in vals if not np.isnan(v[2])]
        auc = float(np.mean(aucs)) if aucs else float("nan")
        return {"loss": loss, "acc": acc, "auc": auc}

    inc = agg(inc_vals)
    nm = agg(nm_vals)
    total_loss = inc["loss"] + nm["loss"]
    aucs = [v for v in [inc["auc"], nm["auc"]] if not np.isnan(v)]
    combined_auc = float(np.mean(aucs)) if aucs else float("nan")
    return {
        "loss": total_loss,
        "inc_loss": inc["loss"],
        "inc_acc": inc["acc"],
        "inc_auc": inc["auc"],
        "nm_loss": nm["loss"],
        "nm_acc": nm["acc"],
        "nm_auc": nm["auc"],
        "combined_auc": float(combined_auc),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
    scaler,
    w_income: float,
    w_nm: float,
    pos_weights: Tuple[float | None, float | None],
    grad_clip: float,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    inc_vals = []
    nm_vals = []

    device_type = "cuda" if device == "cuda" else "cpu"
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        yi = batch["y_income"].to(device, non_blocking=True).float()
        yn = batch["y_never_married"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        # autocast context with device_type when available
        if _USE_NEW_AMP:
            ac = _amp.autocast(device_type=device_type, enabled=mixed_precision)
        else:
            ac = _amp.autocast(enabled=mixed_precision)
        with ac:
            logit_i, logit_n = model(x)
            # Build BCE losses (per-task, with optional pos_weight)
            if pos_weights[0] is not None:
                loss_i = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([float(pos_weights[0])], device=device)
                )(logit_i, yi)
            else:
                loss_i = nn.BCEWithLogitsLoss()(logit_i, yi)
            if pos_weights[1] is not None:
                loss_n = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([float(pos_weights[1])], device=device)
                )(logit_n, yn)
            else:
                loss_n = nn.BCEWithLogitsLoss()(logit_n, yn)

            loss = w_income * loss_i + w_nm * loss_n

        if scaler is not None and mixed_precision:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Metrics (detach tensors)
        bm_i = bin_metrics(logit_i.detach(), yi.detach(), pos_weight=pos_weights[0])
        bm_n = bin_metrics(logit_n.detach(), yn.detach(), pos_weight=pos_weights[1])
        inc_vals.append((bm_i["loss"], bm_i["acc"], bm_i["auc"]))
        nm_vals.append((bm_n["loss"], bm_n["acc"], bm_n["auc"]))
        total += float(loss.item())

    # Aggregate
    def agg(vals):
        if not vals:
            return {"loss": 0.0, "acc": 0.0, "auc": float("nan")}
        loss = float(np.mean([v[0] for v in vals]))
        acc = float(np.mean([v[1] for v in vals]))
        aucs = [v[2] for v in vals if not np.isnan(v[2])]
        auc = float(np.mean(aucs)) if aucs else float("nan")
        return {"loss": loss, "acc": acc, "auc": auc}

    inc = agg(inc_vals)
    nm = agg(nm_vals)
    aucs = [v for v in [inc["auc"], nm["auc"]] if not np.isnan(v)]
    combined_auc = float(np.mean(aucs)) if aucs else float("nan")
    return {
        "loss": float(total / max(1, len(loader))),
        "inc_loss": inc["loss"],
        "inc_acc": inc["acc"],
        "inc_auc": inc["auc"],
        "nm_loss": nm["loss"],
        "nm_acc": nm["acc"],
        "nm_auc": nm["auc"],
        "combined_auc": float(combined_auc),
    }


def save_jsonl(path: str, obj: Dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Dataloaders
    loaders = make_dataloaders(cfg.data_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    device = cfg.device

    # Infer d_in
    xb = next(iter(loaders["train"]))["x"]
    cfg.d_in = int(xb.shape[1])

    # Save config (with inferred d_in)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Model
    model = PLEModel(
        d_in=cfg.d_in,
        d_model=cfg.d_model,
        expert_hidden=cfg.expert_hidden,
        num_levels=cfg.num_levels,
        num_shared_experts=cfg.num_shared_experts,
        num_task_experts=cfg.num_task_experts,
        dropout=cfg.dropout,
    ).to(device)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Pos weights
    if cfg.use_pos_weight:
        inc_pw, nm_pw = compute_pos_weight(loaders["train"])
        pos_w = (inc_pw, nm_pw)
    else:
        pos_w = (None, None)

    # GradScaler per new API if available
    if _USE_NEW_AMP and device == "cuda":
        scaler = _amp.GradScaler("cuda", enabled=cfg.mixed_precision)
    else:
        scaler = _amp.GradScaler(enabled=cfg.mixed_precision and (device == "cuda"))

    best_auc = -float("inf")
    patience = 3
    bad_epochs = 0
    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        tr = train_epoch(
            model,
            loaders["train"],
            device,
            optimizer,
            scaler,
            cfg.w_income,
            cfg.w_never_married,
            pos_w,
            cfg.grad_clip,
            cfg.mixed_precision,
        )
        va = evaluate(model, loaders["val"], device, pos_w)
        scheduler.step()

        comb = va["combined_auc"]
        comb_for_select = comb if not np.isnan(comb) else -float("inf")
        line = {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "lr": optimizer.param_groups[0]["lr"],
        }
        save_jsonl(metrics_path, line)
        print(
            f"Epoch {epoch}/{cfg.epochs} | Train: L={tr['loss']:.3f} inc_auc={tr['inc_auc']:.3f} nm_auc={tr['nm_auc']:.3f} "
            f"| Val: L={va['loss']:.3f} inc_auc={va['inc_auc']:.3f} nm_auc={va['nm_auc']:.3f} | CombAUC={comb:.3f}"
        )

        # Checkpoint last
        torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, os.path.join(cfg.out_dir, "last.pt"))

        # Early stopping + best checkpoint on val combined AUC
        if epoch == 1 or comb_for_select > best_auc:
            best_auc = comb_for_select
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, os.path.join(cfg.out_dir, "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Final Test evaluation
    # Load best checkpoint
    try:
        best_ckpt = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=device)
        model.load_state_dict(best_ckpt["model"])
    except FileNotFoundError:
        print("best.pt not found; falling back to last.pt")
        last_ckpt = torch.load(os.path.join(cfg.out_dir, "last.pt"), map_location=device)
        model.load_state_dict(last_ckpt["model"])
    te = evaluate(model, loaders["test"], device, pos_w)
    report = {
        "test": te,
        "best_val_combined_auc": best_auc,
    }
    with open(os.path.join(cfg.out_dir, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"Test: L={te['loss']:.3f} inc_auc={te['inc_auc']:.3f} nm_auc={te['nm_auc']:.3f} "
        f"inc_acc={te['inc_acc']:.3f} nm_acc={te['nm_acc']:.3f} | CombAUC={te['combined_auc']:.3f}"
    )


if __name__ == "__main__":
    main()
