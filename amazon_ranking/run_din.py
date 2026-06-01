"""Train + evaluate sequence-ranking models on real Amazon reviews (E4 / E5).

Runs one or more models (DIN + non-attention baselines) on a **single shared**
``SequenceRankingDataModule`` build, so every model sees identical splits,
negatives, and eval candidates — a fair comparison. Writes a combined result
JSON and prints a leaderboard.

Designed to run on Colab GPU via SSH + ``nohup`` (results to a Drive path), and
to smoke-test locally on CPU with ``--synthetic`` (no download). All models reuse
the reviewed amazon_ranking components unchanged.

Examples
--------
Local CPU smoke (no download)::

    python -m amazon_ranking.run_din --synthetic --models din,dcn,deepfm,meanpool \
        --epochs 5 --out /tmp/rank_synth.json

Colab GPU fair comparison on Beauty::

    python -m amazon_ranking.run_din --dataset Beauty --models din,dcn,deepfm,meanpool \
        --data-dir /content/amazon \
        --out /content/drive/MyDrive/colab/data/amazon/ranking_Beauty_results.json \
        --heartbeat /content/drive/MyDrive/colab/ranking_heartbeat.json
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request

import pandas as pd
import torch

from tiger_semantic_id.src.data import DatasetConfig, download_dataset

from amazon_ranking.src.baselines import build_baseline
from amazon_ranking.src.datamodule import DataModuleConfig, SequenceRankingDataModule
from amazon_ranking.src.din import DIN, DINTrainConfig, evaluate_ranking, train_din
from amazon_ranking.src.reviews_io import load_reviews_streaming


def _synthetic_reviews(n_users: int = 200, items_per_user: int = 12, n_items: int = 400) -> pd.DataFrame:
    """A tiny deterministic reviews frame for local CPU smoke tests (no download)."""
    rows = []
    ts = 1
    for u in range(n_users):
        for i in range(items_per_user):
            rows.append({"user_id": f"u{u}", "item_id": f"i{(u * 5 + i) % n_items}", "ts": ts})
            ts += 1
    return pd.DataFrame(rows)


def _ensure_downloaded(data_dir: str, dataset_name: str):
    """Return ``(reviews_path, format)``, downloading the reviews file if absent."""
    cfg = DatasetConfig(dataset_name=dataset_name)
    reviews_url, _ = cfg.get_urls()
    reviews_path, _ = download_dataset(data_dir, dataset_name)
    if not os.path.isfile(reviews_path):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Downloading {reviews_url} -> {reviews_path}", flush=True)
        urllib.request.urlretrieve(reviews_url, reviews_path)
    return reviews_path, cfg.get_format()


def load_reviews(args) -> pd.DataFrame:
    if args.synthetic:
        return _synthetic_reviews()
    reviews_path, fmt = _ensure_downloaded(args.data_dir, args.dataset)
    # Streaming projection to [user_id, item_id, ts] — keeps memory bounded on
    # large 2023 dumps (Video_Games) instead of eagerly deserializing full records.
    return load_reviews_streaming(reviews_path, fmt, max_users=args.max_users)


def build_model(name: str, num_items: int, embed_dim: int):
    """Build a model by name. ``din`` -> DIN; otherwise a baselines.* model."""
    if name == "din":
        return DIN(num_items=num_items, embed_dim=embed_dim)
    return build_baseline(name, num_items=num_items, embed_dim=embed_dim)


def _write_heartbeat(path, state) -> None:
    """Atomically write ``{ts, **state}`` to ``path`` (best-effort, never raises)."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"ts": time.time(), **state}, f)
        os.replace(tmp, path)  # atomic; a reader never sees a half-written file
    except Exception as e:  # transient Drive write etc. — log, don't die
        print(f"[heartbeat] write failed: {e}", flush=True)


def start_heartbeat(path, state, interval: int = 30):
    """Daemon thread writing ``{ts, **state}`` to ``path`` every ``interval`` s.

    Tunnel-independent liveness signal; dies with the process. ``state`` is a
    mutable dict updated in place by the caller (``phase``, ``done``). Returns a
    ``threading.Event`` — set it (then the caller does a final synchronous
    ``_write_heartbeat``) so a watcher does not misread post-run staleness as a
    crash. See kb/context/colab/colab-runtime-behavior.md "In-Run Heartbeat".
    """
    if not path:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            _write_heartbeat(path, state)  # best-effort; never kills the daemon
            stop.wait(interval)

    threading.Thread(target=beat, daemon=True).start()
    return stop


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Beauty", choices=["Beauty", "Video_Games"])
    p.add_argument("--models", default="din", help="comma list: din,dcn,deepfm,meanpool")
    p.add_argument("--data-dir", default="/content/drive/MyDrive/colab/data/amazon")
    p.add_argument("--out", default=None, help="combined result JSON path")
    p.add_argument("--heartbeat", default=None, help="Drive path for the liveness heartbeat JSON")
    p.add_argument("--synthetic", action="store_true", help="tiny synthetic frame; skip download (local smoke)")
    p.add_argument("--max-users", type=int, default=0, help="subsample to the first N users (0 = all)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-hist-len", type=int, default=20)
    p.add_argument("--min-user-interactions", type=int, default=5)
    p.add_argument("--n-eval-negatives", type=int, default=100)
    p.add_argument("--n-train-negatives", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    t_start = time.time()

    reviews = load_reviews(args)
    print(
        f"reviews: {len(reviews)} rows, {reviews['user_id'].nunique()} users, "
        f"{reviews['item_id'].nunique()} items",
        flush=True,
    )

    # ONE shared build so every model sees identical splits / negatives / candidates.
    dm_cfg = DataModuleConfig(
        max_hist_len=args.max_hist_len,
        min_user_interactions=args.min_user_interactions,
        n_eval_negatives=args.n_eval_negatives,
        n_train_negatives=args.n_train_negatives,
        seed=args.seed,
    )
    dm = SequenceRankingDataModule.from_reviews(reviews, dm_cfg)
    dm.build()
    n_train = len(dm.train_examples())
    print(f"num_users={dm.num_users} num_items={dm.num_items} train_examples={n_train}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = "synthetic" if args.synthetic else args.dataset
    hb_state = {"phase": "init", "done": [], "models": models, "finished": False}
    hb_stop = start_heartbeat(args.heartbeat, hb_state)

    # Per-model result cache so a runtime death (the ~77-min Colab pattern) only
    # costs the in-flight model: completed models are reloaded and skipped on resume.
    cache_dir = os.path.dirname(args.out) if args.out else "."

    def _pm_path(name):
        return os.path.join(cache_dir or ".", f"{dataset_name}_{name}_result.json")

    results = {}
    for name in models:
        hb_state["phase"] = name
        pmp = _pm_path(name)
        if os.path.isfile(pmp):  # resume: already computed in a prior session
            with open(pmp) as f:
                results[name] = json.load(f)
            hb_state["done"] = list(results.keys())
            print(f"[{name}] cached -> skip ({pmp})", flush=True)
            continue
        # Seed BEFORE constructing the model so init is reproducible per model.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = build_model(name, dm.num_items, args.embed_dim).to(device)
        train_cfg = DINTrainConfig(
            embed_dim=args.embed_dim, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed
        )
        t0 = time.time()
        losses = train_din(model, dm.train_examples(), max_hist_len=args.max_hist_len, cfg=train_cfg, device=device)
        metrics = evaluate_ranking(
            model, dm.eval_examples("test"), max_hist_len=args.max_hist_len, ks=(5, 10, 20), device=device
        )
        results[name] = {
            "losses": losses,
            "metrics": metrics,
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "wall_sec": round(time.time() - t0, 1),
        }
        os.makedirs(cache_dir or ".", exist_ok=True)
        with open(pmp, "w") as f:  # persist immediately so it survives a later crash
            json.dump(results[name], f)
        hb_state["done"] = list(results.keys())
        print(
            f"[{name}] sampled_auc={metrics['sampled_auc']:.4f} recall@10={metrics['recall@10']:.4f} "
            f"ndcg@10={metrics['ndcg@10']:.4f} loss {losses['first_epoch_loss']:.4f}->{losses['last_epoch_loss']:.4f} "
            f"wall={results[name]['wall_sec']}s",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "dataset": dataset_name,
        "device": str(device),
        "num_users": dm.num_users,
        "num_items": dm.num_items,
        "n_train_examples": n_train,
        "config": {
            "epochs": args.epochs,
            "embed_dim": args.embed_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_hist_len": args.max_hist_len,
            "min_user_interactions": args.min_user_interactions,
            "n_eval_negatives": args.n_eval_negatives,
            "n_train_negatives": args.n_train_negatives,
            "seed": args.seed,
        },
        "total_wall_sec": round(time.time() - t_start, 1),
        "models": results,
    }
    out = args.out or f"ranking_{payload['dataset']}_results.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}", flush=True)

    # Final synchronous heartbeat so a watcher sees completion, not stale==dead.
    if args.heartbeat:
        hb_state["phase"] = "finished"
        hb_state["finished"] = True
        if hb_stop is not None:
            hb_stop.set()
        _write_heartbeat(args.heartbeat, hb_state)

    # Leaderboard (sorted by sampled_auc desc) + per-model DoD.
    print("\n=== LEADERBOARD (" + payload["dataset"] + ") ===", flush=True)
    print(f"{'model':<10} {'samp_auc':>9} {'recall@10':>10} {'ndcg@10':>9} {'mrr@10':>8} {'params':>10} {'wall_s':>7}", flush=True)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["metrics"]["sampled_auc"]):
        m = r["metrics"]
        print(
            f"{name:<10} {m['sampled_auc']:>9.4f} {m['recall@10']:>10.4f} {m['ndcg@10']:>9.4f} "
            f"{m['mrr@10']:>8.4f} {r['num_parameters']:>10,} {r['wall_sec']:>7.1f}",
            flush=True,
        )
    worst = min(r["metrics"]["sampled_auc"] for r in results.values())
    print(f"\nDoD(all sampled_auc>0.5): {'PASS' if worst > 0.5 else 'FAIL'} (min={worst:.4f})", flush=True)


if __name__ == "__main__":
    main()
