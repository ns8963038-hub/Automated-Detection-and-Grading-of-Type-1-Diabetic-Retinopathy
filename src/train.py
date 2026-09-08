"""Stage 3/4: train a model and checkpoint the best epoch.

Examples:
    python -m src.train --model efficientnet
    python -m src.train --model baseline
    python -m src.train --model efficientnet --epochs 5 --limit 200   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd
import torch

from src import config as C
from src.dataset import build_dataloaders, class_weights
from src.engine import (build_criterion, build_optimizer, cosine_with_warmup,
                        evaluate, train_one_epoch)
from src.models import (ACTIVATIONS, build_model, count_parameters,
                        get_device)
from src.utils import seed_worker, set_seed


def make_run_name(model_key: str, optimizer: str, activation: str,
                  tag: str | None) -> str:
    """Name a run so ablation variants never overwrite each other.

    Default settings keep the plain model name (`efficientnet_best.pt`), so
    the standard pipeline is unchanged; any deviation gets a suffix.
    """
    if tag:
        return tag
    if optimizer == C.OPTIMIZER and activation == C.ACTIVATION:
        return model_key
    return f"{model_key}_{optimizer}_{activation}"


def train(model_key: str = C.DEFAULT_MODEL, epochs: int | None = None,
          batch_size: int | None = None, lr: float | None = None,
          limit: int | None = None, workers: int = C.NUM_WORKERS,
          no_pretrained: bool = False, optimizer_name: str = C.OPTIMIZER,
          activation: str = C.ACTIVATION, tag: str | None = None,
          patience: int = C.EARLY_STOP_PATIENCE) -> dict:

    cfg = dict(C.MODELS[model_key])
    if epochs is not None:
        cfg["epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if lr is not None:
        cfg["lr"] = lr
    cfg["optimizer"] = optimizer_name
    cfg["activation"] = activation

    device = get_device()
    set_seed(C.SEED)
    run_name = make_run_name(model_key, optimizer_name, activation, tag)

    # The activation only applies to the from-scratch baseline; pretrained
    # backbones ship with their own (EfficientNet uses SiLU, ResNet ReLU) and
    # replacing them would invalidate the ImageNet weights.
    act_note = "" if cfg["arch"] == "baseline_cnn" else "  (backbone's own)"

    print("=" * 68)
    print(f"  Run        : {run_name}")
    print(f"  Model      : {model_key}  ({cfg['arch']})")
    print(f"  Device     : {device}")
    print(f"  Image size : {cfg['img_size']}   Batch: {cfg['batch_size']}")
    print(f"  Epochs     : {cfg['epochs']}   LR: {cfg['lr']}")
    print(f"  Patience   : {patience}"
          f"{'  (early stopping disabled)' if patience >= cfg['epochs'] else ''}")
    print(f"  Optimizer  : {optimizer_name}")
    print(f"  Activation : {activation}{act_note}")
    print("=" * 68)

    loaders = build_dataloaders(cfg["img_size"], cfg["batch_size"],
                                num_workers=workers, seed=C.SEED)

    # --limit trims the loaders for a fast end-to-end smoke test.
    if limit:
        from torch.utils.data import DataLoader, Subset
        for split, ld in loaders.items():
            n = min(limit, len(ld.dataset))
            loaders[split] = DataLoader(
                Subset(ld.dataset, range(n)), batch_size=cfg["batch_size"],
                shuffle=(split == "train"), num_workers=workers,
                worker_init_fn=seed_worker,
            )
        print(f"  [smoke test] loaders capped at {limit} samples each")

    model = build_model(model_key, pretrained=False if no_pretrained else None,
                        activation=activation)
    model.to(device)
    total, trainable = count_parameters(model)
    print(f"  Parameters : {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable\n")

    weights = class_weights() if C.USE_CLASS_WEIGHTS else None
    if weights is not None:
        print("  Class weights:", [f"{w:.2f}" for w in weights.tolist()], "\n")
    criterion = build_criterion(device, weights)

    optimizer = build_optimizer(optimizer_name, model.parameters(),
                                lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = cosine_with_warmup(optimizer, C.WARMUP_EPOCHS, cfg["epochs"],
                                   len(loaders["train"]))

    ckpt_path = C.MODEL_DIR / f"{run_name}_best.pt"
    log_path = C.LOG_DIR / f"{run_name}_history.csv"

    best_score, best_epoch, stale = -1.0, -1, 0
    history = []
    start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer,
                             device, scheduler, epoch)
        va, *_ = evaluate(model, loaders["val"], criterion, device)
        dt = time.time() - t0

        score = va[C.MONITOR_METRIC]
        improved = score > best_score

        print(f"epoch {epoch:02d}/{cfg['epochs']}  {dt:5.1f}s  "
              f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.3f}  |  "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.3f} "
              f"qwk {va['qwk']:.4f}  sens {va['referable_sensitivity']:.3f}"
              f"{'  <- best' if improved else ''}")

        history.append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(dt, 2),
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        })
        pd.DataFrame(history).to_csv(log_path, index=False)

        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "model_key": model_key,
                "run_name": run_name,
                "arch": cfg["arch"],
                "img_size": cfg["img_size"],
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": va,
                "config": cfg,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            }, ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"\nEarly stopping: no {C.MONITOR_METRIC} improvement "
                      f"for {stale} epochs.")
                break

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed/60:.1f} min. "
          f"Best {C.MONITOR_METRIC}={best_score:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint: {ckpt_path}")
    print(f"History   : {log_path}")

    summary = {
        "model_key": model_key, "run_name": run_name, "arch": cfg["arch"],
        "optimizer": optimizer_name, "activation": activation,
        "best_epoch": best_epoch, f"best_val_{C.MONITOR_METRIC}": best_score,
        "epochs_run": len(history), "minutes": round(elapsed / 60, 2),
        "params_millions": round(total / 1e6, 2),
    }
    (C.LOG_DIR / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a DR grading CNN.")
    ap.add_argument("--model", default=C.DEFAULT_MODEL, choices=list(C.MODELS))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--limit", type=int,
                    help="Cap samples per split for a quick smoke test.")
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Train the transfer model from random init (ablation).")
    ap.add_argument("--optimizer", default=C.OPTIMIZER,
                    choices=["adamw", "adam", "sgd", "rmsprop"],
                    help="How gradients become weight updates.")
    ap.add_argument("--activation", default=C.ACTIVATION,
                    choices=list(ACTIVATIONS),
                    help="Non-linearity inside the network (baseline CNN only).")
    ap.add_argument("--tag", default=None,
                    help="Explicit run name for checkpoints and logs.")
    ap.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE,
                    help="Early-stop patience. Set >= --epochs to disable.")
    args = ap.parse_args()

    train(args.model, args.epochs, args.batch_size, args.lr,
          args.limit, args.workers, args.no_pretrained,
          args.optimizer, args.activation, args.tag, args.patience)


if __name__ == "__main__":
    main()
