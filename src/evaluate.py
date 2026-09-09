"""Stage 6/7: evaluate a trained checkpoint on the held-out test split.

Produces every number and figure the results chapter needs:
confusion matrix, per-class precision/recall/F1, QWK, referable-DR
sensitivity, and the training curves.

Run:  python -m src.evaluate --model efficientnet
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src import config as C
from src.dataset import build_dataloaders
from src.engine import build_criterion, evaluate as run_eval, per_class_report
from src.models import build_model, get_device


def available_runs() -> list[str]:
    """Every trained checkpoint on disk, by run name."""
    return sorted(p.stem[:-5] for p in C.MODEL_DIR.glob("*_best.pt"))


def load_checkpoint(run_name: str, device):
    """Load a checkpoint by run name.

    The run name is the model key for a default run and something like
    `baseline_sgd_gelu` for an ablation, so the architecture and activation
    are read back from the checkpoint rather than assumed from the name.
    """
    ckpt_path = C.MODEL_DIR / f"{run_name}_best.pt"
    if not ckpt_path.exists():
        runs = available_runs()
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}.\n"
            + (f"Available runs: {', '.join(runs)}" if runs else
               f"Train one first:  python -m src.train --model {run_name}")
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_key = ckpt.get("model_key", run_name)
    activation = ckpt.get("config", {}).get("activation", C.ACTIVATION)

    model = build_model(model_key, pretrained=False, activation=activation)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Loaded {run_name} (epoch {ckpt['epoch']}, "
          f"val {C.MONITOR_METRIC}={ckpt['val_metrics'][C.MONITOR_METRIC]:.4f})")
    return model, ckpt


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, model_key: str, normalize: bool = True):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, data, fmt, title in [
        (axes[0], cm, "d", "Counts"),
        (axes[1], cm_norm, ".2f", "Normalised by true grade (recall)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=C.CLASS_SHORT, yticklabels=C.CLASS_SHORT,
                    cbar=False, annot_kws={"size": 11})
        ax.set_xlabel("Predicted grade")
        ax.set_ylabel("True grade")
        ax.set_title(title)
    fig.suptitle(f"Confusion Matrix - {model_key}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = C.FIGURE_DIR / f"{model_key}_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out}")
    return cm


def plot_history(model_key: str):
    log_path = C.LOG_DIR / f"{model_key}_history.csv"
    if not log_path.exists():
        return
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="train", marker="o", ms=3)
    axes[0].plot(df["epoch"], df["val_loss"], label="val", marker="s", ms=3)
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="train", marker="o", ms=3)
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val", marker="s", ms=3)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(df["epoch"], df["val_qwk"], color="green", marker="d", ms=3)
    best = df["val_qwk"].idxmax()
    axes[2].axvline(df.loc[best, "epoch"], ls="--", color="grey",
                    label=f"best={df['val_qwk'].max():.4f}")
    axes[2].set_title("Validation QWK"); axes[2].set_xlabel("epoch"); axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Training History - {model_key}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = C.FIGURE_DIR / f"{model_key}_training_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out}")


def plot_grade_distribution():
    """Class balance across the three splits -- a figure for the dataset chapter."""
    df = pd.read_csv(C.SPLITS_CSV)
    pivot = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(["train", "val", "test"])

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=False, ax=ax, colormap="viridis")
    ax.set_xlabel("split"); ax.set_ylabel("images")
    ax.set_title("APTOS 2019 grade distribution per split", fontweight="bold")
    ax.legend([C.CLASS_SHORT[i] for i in pivot.columns], title="DR grade")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = C.FIGURE_DIR / "dataset_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained DR model.")
    ap.add_argument("--model", default=C.DEFAULT_MODEL,
                    help="Run name: a model key, or an ablation tag "
                         "such as baseline_sgd_gelu.")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--tta", type=int, default=0, metavar="N",
                    help="Average predictions over N dihedral views "
                         "(0 = off, 8 = full set). Costs N forward passes.")
    args = ap.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.model, device)
    cfg = ckpt["config"]

    loaders = build_dataloaders(cfg["img_size"], cfg["batch_size"],
                                num_workers=args.workers, splits=(args.split,))
    # Unweighted at evaluation time: class weights are a training device to
    # stop the model collapsing onto grade 0. Leaving them on here would make
    # the reported loss incomparable to any other paper's. Accuracy, QWK and
    # F1 are argmax-based and unaffected either way.
    criterion = build_criterion(device, None)

    metrics, y_true, y_pred, probs = run_eval(
        model, loaders[args.split], criterion, device, desc=args.split,
        tta=args.tta,
    )

    tta_note = f"  ·  TTA x{args.tta}" if args.tta else ""
    print(f"\n{'=' * 62}")
    print(f"  {args.model.upper()} - {args.split} set  (n={len(y_true)}){tta_note}")
    print(f"{'=' * 62}")
    print(f"  Accuracy                : {metrics['accuracy']:.4f}")
    print(f"  Quadratic Weighted Kappa: {metrics['qwk']:.4f}")
    print(f"  Macro F1                : {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1             : {metrics['f1_weighted']:.4f}")
    print("\n  Referable DR (grade >= 2) screening decision:")
    print(f"    Sensitivity (recall)  : {metrics['referable_sensitivity']:.4f}")
    print(f"    Specificity           : {metrics['referable_specificity']:.4f}")
    print(f"    Precision             : {metrics['referable_precision']:.4f}")

    report = per_class_report(y_true, y_pred)
    print("\n  Per-class performance:")
    print(f"    {'grade':24} {'prec':>6} {'recall':>7} {'f1':>6} {'n':>5}")
    for name, r in report.items():
        print(f"    {name:24} {r['precision']:6.3f} {r['recall']:7.3f} "
              f"{r['f1']:6.3f} {r['support']:5d}")
    print(f"{'=' * 62}\n")

    print("Writing figures...")
    cm = plot_confusion(y_true, y_pred, args.model)
    plot_history(args.model)
    plot_grade_distribution()

    results = {
        "model": args.model, "arch": cfg["arch"], "split": args.split,
        "n_samples": int(len(y_true)),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }
    suffix = f"_tta{args.tta}" if args.tta else ""
    out = C.LOG_DIR / f"{args.model}_{args.split}{suffix}_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  results -> {out}")

    np.save(C.LOG_DIR / f"{args.model}_{args.split}{suffix}_probs.npy", probs)


if __name__ == "__main__":
    main()
