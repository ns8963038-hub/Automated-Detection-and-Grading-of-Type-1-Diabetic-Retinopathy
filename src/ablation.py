"""Run an ablation sweep and tabulate the result.

Answers the two questions an examiner is most likely to ask about the training
setup, with measurements rather than assertions:

    Which activation?   relu / leaky_relu / gelu / silu / elu / mish
    Which optimizer?    adamw / adam / sgd / rmsprop

Both sweeps use the `baseline` model, because it is the only one trained from
scratch -- pretrained backbones ship with their own activations and swapping
those would invalidate the ImageNet weights. The baseline trains in roughly
17 minutes, so a sweep is affordable.

Examples:
    python -m src.ablation --sweep optimizer
    python -m src.ablation --sweep activation --epochs 15
    python -m src.ablation --sweep both --epochs 10
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src import config as C
from src.models import ACTIVATIONS
from src.train import make_run_name, train

OPTIMIZERS = ["adamw", "adam", "sgd", "rmsprop"]


def run_sweep(sweep: str, model_key: str = "baseline",
              epochs: int | None = None, limit: int | None = None,
              workers: int = C.NUM_WORKERS) -> pd.DataFrame:
    if sweep == "optimizer":
        combos = [(o, C.ACTIVATION) for o in OPTIMIZERS]
    elif sweep == "activation":
        combos = [(C.OPTIMIZER, a) for a in ACTIVATIONS]
    else:
        combos = [(o, a) for o in OPTIMIZERS for a in ACTIVATIONS]

    print(f"\nAblation sweep '{sweep}' on `{model_key}`: {len(combos)} runs\n")

    rows, started = [], time.time()
    for i, (opt, act) in enumerate(combos, 1):
        run_name = make_run_name(model_key, opt, act, None)
        print(f"\n{'#' * 68}\n#  [{i}/{len(combos)}]  {run_name}\n{'#' * 68}")
        try:
            summary = train(model_key=model_key, epochs=epochs, limit=limit,
                            workers=workers, optimizer_name=opt, activation=act)
            rows.append({
                "optimizer": opt,
                "activation": act,
                "run": run_name,
                "best_val_qwk": summary[f"best_val_{C.MONITOR_METRIC}"],
                "best_epoch": summary["best_epoch"],
                "epochs_run": summary["epochs_run"],
                "minutes": summary["minutes"],
            })
        except Exception as exc:                      # keep the sweep going
            print(f"  RUN FAILED ({type(exc).__name__}): {exc}")
            rows.append({"optimizer": opt, "activation": act, "run": run_name,
                         "best_val_qwk": float("nan"), "best_epoch": -1,
                         "epochs_run": 0, "minutes": 0.0})

    df = pd.DataFrame(rows).sort_values("best_val_qwk", ascending=False)
    df = df.reset_index(drop=True)

    print(f"\n{'=' * 72}")
    print(f"  Ablation: {sweep}   ({(time.time() - started) / 60:.1f} min total)")
    print(f"{'=' * 72}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"{'=' * 72}\n")

    csv_out = C.LOG_DIR / f"ablation_{sweep}.csv"
    df.to_csv(csv_out, index=False)
    print(f"csv -> {csv_out}")

    md_out = C.LOG_DIR / f"ablation_{sweep}.md"
    md_out.write_text(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"md  -> {md_out}")

    (C.LOG_DIR / f"ablation_{sweep}.json").write_text(
        json.dumps(df.to_dict("records"), indent=2))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation sweep over training components.")
    ap.add_argument("--sweep", default="optimizer",
                    choices=["optimizer", "activation", "both"])
    ap.add_argument("--model", default="baseline", choices=list(C.MODELS))
    ap.add_argument("--epochs", type=int,
                    help="Override epochs -- keep sweeps short.")
    ap.add_argument("--limit", type=int, help="Cap samples (smoke test).")
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    args = ap.parse_args()

    run_sweep(args.sweep, args.model, args.epochs, args.limit, args.workers)


if __name__ == "__main__":
    main()
