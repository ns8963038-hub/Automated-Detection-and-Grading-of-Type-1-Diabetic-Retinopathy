"""Build the model comparison table and figure for the results chapter.

Reads every *_test_results.json written by src/evaluate.py and renders one
grouped bar chart plus a markdown/LaTeX table.

Run:  python -m src.compare
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

# Models are distinct identities, so this is a categorical encoding: fixed
# slot order from the validated palette, never cycled or reassigned by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e6e3"
SURFACE = "#fcfcfb"

METRICS = [
    ("accuracy", "Accuracy"),
    ("qwk", "Quadratic\nweighted kappa"),
    ("f1_macro", "Macro F1"),
    ("referable_sensitivity", "Referable DR\nsensitivity"),
]


def collect(split: str = "test") -> pd.DataFrame:
    rows = []
    for key in C.MODELS:
        path = C.LOG_DIR / f"{key}_{split}_results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        summary_path = C.LOG_DIR / f"{key}_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        rows.append({
            "model": key,
            "architecture": data["arch"],
            "params_M": summary.get("params_millions", float("nan")),
            "epochs": summary.get("epochs_run", float("nan")),
            "minutes": summary.get("minutes", float("nan")),
            **{m: data["metrics"].get(m, float("nan")) for m, _ in METRICS},
        })
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, split: str) -> None:
    if df.empty:
        return
    labels = [m[1] for m in METRICS]
    keys = [m[0] for m in METRICS]
    x = np.arange(len(labels))
    n = len(df)
    width = min(0.62 / n, 0.19)

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (_, row) in enumerate(df.iterrows()):
        offset = (i - (n - 1) / 2) * width
        values = [row[k] for k in keys]
        # 2px surface gap between adjacent bars, rounded data-ends.
        bars = ax.bar(x + offset, values, width * 0.88, label=row["model"],
                      color=SERIES[i % len(SERIES)], zorder=3)
        # Direct-label every bar: only a handful, and the exact number is the
        # point of a results table figure.
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK_PRIMARY)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(axis="y", colors=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

    ax.legend(frameon=False, ncols=n, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), fontsize=10, labelcolor=INK_PRIMARY)
    ax.set_title(f"Model comparison on the held-out {split} split",
                 fontsize=13, fontweight="bold", color=INK_PRIMARY, pad=32)

    fig.tight_layout()
    out = C.FIGURE_DIR / f"model_comparison_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare all evaluated models.")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    df = collect(args.split)
    if df.empty:
        print("No evaluation results found. Run src.evaluate first, e.g.:\n"
              "  python -m src.evaluate --model efficientnet")
        return

    df = df.sort_values("qwk", ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 78}")
    print(f"  Model comparison - {args.split} split")
    print(f"{'=' * 78}")
    display = df.rename(columns={m: n.replace("\n", " ") for m, n in METRICS})
    print(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"{'=' * 78}\n")

    csv_out = C.LOG_DIR / f"model_comparison_{args.split}.csv"
    df.to_csv(csv_out, index=False)
    print(f"csv    -> {csv_out}")

    md_out = C.LOG_DIR / f"model_comparison_{args.split}.md"
    md_out.write_text(display.to_markdown(index=False, floatfmt=".4f"))
    print(f"md     -> {md_out}")

    plot_comparison(df, args.split)


if __name__ == "__main__":
    main()
