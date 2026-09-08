"""Dataset-chapter figures: sample grids and the preprocessing walkthrough.

Run:  python -m src.eda
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.preprocess import ben_graham, circle_crop, crop_black_borders

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e6e3"
# Single hue, light->dark: grade is an ordered tier, not a set of identities.
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def _raw_path(image_id: str):
    for ext in (".png", ".jpg"):
        p = C.IMAGES_RAW / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def _read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def samples_per_grade(df: pd.DataFrame, n: int = 4, seed: int = C.SEED) -> None:
    """A grid of raw fundus images, one row per DR grade."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(C.NUM_CLASSES, n, figsize=(3 * n, 3 * C.NUM_CLASSES),
                             facecolor=SURFACE)
    for grade in range(C.NUM_CLASSES):
        pool = df[df["diagnosis"] == grade]["id_code"].tolist()
        picks = rng.choice(pool, size=min(n, len(pool)), replace=False) if pool else []
        for col in range(n):
            ax = axes[grade, col]
            ax.axis("off")
            if col < len(picks):
                path = _raw_path(picks[col])
                img = _read_rgb(path) if path else None
                if img is not None:
                    ax.imshow(img)
            if col == 0:
                ax.set_ylabel(C.CLASS_SHORT[grade])
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_ylabel(f"Grade {grade}\n{C.CLASS_SHORT[grade]}",
                              fontsize=11, color=INK_PRIMARY, rotation=0,
                              ha="right", va="center", labelpad=42)
    fig.suptitle("APTOS 2019 - sample fundus images by DR grade",
                 fontsize=15, fontweight="bold", color=INK_PRIMARY, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = C.FIGURE_DIR / "samples_by_grade.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"figure -> {out}")


def preprocessing_steps(df: pd.DataFrame, seed: int = C.SEED) -> None:
    """Show every stage-2 operation applied to one image, in order."""
    rng = np.random.default_rng(seed)
    # Pick a higher grade -- lesions make the enhancement visible.
    pool = df[df["diagnosis"] >= 3]["id_code"].tolist() or df["id_code"].tolist()
    path = None
    for candidate in rng.permutation(pool):
        path = _raw_path(candidate)
        if path:
            break
    if path is None:
        print("No raw image available for the preprocessing figure.")
        return

    raw = _read_rgb(path)
    cropped = crop_black_borders(raw)
    resized = cv2.resize(cropped, (C.PREPROCESS_SIZE, C.PREPROCESS_SIZE),
                         interpolation=cv2.INTER_AREA)
    enhanced = ben_graham(resized)
    masked = circle_crop(enhanced)

    panels = [
        (raw, f"1. Original\n{raw.shape[1]}x{raw.shape[0]}"),
        (cropped, f"2. Black border cropped\n{cropped.shape[1]}x{cropped.shape[0]}"),
        (resized, f"3. Resized\n{C.PREPROCESS_SIZE}x{C.PREPROCESS_SIZE}"),
        (enhanced, "4. Ben Graham enhancement\n(blur subtracted)"),
        (masked, "5. Circular field-of-view mask\n(network input)"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 5.2),
                             facecolor=SURFACE)
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        # Titles go below each panel: the panels have different aspect ratios,
        # so top-aligned titles sit at different heights and collide with the
        # figure title.
        ax.set_xlabel(title, fontsize=11, color=INK_PRIMARY, labelpad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Stage 2 - preprocessing pipeline", fontsize=15,
                 fontweight="bold", color=INK_PRIMARY, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = C.FIGURE_DIR / "preprocessing_steps.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"figure -> {out}  (source image: {path.name})")


def grade_distribution(df: pd.DataFrame) -> None:
    counts = df["diagnosis"].value_counts().sort_index()
    counts = counts.reindex(range(C.NUM_CLASSES), fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.bar([C.CLASS_SHORT[i] for i in counts.index], counts.values,
                  color=ORDINAL, width=0.68, zorder=3)
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.012,
                f"{v}\n{v / total:.1%}", ha="center", va="bottom",
                fontsize=10, color=INK_SECONDARY)

    ax.set_ylabel("images", color=INK_SECONDARY)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_title(f"Class imbalance in APTOS 2019 (n={total})",
                 fontsize=13, fontweight="bold", color=INK_PRIMARY)
    fig.tight_layout()
    out = C.FIGURE_DIR / "class_imbalance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dataset EDA figures.")
    ap.add_argument("--samples", type=int, default=4,
                    help="Images per grade in the sample grid.")
    args = ap.parse_args()

    if not C.LABELS_CSV.exists():
        print(f"{C.LABELS_CSV} not found. Run `python -m src.download_data` first.")
        return
    df = pd.read_csv(C.LABELS_CSV)

    print(f"Dataset: {len(df)} images\n")
    grade_distribution(df)
    samples_per_grade(df, n=args.samples)
    preprocessing_steps(df)


if __name__ == "__main__":
    main()
