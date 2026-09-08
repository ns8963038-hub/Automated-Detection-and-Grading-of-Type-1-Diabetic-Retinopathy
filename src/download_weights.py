"""Fetch the trained model checkpoints.

Weights are published as GitHub release assets rather than committed, so a
clone of this repository stays small. Everything needed to run the demo app or
reproduce the reported metrics is in these two files.

Run:  python -m src.download_weights
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request

from src import config as C

REPO = "ns8963038-hub/Automated-Detection-and-Grading-of-Type-1-Diabetic-Retinopathy"
TAG = "v1.0"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

WEIGHTS = {
    "efficientnet_best.pt": "EfficientNet-B3, the main model (QWK 0.9005)",
    "baseline_best.pt": "From-scratch CNN baseline (QWK 0.8706)",
}


def _progress(done: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100.0, done * block * 100.0 / total)
    mb = total / 1_048_576
    bar = "#" * int(pct / 3)
    sys.stdout.write(f"\r    [{bar:<33}] {pct:5.1f}% of {mb:.0f} MB")
    sys.stdout.flush()


def download(force: bool = False) -> bool:
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ok = True

    for name, description in WEIGHTS.items():
        target = C.MODEL_DIR / name
        if target.exists() and not force:
            size = target.stat().st_size / 1_048_576
            print(f"  {name} already present ({size:.0f} MB), skipping.")
            continue

        print(f"  {name} - {description}")
        try:
            urllib.request.urlretrieve(f"{BASE}/{name}", target, _progress)
            print(f"\n    saved to {target}")
        except urllib.error.HTTPError as exc:
            print(f"\n    FAILED ({exc.code}): {BASE}/{name}")
            ok = False
        except Exception as exc:                       # network, disk, ...
            print(f"\n    FAILED ({type(exc).__name__}): {exc}")
            ok = False

    if ok:
        print("\nWeights ready. Next:")
        print("  streamlit run app/app.py")
    else:
        print("\nSome downloads failed. You can also grab them by hand from:")
        print(f"  https://github.com/{REPO}/releases/tag/{TAG}")
        print(f"and place them in {C.MODEL_DIR}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Download trained model weights.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the files already exist.")
    args = ap.parse_args()
    sys.exit(0 if download(force=args.force) else 1)


if __name__ == "__main__":
    main()
