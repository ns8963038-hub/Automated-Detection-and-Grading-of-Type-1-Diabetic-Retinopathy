"""Stage 1: fetch the APTOS 2019 Blindness Detection dataset from Kaggle.

Two sources are supported.

  mirror       (default)  kaggle datasets download mariaherrerot/aptos2019
                          A complete, faithful copy of the APTOS 2019 training
                          data: all 3,662 graded fundus images, pre-split into
                          train/val/test folders. Needs only a logged-in Kaggle
                          CLI.

  competition             kaggle competitions download -c aptos2019-...
                          The original competition archive. Identical images,
                          but every download 403s until you personally accept
                          the competition rules in a browser:
                            https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules

Both end with the same layout, so everything downstream is source-agnostic:

    data/raw/train.csv          id_code,diagnosis   (3,662 rows)
    data/raw/train_images/      3,662 .png files

Authenticate once with:   .venv/bin/kaggle auth login
Then run:                 python -m src.download_data
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src import config as C

COMPETITION = "aptos2019-blindness-detection"
MIRROR = "mariaherrerot/aptos2019"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"

# The mirror ships the same 3,662 rows split across three CSVs.
MIRROR_CSVS = ["train_1.csv", "valid.csv", "test.csv"]
MIRROR_IMAGE_DIRS = ["train_images", "val_images", "test_images"]

# Official APTOS 2019 grade counts -- used to verify we got the real thing.
EXPECTED_COUNTS = {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def check_auth() -> bool:
    """True if the Kaggle CLI has usable credentials."""
    import os
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    kaggle_dir = Path.home() / ".kaggle"
    for name in ("credentials.json", "access_token", "kaggle.json"):
        if (kaggle_dir / name).exists():
            return True
    return False


def print_auth_help() -> None:
    print(f"""
{'=' * 70}
  Kaggle authentication not found.
{'=' * 70}

  Log in with the browser flow (easiest -- nothing to copy or paste):

      .venv/bin/kaggle auth login

  Then re-run:

      .venv/bin/python -m src.download_data
{'=' * 70}
""")


def _kaggle(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "kaggle", *args]).returncode


# --------------------------------------------------------------------------
# Mirror source
# --------------------------------------------------------------------------
def download_mirror(force: bool = False) -> None:
    """Download the mirror dataset and normalise it to the standard layout."""
    staging = C.DATA_RAW / "_mirror"
    staging.mkdir(parents=True, exist_ok=True)

    marker = staging / ".download_complete"
    if not marker.exists() or force:
        print(f"Downloading {MIRROR} (~8.6 GB). This takes a while.\n")
        rc = _kaggle("datasets", "download", MIRROR, "-p", str(staging), "--unzip")
        if rc != 0:
            print("\nDownload failed. Check your connection and that "
                  "`kaggle auth login` succeeded.")
            sys.exit(1)
        marker.touch()
    else:
        print(f"Reusing existing download in {staging}")

    consolidate_mirror(staging)


def consolidate_mirror(staging: Path) -> None:
    """Merge the mirror's three splits into one train.csv + train_images/."""
    print("\nConsolidating mirror layout...")

    # --- labels -------------------------------------------------------
    frames = []
    for name in MIRROR_CSVS:
        matches = list(staging.rglob(name))
        if not matches:
            print(f"  WARNING: {name} not found in {staging}")
            continue
        frames.append(pd.read_csv(matches[0]))
    if not frames:
        print(f"  ERROR: no label CSVs found under {staging}")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="id_code").reset_index(drop=True)
    df.to_csv(C.LABELS_CSV, index=False)
    print(f"  labels -> {C.LABELS_CSV}  ({len(df)} rows)")

    # --- images -------------------------------------------------------
    C.IMAGES_RAW.mkdir(parents=True, exist_ok=True)
    moved, skipped = 0, 0
    for dirname in MIRROR_IMAGE_DIRS:
        for src_dir in staging.rglob(dirname):
            if not src_dir.is_dir():
                continue
            for img in src_dir.rglob("*.png"):
                dst = C.IMAGES_RAW / img.name
                if dst.exists():
                    skipped += 1
                    continue
                shutil.move(str(img), str(dst))
                moved += 1
    print(f"  images -> {C.IMAGES_RAW}  ({moved} moved, {skipped} already present)")


# --------------------------------------------------------------------------
# Competition source
# --------------------------------------------------------------------------
def download_competition(force: bool = False) -> None:
    zip_path = C.DATA_RAW / f"{COMPETITION}.zip"
    if not zip_path.exists() or force:
        print(f"Downloading {COMPETITION} (~10 GB)...")
        rc = _kaggle("competitions", "download", "-c", COMPETITION,
                     "-p", str(C.DATA_RAW))
        if rc != 0:
            print(f"""
Download failed with 403 Forbidden.

This competition requires you to accept its rules before ANY file can be
downloaded, even with valid credentials. Open this page and click
"I Understand and Accept":

    {RULES_URL}

Or skip the rules entirely by using the mirror instead:

    .venv/bin/python -m src.download_data --source mirror
""")
            sys.exit(1)

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(C.DATA_RAW)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def verify() -> bool:
    """Check the layout and compare the grade counts against official APTOS."""
    if not C.LABELS_CSV.exists():
        print(f"ERROR: {C.LABELS_CSV} missing.")
        return False

    df = pd.read_csv(C.LABELS_CSV)
    n_images = len(list(C.IMAGES_RAW.glob("*.png"))) + len(list(C.IMAGES_RAW.glob("*.jpg")))
    counts = df["diagnosis"].value_counts().sort_index().to_dict()

    print(f"\n{'=' * 62}")
    print(f"  Labels : {len(df)} rows")
    print(f"  Images : {n_images} files in {C.IMAGES_RAW.name}/")
    print("\n  Grade distribution:")
    for grade in range(C.NUM_CLASSES):
        count = counts.get(grade, 0)
        pct = 100 * count / max(len(df), 1)
        print(f"    {C.CLASS_NAMES[grade]:24} {count:5d}  {pct:5.1f}%  "
              f"{'#' * int(pct / 2)}")

    ok = counts == EXPECTED_COUNTS
    print()
    if ok:
        print("  Grade counts match the official APTOS 2019 distribution.")
    else:
        print("  NOTE: counts differ from official APTOS 2019.")
        print(f"    expected {EXPECTED_COUNTS}")
        print(f"    got      {counts}")

    missing = sum(
        1 for i in df["id_code"]
        if not (C.IMAGES_RAW / f"{i}.png").exists()
        and not (C.IMAGES_RAW / f"{i}.jpg").exists()
    )
    if missing:
        print(f"  WARNING: {missing} labelled images are missing from disk.")
    print(f"{'=' * 62}")
    print("\nNext:  .venv/bin/python -m src.preprocess")
    return ok and missing == 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Download the APTOS 2019 dataset.")
    ap.add_argument("--source", default="mirror", choices=["mirror", "competition"],
                    help="mirror avoids the competition rules gate (default).")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if data is already present.")
    ap.add_argument("--verify-only", action="store_true",
                    help="Just check what is already on disk.")
    args = ap.parse_args()

    if args.verify_only:
        verify()
        return

    C.DATA_RAW.mkdir(parents=True, exist_ok=True)

    if C.LABELS_CSV.exists() and C.IMAGES_RAW.exists() and not args.force:
        n = len(list(C.IMAGES_RAW.glob("*.png")))
        if n >= 3600:
            print(f"Dataset already present ({n} images).")
            verify()
            return

    if not check_auth():
        print_auth_help()
        sys.exit(1)

    if args.source == "mirror":
        download_mirror(force=args.force)
    else:
        download_competition(force=args.force)

    verify()


if __name__ == "__main__":
    main()
