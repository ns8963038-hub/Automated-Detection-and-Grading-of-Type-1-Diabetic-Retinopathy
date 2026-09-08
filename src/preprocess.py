"""Stage 2 + 3: image preprocessing and the train/val/test split.

Fundus photographs arrive at wildly different resolutions, aspect ratios and
illumination levels because they come from several clinics and camera models.
This module normalises them once, writes the result to disk, and builds a
stratified split so every later stage reads cheap, uniform inputs.

Run:  python -m src.preprocess
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import config as C


# --------------------------------------------------------------------------
# Individual preprocessing operations
# --------------------------------------------------------------------------
def crop_black_borders(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Trim the uninformative black frame that surrounds most fundus images.

    A mask of "bright enough" pixels is projected onto each axis and the image
    is cut to that bounding box. Falls back to the original image when the
    photograph is so dark that the mask collapses.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    mask = gray > tol
    if mask.sum() == 0:
        return img

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    cropped = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    # A degenerate crop means the threshold ate the whole image.
    if cropped.size == 0 or min(cropped.shape[:2]) < 10:
        return img
    return cropped


def circle_crop(img: np.ndarray) -> np.ndarray:
    """Mask everything outside the circular retinal field of view.

    Corner pixels carry no clinical signal but do carry camera-specific
    artefacts, which a CNN will happily latch onto as a shortcut.
    """
    h, w = img.shape[:2]
    centre = (w // 2, h // 2)
    radius = min(centre[0], centre[1])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, centre, radius, 255, thickness=-1)
    return cv2.bitwise_and(img, img, mask=mask)


def ben_graham(img: np.ndarray, sigma_divisor: int = C.BEN_GRAHAM_SIGMA) -> np.ndarray:
    """Subtract a heavily blurred copy of the image from itself.

    This is the preprocessing from Ben Graham's winning solution to the 2015
    Diabetic Retinopathy competition. Removing the low-frequency component
    cancels out per-camera colour casts and uneven illumination, leaving the
    high-frequency detail -- microaneurysms, haemorrhages, exudates -- clearly
    visible. It is the single highest-impact preprocessing step for this task.
    """
    size = max(img.shape[:2])
    sigma = max(size / sigma_divisor, 1.0)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def preprocess_image(img: np.ndarray, size: int = C.PREPROCESS_SIZE) -> np.ndarray:
    """Full stage-2 chain: crop -> square resize -> circular mask -> enhance."""
    img = crop_black_borders(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if C.BEN_GRAHAM:
        img = ben_graham(img)
    if C.CIRCLE_CROP:
        img = circle_crop(img)
    return img


def load_and_preprocess(path, size: int = C.PREPROCESS_SIZE) -> np.ndarray | None:
    """Read an image from disk and run the preprocessing chain. RGB output."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_image(img, size=size)


# --------------------------------------------------------------------------
# Batch cache builder
# --------------------------------------------------------------------------
def _process_one(args) -> tuple[str, bool]:
    src_path, dst_path, size = args
    try:
        out = load_and_preprocess(src_path, size=size)
        if out is None:
            return str(src_path), False
        # cv2 writes BGR, so flip back before saving
        cv2.imwrite(str(dst_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return str(src_path), True
    except Exception:
        return str(src_path), False


def build_cache(overwrite: bool = False, workers: int = 8) -> pd.DataFrame:
    """Preprocess every training image once and cache it as PNG."""
    if not C.LABELS_CSV.exists():
        raise FileNotFoundError(
            f"{C.LABELS_CSV} not found. Run `python -m src.download_data` first."
        )

    df = pd.read_csv(C.LABELS_CSV)
    C.IMAGES_PROCESSED.mkdir(parents=True, exist_ok=True)

    jobs = []
    for image_id in df["id_code"]:
        src = C.IMAGES_RAW / f"{image_id}.png"
        if not src.exists():
            src = C.IMAGES_RAW / f"{image_id}.jpg"
        dst = C.IMAGES_PROCESSED / f"{image_id}.png"
        if dst.exists() and not overwrite:
            continue
        jobs.append((src, dst, C.PREPROCESS_SIZE))

    if not jobs:
        print(f"Cache already complete: {len(df)} images in {C.IMAGES_PROCESSED}")
        return df

    print(f"Preprocessing {len(jobs)} images at {C.PREPROCESS_SIZE}px "
          f"(circle_crop={C.CIRCLE_CROP}, ben_graham={C.BEN_GRAHAM})")

    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_one, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
            path, ok = fut.result()
            if not ok:
                failures.append(path)

    if failures:
        print(f"WARNING: {len(failures)} images failed. First few: {failures[:5]}")
    return df


# --------------------------------------------------------------------------
# Stage 3: stratified split
# --------------------------------------------------------------------------
def image_hashes(ids) -> pd.Series:
    """MD5 of each cached image, used to detect duplicate photographs.

    APTOS 2019 contains the same photograph more than once under different
    id_codes. Hashing the preprocessed file is enough to find them, and it is
    exact rather than approximate.
    """
    out = {}
    for image_id in tqdm(list(ids), desc="hashing", unit="img", leave=False):
        path = C.IMAGES_PROCESSED / f"{image_id}.png"
        out[image_id] = hashlib.md5(path.read_bytes()).hexdigest()
    return pd.Series(out)


def build_splits(force: bool = False, group_aware: bool = True) -> pd.DataFrame:
    """Split into train/val/test, stratified so every grade keeps its ratio.

    Two things make a naive split wrong here.

    **Stratification.** Grades 3 and 4 have only ~200-300 examples each, so a
    random split would give wildly different class balances across the three
    sets and make the validation curve meaningless.

    **Grouping.** APTOS contains duplicate photographs under different
    id_codes. If copies of one image land in both train and test, the test set
    is no longer held out for those rows. Splitting on the image hash rather
    than the row keeps every copy of a photograph inside a single split.
    """
    if C.SPLITS_CSV.exists() and not force:
        print(f"Reusing existing split: {C.SPLITS_CSV}")
        return pd.read_csv(C.SPLITS_CSV)

    df = pd.read_csv(C.LABELS_CSV)
    df = df.rename(columns={"diagnosis": "label"})

    # Only keep rows whose cached image actually exists.
    df["exists"] = df["id_code"].map(
        lambda i: (C.IMAGES_PROCESSED / f"{i}.png").exists()
    )
    missing = int((~df["exists"]).sum())
    if missing:
        print(f"Dropping {missing} rows with no cached image.")
    df = df[df["exists"]].drop(columns="exists").reset_index(drop=True)

    if group_aware:
        df["hash"] = df["id_code"].map(image_hashes(df["id_code"]))
        n_dupes = len(df) - df["hash"].nunique()
        conflicting = (df.groupby("hash")["label"].nunique() > 1).sum()
        if n_dupes:
            print(f"Found {n_dupes} duplicate rows across "
                  f"{(df.groupby('hash').size() > 1).sum()} groups; "
                  f"{conflicting} groups disagree on the label.")
            print("Splitting on image content so no photograph spans splits.")
        # One row per unique photograph, labelled by the group's modal grade.
        units = (df.groupby("hash")["label"]
                   .agg(lambda x: x.mode().iat[0])
                   .reset_index())
        key = "hash"
    else:
        units = df[["id_code", "label"]].copy()
        key = "id_code"

    train_val, test = train_test_split(
        units, test_size=C.TEST_SIZE, stratify=units["label"],
        random_state=C.SEED,
    )
    val_ratio = C.VAL_SIZE / (1.0 - C.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["label"],
        random_state=C.SEED,
    )

    assignment = {}
    for frame, name in ((train, "train"), (val, "val"), (test, "test")):
        for k in frame[key]:
            assignment[k] = name
    out = df.assign(split=df[key].map(assignment)).reset_index(drop=True)

    if group_aware:
        # Verify the guarantee rather than trusting it.
        spans = (out.groupby("hash")["split"].nunique() > 1).sum()
        assert spans == 0, f"{spans} image groups still span splits"
        print("Verified: no photograph appears in more than one split.")

    out.to_csv(C.SPLITS_CSV, index=False)
    print(f"\nSplit written to {C.SPLITS_CSV}")
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\nTotals:", out["split"].value_counts().to_dict())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess APTOS images and build splits.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-preprocess images that are already cached.")
    ap.add_argument("--force-split", action="store_true",
                    help="Rebuild splits.csv even if it exists.")
    ap.add_argument("--no-group-split", action="store_true",
                    help="Split per row instead of per unique photograph "
                         "(allows duplicate images to span splits).")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    build_cache(overwrite=args.overwrite, workers=args.workers)
    build_splits(force=args.force_split,
                 group_aware=not args.no_group_split)


if __name__ == "__main__":
    main()
