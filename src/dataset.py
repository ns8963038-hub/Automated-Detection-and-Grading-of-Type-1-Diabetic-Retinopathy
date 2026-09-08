"""Dataset and augmentation pipeline.

Reads the cached, preprocessed PNGs produced by src/preprocess.py.
"""
from __future__ import annotations

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src import config as C
from src.utils import seed_worker


class RetinaDataset(Dataset):
    """Fundus images with their DR grade (0-4)."""

    def __init__(self, df: pd.DataFrame, transform=None, image_dir=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_dir = image_dir or C.IMAGES_PROCESSED
        self.has_labels = "label" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.image_dir / f"{row['id_code']}.png"

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read cached image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        label = int(row["label"]) if self.has_labels else -1
        return img, torch.tensor(label, dtype=torch.long)


# --------------------------------------------------------------------------
# Augmentations
# --------------------------------------------------------------------------
def train_transform(img_size: int, seed: int | None = None) -> A.Compose:
    """Augmentations chosen to match real variation between fundus cameras.

    Retinal images have no canonical orientation (left and right eyes are
    mirror images of each other), so flips and full rotations are safe and
    effectively multiply the small dataset. Brightness/contrast jitter
    simulates the exposure differences between clinics.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0),
                            ratio=(0.95, 1.05)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1),
                 rotate=(-180, 180), border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                             val_shift_limit=8, p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4),
                        hole_height_range=(0.05, 0.12),
                        hole_width_range=(0.05, 0.12), p=0.25),
        A.Normalize(mean=C.MEAN, std=C.STD),
        ToTensorV2(),
    ], seed=seed)


def eval_transform(img_size: int) -> A.Compose:
    """Deterministic pipeline for validation, test and inference."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=C.MEAN, std=C.STD),
        ToTensorV2(),
    ])


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_splits() -> pd.DataFrame:
    if not C.SPLITS_CSV.exists():
        raise FileNotFoundError(
            f"{C.SPLITS_CSV} not found. Run `python -m src.preprocess` first."
        )
    return pd.read_csv(C.SPLITS_CSV)


def build_dataloaders(img_size: int, batch_size: int,
                      num_workers: int = C.NUM_WORKERS,
                      splits: tuple[str, ...] = ("train", "val", "test"),
                      seed: int | None = None):
    """Return a dict of DataLoaders keyed by split name.

    When `seed` is given, shuffling and per-worker augmentation RNGs are made
    deterministic -- without this each worker process seeds itself from entropy
    and two runs with the same config produce different augmentations.
    """
    df = load_splits()
    loaders = {}
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    for split in splits:
        subset = df[df["split"] == split]
        if subset.empty:
            continue
        is_train = split == "train"
        ds = RetinaDataset(
            subset,
            transform=(train_transform(img_size, seed=seed) if is_train
                       else eval_transform(img_size)),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=False,          # MPS does not benefit from pinned memory
            drop_last=is_train,
            persistent_workers=num_workers > 0,
            generator=generator if is_train else None,
            worker_init_fn=seed_worker if seed is not None else None,
        )
    return loaders


def class_weights() -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Grade 0 accounts for roughly half of APTOS while grade 3 is under 6%.
    Without weighting the model can score ~50% accuracy by predicting "No DR"
    for everything -- exactly the failure mode that matters clinically, since
    missing severe cases is far worse than over-referring mild ones.
    """
    df = load_splits()
    counts = df[df["split"] == "train"]["label"].value_counts().sort_index()
    counts = counts.reindex(range(C.NUM_CLASSES), fill_value=0)
    weights = counts.sum() / (C.NUM_CLASSES * counts.clip(lower=1))
    weights = weights / weights.mean()
    return torch.tensor(weights.values, dtype=torch.float32)
