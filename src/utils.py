"""Shared helpers: seeding and reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches.

    `torch.manual_seed` alone is not enough: albumentations draws from numpy,
    shuffling and dropout draw from torch, and some ops read PYTHONHASHSEED.
    Without all four, two runs of the same config give different augmentations
    and are not comparable -- which matters when the whole point of the
    ablation sweeps is to attribute a difference to one changed component.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic, distinct seed.

    Two separate problems are handled here.

    Workers are separate processes, so without seeding they draw from system
    entropy and the augmentation stream differs between runs.

    Albumentations 2.x keeps its own RNG inside the Compose object, which is
    *copied* into every worker along with its state. Seeding numpy alone
    therefore leaves all workers generating the identical augmentation
    sequence. Reseeding each worker's Compose by worker id decorrelates them
    while keeping the whole thing reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    dataset = getattr(dataset, "dataset", dataset)   # unwrap Subset
    transform = getattr(dataset, "transform", None)
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(int(worker_seed))
