"""Dataset loading and shared numeric preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class MNISTDataset:
    """Container for the local MNIST split."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_mnist_dataset(dataset_path: str | Path) -> MNISTDataset:
    """Load the local MNIST NPZ file into memory."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"MNIST dataset not found at {path}")

    with np.load(path) as dataset:
        required_keys = {"x_train", "y_train", "x_test", "y_test"}
        missing_keys = required_keys.difference(dataset.files)
        if missing_keys:
            joined = ", ".join(sorted(missing_keys))
            raise KeyError(f"MNIST dataset is missing required arrays: {joined}")

        return MNISTDataset(
            x_train=dataset["x_train"].copy(),
            y_train=dataset["y_train"].copy(),
            x_test=dataset["x_test"].copy(),
            y_test=dataset["y_test"].copy(),
        )


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize images into float32 values in the [0, 1] range."""

    normalized = images.astype(np.float32)
    if normalized.max(initial=0.0) > 1.0:
        normalized /= 255.0
    return np.clip(normalized, 0.0, 1.0)


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten image batches into vectors suitable for sklearn models."""

    if images.ndim == 2:
        images = images[np.newaxis, ...]
    return images.reshape(images.shape[0], -1)


def select_subset(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim a dataset deterministically for faster tests or smoke runs."""

    if max_samples is None or max_samples >= len(features):
        return features, labels
    return features[:max_samples], labels[:max_samples]
