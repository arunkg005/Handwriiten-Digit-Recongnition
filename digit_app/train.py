"""Training pipeline for the handwritten digit recognition MVP."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from .config import ARTIFACT_DIR, CONFUSION_MATRIX_PATH, DATASET_PATH, METRICS_PATH, MODEL_PATH
from .data import flatten_images, load_mnist_dataset, normalize_images, select_subset

MPLCONFIGDIR = ARTIFACT_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


@dataclass(slots=True)
class TrainingArtifacts:
    """Metadata emitted by the persisted training run."""

    model_path: str
    metrics_path: str
    confusion_matrix_path: str | None
    test_accuracy: float
    classes: list[int]
    num_training_examples: int
    num_test_examples: int
    model_name: str


def build_classifier() -> MLPClassifier:
    """Create the production classifier used for canvas digit prediction."""

    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=25,
        early_stopping=True,
        n_iter_no_change=3,
        random_state=42,
    )


def train_and_save(
    dataset_path: str | Path = DATASET_PATH,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
    confusion_matrix_path: str | Path = CONFUSION_MATRIX_PATH,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    save_confusion_matrix: bool = True,
) -> TrainingArtifacts:
    """Train the current classifier and persist the generated artifacts."""

    dataset = load_mnist_dataset(dataset_path)

    x_train = flatten_images(normalize_images(dataset.x_train))
    y_train = dataset.y_train.astype(np.int64)
    x_test = flatten_images(normalize_images(dataset.x_test))
    y_test = dataset.y_test.astype(np.int64)

    x_train, y_train = select_subset(x_train, y_train, max_train_samples)
    x_test, y_test = select_subset(x_test, y_test, max_test_samples)

    classifier = build_classifier()
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    model_output_path = Path(model_path)
    metrics_output_path = Path(metrics_path)
    confusion_output_path = Path(confusion_matrix_path)
    artifact_dir = model_output_path.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, model_output_path)

    confusion_path_value: str | None = None
    if save_confusion_matrix:
        cm = confusion_matrix(y_test, predictions, labels=classifier.classes_)
        figure, axis = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_).plot(
            ax=axis,
            cmap="Blues",
            colorbar=False,
        )
        axis.set_title("MNIST MLP Confusion Matrix")
        figure.tight_layout()
        figure.savefig(confusion_output_path, dpi=150)
        plt.close(figure)
        confusion_path_value = str(confusion_output_path)

    artifacts = TrainingArtifacts(
        model_path=str(model_output_path),
        metrics_path=str(metrics_output_path),
        confusion_matrix_path=confusion_path_value,
        test_accuracy=accuracy,
        classes=[int(label) for label in classifier.classes_.tolist()],
        num_training_examples=int(len(x_train)),
        num_test_examples=int(len(x_test)),
        model_name="mlp_classifier",
    )

    with metrics_output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(asdict(artifacts), metrics_file, indent=2)

    return artifacts


if __name__ == "__main__":
    artifacts = train_and_save()
    print(json.dumps(asdict(artifacts), indent=2))
