"""Binary rejector model used on top of the digit classifier."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATASET_PATH, MODEL_PATH, REJECTOR_METRICS_PATH, REJECTOR_MODEL_PATH
from .data import load_mnist_dataset, normalize_images, select_subset
from .preprocess import prepare_canvas_image

REJECTOR_INK_THRESHOLD = 0.12
REJECTOR_THRESHOLD = 0.55


@dataclass(slots=True)
class RejectorArtifacts:
    """Metadata emitted by the persisted rejector training run."""

    model_path: str
    metrics_path: str
    accuracy: float
    num_positive_examples: int
    num_negative_examples: int
    model_name: str


@dataclass(slots=True)
class RejectorDecision:
    """Decision returned by the binary rejector."""

    probability: float
    is_not_a_number: bool
    is_uncertain: bool
    status: str


def build_rejector() -> Pipeline:
    """Create the binary rejector used on top of the digit classifier."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _probability_entropy(probabilities: np.ndarray) -> float:
    normalized = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(-(normalized * np.log(normalized)).sum())


def build_rejector_features(probabilities: np.ndarray, normalized_image: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    top1 = float(np.max(probs))
    if probs.size > 1:
        top2 = float(np.partition(probs, -2)[-2])
        top3_sum = float(np.partition(probs, -3)[-3:].sum())
    else:
        top2 = 0.0
        top3_sum = top1

    entropy = _probability_entropy(probs)
    foreground_ratio = float((normalized_image > REJECTOR_INK_THRESHOLD).mean())
    return np.array([top1, top2, top1 - top2, top3_sum, entropy, foreground_ratio], dtype=np.float64)


def _build_random_negative_canvas(rng: np.random.Generator) -> np.ndarray:
    canvas = Image.new("L", (280, 280), 255)
    draw = ImageDraw.Draw(canvas)

    for _ in range(int(rng.integers(2, 6))):
        choice = int(rng.integers(0, 4))
        width = int(rng.integers(5, 22))
        x1 = int(rng.integers(10, 270))
        y1 = int(rng.integers(10, 270))
        x2 = int(rng.integers(10, 270))
        y2 = int(rng.integers(10, 270))
        if choice == 0:
            draw.line((x1, y1, x2, y2), fill=0, width=width)
        elif choice == 1:
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            draw.rectangle((left, top, right, bottom), outline=0, width=width)
        elif choice == 2:
            radius = int(rng.integers(8, 40))
            draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), outline=0, width=width)
        else:
            offset = int(rng.integers(-45, 45))
            draw.arc((x1 - 45, y1 - 30, x1 + 45, y1 + 30), start=offset, end=offset + 180, fill=0, width=width)

    noise = np.asarray(canvas, dtype=np.uint8)
    random_speckles = rng.integers(0, 255, size=noise.shape, dtype=np.uint8)
    mask = rng.random(noise.shape) < 0.03
    noise = np.where(mask, random_speckles, noise)
    return noise


def _collect_rejector_examples(
    digit_classifier,
    positive_count: int,
    negative_count: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_mnist_dataset(DATASET_PATH)
    x_train = normalize_images(dataset.x_train)
    y_train = dataset.y_train.astype(np.int64)
    x_train, y_train = select_subset(x_train, y_train, positive_count)

    positive_features: list[np.ndarray] = []
    positive_labels: list[int] = []
    for image in x_train:
        flattened = image.reshape(1, -1)
        probabilities = digit_classifier.predict_proba(flattened)[0]
        positive_features.append(build_rejector_features(probabilities, image))
        positive_labels.append(1)

    negative_features: list[np.ndarray] = []
    negative_labels: list[int] = []
    rng = np.random.default_rng(random_state)
    attempts = 0
    while len(negative_features) < negative_count and attempts < negative_count * 12:
        attempts += 1
        canvas = _build_random_negative_canvas(rng)
        preprocess_result = prepare_canvas_image(canvas)
        if preprocess_result.is_blank or preprocess_result.normalized_image is None:
            continue
        flattened = preprocess_result.normalized_image.reshape(1, -1)
        probabilities = digit_classifier.predict_proba(flattened)[0]
        negative_features.append(build_rejector_features(probabilities, preprocess_result.normalized_image))
        negative_labels.append(0)

    if not negative_features:
        raise RuntimeError("Could not generate rejector negative examples.")

    features = np.vstack([np.vstack(positive_features), np.vstack(negative_features)])
    labels = np.array(positive_labels + negative_labels, dtype=np.int64)
    return features, labels


def train_and_save_rejector(
    classifier_path: str | Path = MODEL_PATH,
    model_path: str | Path = REJECTOR_MODEL_PATH,
    metrics_path: str | Path = REJECTOR_METRICS_PATH,
    positive_count: int = 80,
    negative_count: int = 80,
    random_state: int = 42,
) -> RejectorArtifacts:
    """Train the binary rejector and persist it to disk."""

    classifier = joblib.load(classifier_path)
    features, labels = _collect_rejector_examples(
        classifier,
        positive_count=positive_count,
        negative_count=negative_count,
        random_state=random_state,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )

    rejector = build_rejector()
    rejector.fit(x_train, y_train)
    predictions = rejector.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    model_output_path = Path(model_path)
    metrics_output_path = Path(metrics_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(rejector, model_output_path)

    artifacts = RejectorArtifacts(
        model_path=str(model_output_path),
        metrics_path=str(metrics_output_path),
        accuracy=accuracy,
        num_positive_examples=int((labels == 1).sum()),
        num_negative_examples=int((labels == 0).sum()),
        model_name="digit_rejector",
    )

    with metrics_output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(asdict(artifacts), metrics_file, indent=2)

    return artifacts


def load_rejector(model_path: str | Path = REJECTOR_MODEL_PATH):
    """Load a persisted rejector model."""

    return joblib.load(model_path)


if __name__ == "__main__":
    artifacts = train_and_save_rejector()
    print(json.dumps(asdict(artifacts), indent=2))