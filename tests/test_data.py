from __future__ import annotations

import json

import joblib

from digit_app.config import DATASET_PATH
from digit_app.data import flatten_images, load_mnist_dataset, normalize_images


def test_load_mnist_dataset_shapes(mnist_dataset):
    assert mnist_dataset.x_train.shape == (60000, 28, 28)
    assert mnist_dataset.y_train.shape == (60000,)
    assert mnist_dataset.x_test.shape == (10000, 28, 28)
    assert mnist_dataset.y_test.shape == (10000,)


def test_normalize_and_flatten_pipeline(mnist_dataset):
    normalized = normalize_images(mnist_dataset.x_train[:4])
    flattened = flatten_images(normalized)

    assert normalized.dtype.name == "float32"
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert flattened.shape == (4, 784)


def test_training_artifacts_round_trip(trained_artifacts):
    model = joblib.load(trained_artifacts["model_path"])

    with trained_artifacts["metrics_path"].open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    dataset = load_mnist_dataset(DATASET_PATH)
    features = flatten_images(normalize_images(dataset.x_test[:3]))
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (3, 10)
    assert metrics["model_name"] == "mlp_classifier"
    assert metrics["test_accuracy"] > 0.80
    assert trained_artifacts["confusion_path"].exists()
