from __future__ import annotations

import json

import numpy as np
import joblib

from digit_app.config import DATASET_PATH
from digit_app.data import load_mnist_dataset, normalize_images
from digit_app.rejector import build_rejector_features


def test_rejector_artifacts_round_trip(trained_artifacts):
    rejector = joblib.load(trained_artifacts["rejector_model_path"])

    with trained_artifacts["rejector_metrics_path"].open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    dataset = load_mnist_dataset(DATASET_PATH)
    image = normalize_images(dataset.x_test[:1])[0]
    probabilities = np.full(10, 0.1, dtype=np.float64)
    features = build_rejector_features(probabilities, image)
    rejector_probabilities = rejector.predict_proba(features.reshape(1, -1))

    assert rejector_probabilities.shape == (1, 2)
    assert metrics["model_name"] == "digit_rejector"
    assert metrics["accuracy"] > 0.80