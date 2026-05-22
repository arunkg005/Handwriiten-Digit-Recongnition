from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

collect_ignore = ["test_results.txt"]

from digit_app.config import DATASET_PATH
from digit_app.data import load_mnist_dataset
from digit_app.inference import DigitPredictor
from digit_app.service import DigitRecognitionService
from digit_app.train import train_and_save
from digit_app.rejector import train_and_save_rejector


@pytest.fixture(scope="session")
def mnist_dataset():
    return load_mnist_dataset(DATASET_PATH)


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    model_path = artifact_dir / "model.joblib"
    metrics_path = artifact_dir / "metrics.json"
    confusion_path = artifact_dir / "confusion_matrix.png"
    rejector_model_path = artifact_dir / "rejector.joblib"
    rejector_metrics_path = artifact_dir / "rejector_metrics.json"

    artifacts = train_and_save(
        dataset_path=DATASET_PATH,
        model_path=model_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_path,
        max_train_samples=1500,
        max_test_samples=250,
        save_confusion_matrix=True,
    )

    train_and_save_rejector(
        classifier_path=model_path,
        model_path=rejector_model_path,
        metrics_path=rejector_metrics_path,
        positive_count=60,
        negative_count=60,
    )

    return {
        "artifacts": artifacts,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "confusion_path": confusion_path,
        "rejector_model_path": rejector_model_path,
        "rejector_metrics_path": rejector_metrics_path,
    }


@pytest.fixture()
def predictor(trained_artifacts):
    return DigitPredictor(
        model_path=trained_artifacts["model_path"],
        metrics_path=trained_artifacts["metrics_path"],
        auto_train=False,
    )


@pytest.fixture()
def digit_service(trained_artifacts):
    return DigitRecognitionService(
        model_path=trained_artifacts["model_path"],
        metrics_path=trained_artifacts["metrics_path"],
        rejector_model_path=trained_artifacts["rejector_model_path"],
        rejector_metrics_path=trained_artifacts["rejector_metrics_path"],
        auto_train=False,
    )
