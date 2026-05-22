"""Shared configuration and filesystem paths for the MVP."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
WEB_STATIC_DIR = WEB_DIR / "static"
DATASET_PATH = ROOT_DIR / "mnist.npz"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = ARTIFACT_DIR / "confusion_matrix.png"
REJECTOR_MODEL_PATH = ARTIFACT_DIR / "rejector.joblib"
REJECTOR_METRICS_PATH = ARTIFACT_DIR / "rejector_metrics.json"
