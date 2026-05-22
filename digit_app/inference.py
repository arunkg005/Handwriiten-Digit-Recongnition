"""Prediction service boundary for the digit recognition MVP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from .config import METRICS_PATH, MODEL_PATH
from .preprocess import NormalizedBBox, prepare_canvas_image
from .train import train_and_save


@dataclass(slots=True)
class PredictionRegion:
    """Prediction metadata for one detected canvas region."""

    id: str
    bbox: dict[str, float]
    digit: int | None
    confidence: float
    probabilities: dict[str, float]
    status: str
    is_ambiguous: bool
    processed_image: np.ndarray | None


@dataclass(slots=True)
class PredictionResult:
    """Inference output that stays independent from the UI framework."""

    digit: int | None
    confidence: float | None
    probabilities: dict[str, float]
    status: str
    processed_image: np.ndarray | None
    active_region_id: str | None
    regions: list[PredictionRegion]


def empty_probability_map() -> dict[str, float]:
    """Create an ordered probability map for all digits."""

    return {str(digit): 0.0 for digit in range(10)}


class DigitPredictor:
    """Lazy-loading local prediction service backed by sklearn artifacts."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        metrics_path: str | Path = METRICS_PATH,
        auto_train: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.auto_train = auto_train
        self._model: Any = None
        self._classes: list[int] = list(range(10))
        self._model_mtime: float | None = None
        self._metrics_mtime: float | None = None

    def _ensure_loaded(self) -> None:
        """Load saved artifacts and train them on demand if needed."""

        model_mtime = self.model_path.stat().st_mtime if self.model_path.exists() else None
        metrics_mtime = self.metrics_path.stat().st_mtime if self.metrics_path.exists() else None

        if self._model is not None and self._model_mtime is None and self._metrics_mtime is None:
            return

        if (
            self._model is not None
            and self._model_mtime == model_mtime
            and self._metrics_mtime == metrics_mtime
        ):
            return

        if self.auto_train and not self.model_path.exists():
            train_and_save(model_path=self.model_path, metrics_path=self.metrics_path)
            model_mtime = self.model_path.stat().st_mtime if self.model_path.exists() else None
            metrics_mtime = self.metrics_path.stat().st_mtime if self.metrics_path.exists() else None

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Model artifact not found. Run the training pipeline before starting the app."
            )

        self._model = joblib.load(self.model_path)

        if self.metrics_path.exists():
            with self.metrics_path.open("r", encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
            self._classes = [int(label) for label in metrics.get("classes", self._classes)]
        elif hasattr(self._model, "classes_"):
            self._classes = [int(label) for label in self._model.classes_.tolist()]

        self._model_mtime = model_mtime
        self._metrics_mtime = metrics_mtime

    def _predict_region(
        self,
        flattened: np.ndarray,
        region_id: str,
        normalized_bbox: NormalizedBBox,
        processed_image: np.ndarray | None,
    ) -> PredictionRegion:
        """Predict a single segmented region."""

        probabilities = self._model.predict_proba(flattened.reshape(1, -1))[0]
        probability_map = {
            str(label): float(probability)
            for label, probability in zip(self._classes, probabilities, strict=False)
        }

        predicted_index = int(np.argmax(probabilities))
        predicted_digit = self._classes[predicted_index]
        confidence = float(probabilities[predicted_index])

        return PredictionRegion(
            id=region_id,
            bbox=normalized_bbox.as_dict(),
            digit=predicted_digit,
            confidence=confidence,
            probabilities={**empty_probability_map(), **probability_map},
            status="Prediction ready.",
            is_ambiguous=False,
            processed_image=processed_image,
        )

    def predict_digit(
        self,
        image: Any,
        selection_hint_bbox: Mapping[str, float] | None = None,
    ) -> PredictionResult:
        """Run shared preprocessing and return structured region predictions."""

        self._ensure_loaded()
        preprocess_result = prepare_canvas_image(image, selection_hint_bbox=selection_hint_bbox)
        if preprocess_result.is_blank:
            return PredictionResult(
                digit=None,
                confidence=None,
                probabilities=empty_probability_map(),
                status=preprocess_result.status,
                processed_image=None,
                active_region_id=None,
                regions=[],
            )

        regions = [
            self._predict_region(
                flattened=region.flattened,
                region_id=region.id,
                normalized_bbox=region.normalized_bbox,
                processed_image=region.processed_preview,
            )
            for region in preprocess_result.regions
        ]
        regions_by_id = {region.id: region for region in regions}
        active_region = regions_by_id.get(preprocess_result.active_region_id or "")
        if active_region is None:
            active_region = regions[-1]

        return PredictionResult(
            digit=active_region.digit,
            confidence=active_region.confidence,
            probabilities=active_region.probabilities,
            status=active_region.status,
            processed_image=active_region.processed_image,
            active_region_id=active_region.id,
            regions=regions,
        )
