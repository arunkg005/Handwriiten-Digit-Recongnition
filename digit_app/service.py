"""Composition service that applies the rejector on top of the digit classifier."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import MODEL_PATH, METRICS_PATH, REJECTOR_MODEL_PATH, REJECTOR_METRICS_PATH
from .inference import DigitPredictor, PredictionResult, PredictionRegion
from .rejector import (
    REJECTOR_THRESHOLD,
    build_rejector_features,
    load_rejector,
    train_and_save_rejector,
)

GEOMETRY_MAX_BBOX_AREA_RATIO = 0.45
GEOMETRY_MAX_FOREGROUND_RATIO = 0.30
GEOMETRY_MIN_FOREGROUND_RATIO = 0.03
GEOMETRY_BBOX_FOREGROUND_REJECT_RATIO = 0.10
GEOMETRY_MAX_POINT_SPAN = 9
GEOMETRY_MAX_POINT_FOREGROUND_RATIO = 0.06


class DigitRecognitionService:
    """Digit classifier plus binary rejector."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        metrics_path: str | Path = METRICS_PATH,
        rejector_model_path: str | Path = REJECTOR_MODEL_PATH,
        rejector_metrics_path: str | Path = REJECTOR_METRICS_PATH,
        auto_train: bool = True,
    ) -> None:
        self.predictor = DigitPredictor(model_path=model_path, metrics_path=metrics_path, auto_train=auto_train)
        self.rejector_model_path = Path(rejector_model_path)
        self.rejector_metrics_path = Path(rejector_metrics_path)
        self.auto_train = auto_train
        self._rejector: Any = None
        self._rejector_mtime: float | None = None

    def _ensure_rejector_loaded(self) -> None:
        """Load or train the rejector model on demand."""

        model_mtime = self.rejector_model_path.stat().st_mtime if self.rejector_model_path.exists() else None
        if self._rejector is not None and self._rejector_mtime == model_mtime:
            return

        if self.auto_train and not self.rejector_model_path.exists():
            self.predictor._ensure_loaded()
            train_and_save_rejector(
                classifier_path=self.predictor.model_path,
                model_path=self.rejector_model_path,
                metrics_path=self.rejector_metrics_path,
            )
            model_mtime = self.rejector_model_path.stat().st_mtime if self.rejector_model_path.exists() else None

        if not self.rejector_model_path.exists():
            raise FileNotFoundError("Rejector artifact not found. Train the rejector before starting the app.")

        self._rejector = load_rejector(self.rejector_model_path)
        self._rejector_mtime = model_mtime

    def _gate_region(self, region: PredictionRegion) -> PredictionRegion:
        """Apply the rejector decision to one predicted region."""

        if region.processed_image is None:
            return region

        normalized_image = region.processed_image.astype(np.float32) / 255.0
        bbox_area_ratio = float(region.bbox["width"] * region.bbox["height"])
        foreground_ratio = float((normalized_image > 0.12).mean())
        foreground_mask = normalized_image > 0.12
        active_rows = np.where(foreground_mask.any(axis=1))[0]
        active_cols = np.where(foreground_mask.any(axis=0))[0]
        row_span = int(active_rows[-1] - active_rows[0] + 1) if active_rows.size else 0
        col_span = int(active_cols[-1] - active_cols[0] + 1) if active_cols.size else 0

        # Reject tiny strokes/dots based on original bounding box size or processed spans
        if (
            (region.bbox["width"] < 0.12 and region.bbox["height"] < 0.12)
            or (row_span <= GEOMETRY_MAX_POINT_SPAN and col_span <= GEOMETRY_MAX_POINT_SPAN and foreground_ratio <= GEOMETRY_MAX_POINT_FOREGROUND_RATIO)
        ):
            return replace(
                region,
                digit=None,
                status="Uncertain. Prediction may be unreliable.",
                is_ambiguous=True,
            )

        if (
            foreground_ratio > GEOMETRY_MAX_FOREGROUND_RATIO
            or foreground_ratio < GEOMETRY_MIN_FOREGROUND_RATIO
            or (
                bbox_area_ratio > GEOMETRY_MAX_BBOX_AREA_RATIO
                and foreground_ratio > GEOMETRY_BBOX_FOREGROUND_REJECT_RATIO
            )
        ):
            return replace(
                region,
                digit=None,
                status="Uncertain. Prediction may be unreliable.",
                is_ambiguous=True,
            )

        features = build_rejector_features(
            np.array([region.probabilities[str(digit)] for digit in range(10)], dtype=np.float64),
            normalized_image,
        )
        digit_like_probability = float(self._rejector.predict_proba(features.reshape(1, -1))[0][1])
        if digit_like_probability < REJECTOR_THRESHOLD:
            return replace(
                region,
                digit=None,
                status="Uncertain. Prediction may be unreliable.",
                is_ambiguous=True,
            )
        return replace(region, status="Prediction ready.", is_ambiguous=False)

    def predict_digit(self, image: Any, selection_hint_bbox: Mapping[str, float] | None = None) -> PredictionResult:
        """Run classifier predictions and then apply the rejector gate."""

        self.predictor._ensure_loaded()
        self._ensure_rejector_loaded()

        base_result = self.predictor.predict_digit(image, selection_hint_bbox=selection_hint_bbox)
        if base_result.digit is None or not base_result.regions:
            return base_result

        gated_regions = [self._gate_region(region) for region in base_result.regions]
        active_region = next((region for region in gated_regions if region.id == base_result.active_region_id), gated_regions[-1])

        return replace(
            base_result,
            digit=active_region.digit,
            confidence=active_region.confidence,
            status=active_region.status,
            processed_image=active_region.processed_image,
            active_region_id=active_region.id,
            regions=gated_regions,
        )