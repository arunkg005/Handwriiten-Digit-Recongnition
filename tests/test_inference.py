from __future__ import annotations

import numpy as np

from digit_app.inference import PredictionRegion, PredictionResult
from digit_app.service import DigitRecognitionService


class FakePredictor:
    def __init__(self, result):
        self.result = result

    def _ensure_loaded(self):
        return None

    def predict_digit(self, image, selection_hint_bbox=None):
        return self.result


class DummyRejector:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, features):
        return np.array([[1.0 - self.probability, self.probability]], dtype=np.float64)


def test_predict_digit_returns_probability_distribution(mnist_dataset, predictor):
    sample = mnist_dataset.x_test[0]
    result = predictor.predict_digit(sample)

    assert isinstance(result, PredictionResult)
    assert result.digit is not None
    assert result.confidence is not None
    assert result.processed_image is not None
    assert result.active_region_id is not None
    assert len(result.regions) == 1
    assert len(result.probabilities) == 10
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5


def test_blank_input_returns_no_prediction(predictor):
    blank = np.zeros((280, 280, 4), dtype=np.uint8)
    result = predictor.predict_digit(blank)

    assert result.digit is None
    assert result.confidence is None
    assert result.processed_image is None
    assert result.active_region_id is None
    assert result.regions == []
    assert result.probabilities["0"] == 0.0


def test_predict_digit_returns_multiple_regions(predictor):
    canvas = np.full((280, 280), 255, dtype=np.uint8)
    canvas[40:240, 40:80] = 0
    canvas[40:240, 190:230] = 0

    result = predictor.predict_digit(canvas, selection_hint_bbox={"x": 0.65, "y": 0.1, "width": 0.2, "height": 0.8})

    assert result.active_region_id is not None
    assert len(result.regions) == 2
    assert result.regions[1].id == result.active_region_id


def test_digit_service_marks_low_confidence_prediction_uncertain():
    region = PredictionRegion(
        id="region-1",
        bbox={"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3},
        digit=4,
        confidence=0.41,
        probabilities={str(index): (0.11 if index != 4 else 0.41) for index in range(10)},
        status="Prediction ready.",
        is_ambiguous=False,
        processed_image=(lambda image: (image.__setitem__((slice(10, 14), slice(10, 18)), 255), image)[1])(
            np.zeros((28, 28), dtype=np.uint8)
        ),
    )
    result = PredictionResult(
        digit=4,
        confidence=0.41,
        probabilities=region.probabilities,
        status="Prediction ready.",
        processed_image=region.processed_image,
        active_region_id="region-1",
        regions=[region],
    )
    service = DigitRecognitionService(auto_train=False, rejector_model_path="d:/tmp/rejector.joblib")
    service.predictor = FakePredictor(result)
    service._rejector = DummyRejector(0.60)
    service._rejector_mtime = None

    gated = service.predict_digit(np.full((280, 280), 255, dtype=np.uint8))

    assert gated.digit == 4
    assert gated.status.startswith("Uncertain")
    assert gated.regions[0].is_ambiguous is True


def test_digit_service_marks_point_like_input_uncertain():
    region = PredictionRegion(
        id="region-1",
        bbox={"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.1},
        digit=4,
        confidence=0.99,
        probabilities={str(index): (0.01 if index != 4 else 0.99) for index in range(10)},
        status="Prediction ready.",
        is_ambiguous=False,
        processed_image=(lambda image: (image.__setitem__((slice(11, 16), slice(11, 16)), 255), image)[1])(
            np.zeros((28, 28), dtype=np.uint8)
        ),
    )
    result = PredictionResult(
        digit=4,
        confidence=0.99,
        probabilities=region.probabilities,
        status="Prediction ready.",
        processed_image=region.processed_image,
        active_region_id="region-1",
        regions=[region],
    )
    service = DigitRecognitionService(auto_train=False, rejector_model_path="d:/tmp/rejector.joblib")
    service.predictor = FakePredictor(result)
    service._rejector = DummyRejector(0.99)
    service._rejector_mtime = None

    gated = service.predict_digit(np.full((280, 280), 255, dtype=np.uint8))

    assert gated.digit == 4
    assert gated.status.startswith("Uncertain")
    assert gated.regions[0].is_ambiguous is True


def test_digit_service_rejects_garbage_as_not_a_number():
    region = PredictionRegion(
        id="region-1",
        bbox={"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3},
        digit=4,
        confidence=0.91,
        probabilities={str(index): (0.06 if index == 7 else 0.94 if index == 4 else 0.0) for index in range(10)},
        status="Prediction ready.",
        is_ambiguous=False,
        processed_image=np.full((28, 28), 255, dtype=np.uint8),
    )
    result = PredictionResult(
        digit=4,
        confidence=0.91,
        probabilities=region.probabilities,
        status="Prediction ready.",
        processed_image=region.processed_image,
        active_region_id="region-1",
        regions=[region],
    )
    service = DigitRecognitionService(auto_train=False, rejector_model_path="d:/tmp/rejector.joblib")
    service.predictor = FakePredictor(result)
    service._rejector = DummyRejector(0.95)
    service._rejector_mtime = None

    garbage = np.zeros((280, 280), dtype=np.uint8)
    garbage[15:260, 15:265] = 255
    garbage[30:250, 40:90] = 0
    garbage[55:70, 85:200] = 0
    garbage[140:170, 30:220] = 0

    gated = service.predict_digit(garbage)

    assert gated.digit is None
    assert gated.status.startswith("Not a number")
    assert gated.regions[0].digit is None
    assert gated.regions[0].is_ambiguous is True
