from __future__ import annotations

import numpy as np

from digit_app.preprocess import prepare_canvas_image


def test_blank_canvas_returns_no_digit():
    blank = np.zeros((280, 280, 4), dtype=np.uint8)
    result = prepare_canvas_image(blank)

    assert result.is_blank is True
    assert result.flattened is None
    assert result.normalized_image is None
    assert "No digit detected" in result.status


def test_drawn_canvas_is_centered_and_resized():
    canvas = np.full((280, 280), 255, dtype=np.uint8)
    canvas[40:240, 120:160] = 0
    result = prepare_canvas_image(canvas)

    assert result.is_blank is False
    assert result.normalized_image is not None
    assert result.processed_preview is not None
    assert result.normalized_image.shape == (28, 28)
    assert result.flattened.shape == (784,)
    assert result.normalized_image.max() <= 1.0
    assert result.normalized_image.min() >= 0.0

    rows, cols = np.where(result.normalized_image > 0.2)
    assert rows.min() >= 0
    assert rows.max() < 28
    assert cols.min() >= 0
    assert cols.max() < 28


def test_gradio_editor_payload_is_supported():
    canvas = np.full((280, 280, 4), 255, dtype=np.uint8)
    canvas[50:230, 120:160, :3] = 0
    payload = {"background": None, "layers": [], "composite": canvas}

    result = prepare_canvas_image(payload)

    assert result.is_blank is False
    assert result.flattened is not None
    assert result.processed_preview is not None


def test_dark_background_with_light_digit_is_supported():
    canvas = np.zeros((280, 280), dtype=np.uint8)
    canvas[40:240, 120:160] = 255

    result = prepare_canvas_image(canvas)

    assert result.is_blank is False
    assert result.flattened is not None
    assert result.processed_preview is not None


def test_prepare_canvas_image_detects_multiple_regions():
    canvas = np.full((280, 280), 255, dtype=np.uint8)
    canvas[40:240, 40:80] = 0
    canvas[40:240, 190:230] = 0

    result = prepare_canvas_image(canvas, selection_hint_bbox={"x": 0.65, "y": 0.1, "width": 0.2, "height": 0.8})

    assert result.is_blank is False
    assert len(result.regions) == 2
    assert result.active_region_id == "region-2"
    assert result.normalized_bbox is not None


def test_prepare_canvas_image_merges_touching_strokes():
    canvas = np.full((280, 280), 255, dtype=np.uint8)
    canvas[60:220, 80:110] = 0
    canvas[60:220, 112:142] = 0

    result = prepare_canvas_image(canvas)

    assert result.is_blank is False
    assert len(result.regions) == 1


def test_prepare_canvas_image_ignores_tiny_noise():
    canvas = np.full((280, 280), 255, dtype=np.uint8)
    canvas[10:12, 10:12] = 0

    result = prepare_canvas_image(canvas)

    assert result.is_blank is True
    assert result.regions == []
