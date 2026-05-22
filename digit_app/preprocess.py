"""Shared preprocessing for live canvas input and helper image transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .data import flatten_images, normalize_images

BLANK_THRESHOLD = 0.22
MIN_FOREGROUND_PIXELS = 24
TARGET_BOX_SIZE = 20
TARGET_CANVAS_SIZE = 28
COMPONENT_PADDING = 6
MORPH_KERNEL_RADIUS = 1


@dataclass(slots=True)
class NormalizedBBox:
    """Normalized bounding box for a detected canvas region."""

    x: float
    y: float
    width: float
    height: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(slots=True)
class ComponentBounds:
    """Pixel bounds for a connected foreground component."""

    top: int
    bottom: int
    left: int
    right: int
    pixel_count: int


@dataclass(slots=True)
class RegionPreprocessResult:
    """Preprocessed single-region result derived from a full canvas."""

    id: str
    normalized_image: np.ndarray
    flattened: np.ndarray
    processed_preview: np.ndarray
    normalized_bbox: NormalizedBBox


@dataclass(slots=True)
class PreprocessResult:
    """Result of converting freehand input into MNIST-like model features."""

    normalized_image: np.ndarray | None
    flattened: np.ndarray | None
    processed_preview: np.ndarray | None
    status: str
    is_blank: bool
    inverted: bool
    active_region_id: str | None
    normalized_bbox: NormalizedBBox | None
    regions: list[RegionPreprocessResult]


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert arbitrary numeric arrays into a uint8 image."""

    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr

    converted = arr.astype(np.float32)
    if converted.max(initial=0.0) <= 1.0:
        converted *= 255.0
    return np.clip(converted, 0.0, 255.0).astype(np.uint8)


def _unwrap_editor_payload(image: Any) -> Any:
    """Extract the composite image from a Gradio image editor payload."""

    if not isinstance(image, Mapping):
        return image

    if image.get("composite") is not None:
        return image["composite"]
    if image.get("background") is not None:
        return image["background"]

    layers = image.get("layers") or []
    if layers:
        return layers[-1]

    raise ValueError("The editor payload did not contain a drawable image.")


def _to_grayscale_array(image: Image.Image | np.ndarray | Any) -> np.ndarray:
    """Coerce PIL or numpy image inputs into a 2D grayscale uint8 array."""

    if image is None:
        raise ValueError("No image was provided for preprocessing.")

    image = _unwrap_editor_payload(image)

    if isinstance(image, Image.Image):
        pil_image = image
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            pil_image = Image.fromarray(_to_uint8(array), mode="L")
        elif array.ndim == 3 and array.shape[2] == 4:
            pil_image = Image.fromarray(_to_uint8(array), mode="RGBA")
        elif array.ndim == 3 and array.shape[2] == 3:
            pil_image = Image.fromarray(_to_uint8(array), mode="RGB")
        else:
            raise ValueError(f"Unsupported image shape for preprocessing: {array.shape}")

    if pil_image.mode == "RGBA":
        background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
        pil_image = Image.alpha_composite(background, pil_image).convert("L")
    else:
        pil_image = pil_image.convert("L")

    return np.asarray(pil_image, dtype=np.uint8)


def _needs_inversion(normalized_image: np.ndarray) -> bool:
    """Detect whether the background is bright and should be inverted."""

    top = normalized_image[0, :]
    bottom = normalized_image[-1, :]
    left = normalized_image[:, 0]
    right = normalized_image[:, -1]
    border_mean = np.concatenate([top, bottom, left, right]).mean()
    return bool(border_mean > 0.5)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask with a square kernel."""

    if radius <= 0:
      return mask.copy()

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    windows = []
    size = radius * 2 + 1
    for row_offset in range(size):
        for col_offset in range(size):
            windows.append(
                padded[
                    row_offset : row_offset + mask.shape[0],
                    col_offset : col_offset + mask.shape[1],
                ]
            )
    return np.logical_or.reduce(windows)


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a boolean mask with a square kernel."""

    if radius <= 0:
      return mask.copy()

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    windows = []
    size = radius * 2 + 1
    for row_offset in range(size):
        for col_offset in range(size):
            windows.append(
                padded[
                    row_offset : row_offset + mask.shape[0],
                    col_offset : col_offset + mask.shape[1],
                ]
            )
    return np.logical_and.reduce(windows)


def _close_and_dilate(mask: np.ndarray) -> np.ndarray:
    """Smooth small gaps so multi-stroke digits stay merged as one component."""

    dilated = _binary_dilate(mask, MORPH_KERNEL_RADIUS)
    closed = _binary_erode(dilated, MORPH_KERNEL_RADIUS)
    return _binary_dilate(closed, MORPH_KERNEL_RADIUS)


def _find_component_bounds(mask: np.ndarray) -> list[ComponentBounds]:
    """Find connected foreground components using 8-connected labeling."""

    if not mask.any():
        return []

    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[ComponentBounds] = []
    neighbor_offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            top = bottom = row
            left = right = col
            pixel_count = 0

            while stack:
                current_row, current_col = stack.pop()
                pixel_count += 1
                top = min(top, current_row)
                bottom = max(bottom, current_row)
                left = min(left, current_col)
                right = max(right, current_col)

                for row_offset, col_offset in neighbor_offsets:
                    next_row = current_row + row_offset
                    next_col = current_col + col_offset
                    if (
                        next_row < 0
                        or next_row >= height
                        or next_col < 0
                        or next_col >= width
                        or visited[next_row, next_col]
                        or not mask[next_row, next_col]
                    ):
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            if pixel_count >= MIN_FOREGROUND_PIXELS:
                components.append(
                    ComponentBounds(
                        top=top,
                        bottom=bottom,
                        left=left,
                        right=right,
                        pixel_count=pixel_count,
                    )
                )

    return sorted(components, key=lambda bounds: (bounds.top, bounds.left, bounds.bottom, bounds.right))


def _expand_bounds(bounds: ComponentBounds, image_shape: tuple[int, int]) -> ComponentBounds:
    """Pad bounds slightly so crop edges do not clip pen strokes."""

    height, width = image_shape
    return ComponentBounds(
        top=max(0, bounds.top - COMPONENT_PADDING),
        bottom=min(height - 1, bounds.bottom + COMPONENT_PADDING),
        left=max(0, bounds.left - COMPONENT_PADDING),
        right=min(width - 1, bounds.right + COMPONENT_PADDING),
        pixel_count=bounds.pixel_count,
    )


def _resize_preserving_aspect(image: np.ndarray) -> np.ndarray:
    """Resize the longest side of the cropped digit to the target box size."""

    height, width = image.shape
    scale = TARGET_BOX_SIZE / max(height, width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    pil_image = Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    resized = pil_image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _center_on_canvas(image: np.ndarray) -> np.ndarray:
    """Pad a resized digit into a centered 28x28 MNIST-style canvas."""

    canvas = np.zeros((TARGET_CANVAS_SIZE, TARGET_CANVAS_SIZE), dtype=np.float32)
    height, width = image.shape
    top = (TARGET_CANVAS_SIZE - height) // 2
    left = (TARGET_CANVAS_SIZE - width) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def _shift_with_zeros(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """Shift a 2D image on a zero background without wrapping values."""

    shifted = np.zeros_like(image)

    source_y_start = max(0, -shift_y)
    source_y_end = image.shape[0] - max(0, shift_y)
    source_x_start = max(0, -shift_x)
    source_x_end = image.shape[1] - max(0, shift_x)

    target_y_start = max(0, shift_y)
    target_y_end = target_y_start + (source_y_end - source_y_start)
    target_x_start = max(0, shift_x)
    target_x_end = target_x_start + (source_x_end - source_x_start)

    shifted[target_y_start:target_y_end, target_x_start:target_x_end] = image[
        source_y_start:source_y_end,
        source_x_start:source_x_end,
    ]
    return shifted


def _center_by_mass(image: np.ndarray) -> np.ndarray:
    """Shift the digit toward the visual center using its intensity centroid."""

    total_mass = float(image.sum())
    if total_mass <= 0.0:
        return image

    rows, cols = np.indices(image.shape)
    center_y = float((rows * image).sum() / total_mass)
    center_x = float((cols * image).sum() / total_mass)
    desired_center = (TARGET_CANVAS_SIZE - 1) / 2.0
    shift_y = int(round(desired_center - center_y))
    shift_x = int(round(desired_center - center_x))
    return _shift_with_zeros(image, shift_y, shift_x)


def _build_region_result(
    region_id: str,
    working: np.ndarray,
    bounds: ComponentBounds,
    image_shape: tuple[int, int],
) -> RegionPreprocessResult:
    """Prepare one detected component for downstream classification."""

    expanded = _expand_bounds(bounds, image_shape)
    cropped = working[expanded.top : expanded.bottom + 1, expanded.left : expanded.right + 1]
    resized = _resize_preserving_aspect(cropped)
    centered = _center_on_canvas(resized)
    centered = _center_by_mass(centered)

    if centered.max(initial=0.0) > 0.0:
        centered /= centered.max()

    preview = np.clip(centered * 255.0, 0.0, 255.0).astype(np.uint8)
    flattened = flatten_images(centered)[0]
    height, width = image_shape

    return RegionPreprocessResult(
        id=region_id,
        normalized_image=centered,
        flattened=flattened,
        processed_preview=preview,
        normalized_bbox=NormalizedBBox(
            x=expanded.left / width,
            y=expanded.top / height,
            width=(expanded.right - expanded.left + 1) / width,
            height=(expanded.bottom - expanded.top + 1) / height,
        ),
    )


def _selection_hint_center(selection_hint_bbox: Mapping[str, float]) -> tuple[float, float]:
    """Return the center of a normalized selection hint."""

    return (
        float(selection_hint_bbox["x"]) + float(selection_hint_bbox["width"]) / 2.0,
        float(selection_hint_bbox["y"]) + float(selection_hint_bbox["height"]) / 2.0,
    )


def _region_center(region: RegionPreprocessResult) -> tuple[float, float]:
    """Return the center point of a normalized region box."""

    return (
        region.normalized_bbox.x + region.normalized_bbox.width / 2.0,
        region.normalized_bbox.y + region.normalized_bbox.height / 2.0,
    )


def _intersection_area(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Compute normalized box overlap area."""

    left = max(float(first["x"]), float(second["x"]))
    top = max(float(first["y"]), float(second["y"]))
    right = min(float(first["x"]) + float(first["width"]), float(second["x"]) + float(second["width"]))
    bottom = min(float(first["y"]) + float(first["height"]), float(second["y"]) + float(second["height"]))
    return max(0.0, right - left) * max(0.0, bottom - top)


def _pick_active_region(
    regions: list[RegionPreprocessResult],
    selection_hint_bbox: Mapping[str, float] | None,
) -> RegionPreprocessResult:
    """Choose the active region using overlap-first selection semantics."""

    if selection_hint_bbox is None:
        return regions[-1]

    hint_box = {
        "x": float(selection_hint_bbox["x"]),
        "y": float(selection_hint_bbox["y"]),
        "width": float(selection_hint_bbox["width"]),
        "height": float(selection_hint_bbox["height"]),
    }
    hint_center_x, hint_center_y = _selection_hint_center(hint_box)

    def score(region: RegionPreprocessResult) -> tuple[float, float, float, float]:
        region_box = region.normalized_bbox.as_dict()
        overlap = _intersection_area(region_box, hint_box)
        center_x, center_y = _region_center(region)
        distance = ((center_x - hint_center_x) ** 2 + (center_y - hint_center_y) ** 2) ** 0.5
        return (overlap, -distance, region.normalized_bbox.y, region.normalized_bbox.x)

    return max(regions, key=score)


def _coerce_selection_hint_bbox(selection_hint_bbox: Mapping[str, float] | None) -> dict[str, float] | None:
    """Normalize optional frontend selection hint payload."""

    if selection_hint_bbox is None:
        return None

    try:
        hint = {
            "x": float(selection_hint_bbox["x"]),
            "y": float(selection_hint_bbox["y"]),
            "width": float(selection_hint_bbox["width"]),
            "height": float(selection_hint_bbox["height"]),
        }
    except (KeyError, TypeError, ValueError):
        return None

    if hint["width"] <= 0.0 or hint["height"] <= 0.0:
        return None

    hint["x"] = min(max(hint["x"], 0.0), 1.0)
    hint["y"] = min(max(hint["y"], 0.0), 1.0)
    hint["width"] = min(max(hint["width"], 0.0), 1.0 - hint["x"])
    hint["height"] = min(max(hint["height"], 0.0), 1.0 - hint["y"])
    if hint["width"] <= 0.0 or hint["height"] <= 0.0:
        return None
    return hint


def prepare_canvas_image(
    image: Image.Image | np.ndarray | Any,
    selection_hint_bbox: Mapping[str, float] | None = None,
) -> PreprocessResult:
    """Convert raw UI input into one or more normalized 28x28 feature vectors."""

    grayscale = _to_grayscale_array(image)
    normalized = normalize_images(grayscale)

    inverted = _needs_inversion(normalized)
    working = 1.0 - normalized if inverted else normalized

    raw_mask = working > BLANK_THRESHOLD
    if int(raw_mask.sum()) < MIN_FOREGROUND_PIXELS:
        return PreprocessResult(
            normalized_image=None,
            flattened=None,
            processed_preview=None,
            status="No digit detected. Draw a digit to get a prediction.",
            is_blank=True,
            inverted=inverted,
            active_region_id=None,
            normalized_bbox=None,
            regions=[],
        )

    merged_mask = _close_and_dilate(raw_mask)
    component_bounds = _find_component_bounds(merged_mask)
    regions = [
        _build_region_result(
            region_id=f"region-{index + 1}",
            working=working,
            bounds=bounds,
            image_shape=working.shape,
        )
        for index, bounds in enumerate(component_bounds)
    ]

    if not regions:
        return PreprocessResult(
            normalized_image=None,
            flattened=None,
            processed_preview=None,
            status="No digit detected. Draw a digit to get a prediction.",
            is_blank=True,
            inverted=inverted,
            active_region_id=None,
            normalized_bbox=None,
            regions=[],
        )

    active_region = _pick_active_region(regions, _coerce_selection_hint_bbox(selection_hint_bbox))
    return PreprocessResult(
        normalized_image=active_region.normalized_image,
        flattened=active_region.flattened,
        processed_preview=active_region.processed_preview,
        status="Prediction ready.",
        is_blank=False,
        inverted=inverted,
        active_region_id=active_region.id,
        normalized_bbox=active_region.normalized_bbox,
        regions=regions,
    )
