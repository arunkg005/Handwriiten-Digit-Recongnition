"""Plain HTML/CSS/JS web app for handwritten digit recognition."""

from __future__ import annotations

import base64
import io
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel
from PIL import Image

from .config import WEB_DIR, WEB_STATIC_DIR
from .inference import PredictionResult
from .service import DigitRecognitionService


class SelectionHintBBox(BaseModel):
    """Normalized box hint pointing at the region the user most recently touched."""

    x: float
    y: float
    width: float
    height: float


class PredictRequest(BaseModel):
    """JSON payload sent by the browser canvas."""

    image: str
    selection_hint_bbox: SelectionHintBBox | None = None


def decode_data_url_image(data_url: str) -> Image.Image:
    """Decode a browser canvas data URL into a PIL image."""

    if not data_url or "," not in data_url:
        raise ValueError("Invalid image payload.")

    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Image payload must be base64 encoded.")

    try:
        raw_bytes = base64.b64decode(encoded)
    except ValueError as exc:
        raise ValueError("Could not decode the uploaded image.") from exc

    return Image.open(io.BytesIO(raw_bytes)).convert("RGBA")


def serialize_prediction(result: PredictionResult) -> dict[str, Any]:
    """Convert a prediction result into a browser-friendly response."""

    processed_preview = (
        np.asarray(result.processed_image, dtype=int).tolist()
        if result.processed_image is not None
        else None
    )

    return {
        "digit": result.digit,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "status": result.status,
        "processed_preview": processed_preview,
        "active_region_id": result.active_region_id,
        "regions": [
            {
                "id": region.id,
                "bbox": region.bbox,
                "digit": region.digit,
                "confidence": region.confidence,
                "probabilities": region.probabilities,
                "status": region.status,
                "is_ambiguous": region.is_ambiguous,
                "processed_preview": (
                    np.asarray(region.processed_image, dtype=int).tolist()
                    if region.processed_image is not None
                    else None
                ),
            }
            for region in result.regions
        ],
    }


def create_app(predictor: DigitRecognitionService | None = None) -> FastAPI:
    """Create the FastAPI application and static asset routes."""

    predictor = predictor or DigitRecognitionService()
    app = FastAPI(title="Handwritten Digit Recognition")

    app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")

    @app.get("/api/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/predict")
    def predict(payload: PredictRequest) -> dict[str, Any]:
        try:
            image = decode_data_url_image(payload.image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        selection_hint_bbox = (
            payload.selection_hint_bbox.model_dump()
            if payload.selection_hint_bbox is not None
            else None
        )
        result = predictor.predict_digit(image, selection_hint_bbox=selection_hint_bbox)
        return serialize_prediction(result)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
