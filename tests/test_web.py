from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from digit_app.web import create_app, decode_data_url_image
from digit_app.web import serialize_prediction
from digit_app.inference import PredictionResult


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def test_decode_data_url_image_round_trip():
    image = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
    data_url = image_to_data_url(image)

    decoded = decode_data_url_image(data_url)

    assert decoded.size == (10, 10)
    assert decoded.mode == "RGBA"


def test_serialize_prediction_shape():
    result = PredictionResult(
        digit=8,
        confidence=0.84,
        probabilities={str(index): 0.1 for index in range(10)},
        status="Prediction ready.",
        processed_image=[[0] * 28 for _ in range(28)],
        active_region_id="region-1",
        regions=[],
    )

    payload = serialize_prediction(result)

    assert payload["digit"] == 8
    assert payload["confidence"] == 0.84
    assert len(payload["processed_preview"]) == 28
    assert payload["active_region_id"] == "region-1"
    assert payload["regions"] == []
    assert len(payload["probabilities"]) == 10


def test_index_route_renders_dashboard(predictor):
    client = TestClient(create_app(predictor=predictor))

    response = client.get("/")

    assert response.status_code == 200
    assert "Handwritten Digit Recognition" in response.text
    assert "The App" in response.text
    assert "About" in response.text
    assert "Understanding the Core" in response.text
    assert "Features" in response.text
    assert "Probability Breakdown" in response.text


def test_robots_route_serves_crawler_directives(predictor):
    client = TestClient(create_app(predictor=predictor))

    response = client.get("/robots.txt")

    assert response.status_code == 200
    assert response.text == "User-agent: *\nAllow: /\n"


def test_predict_endpoint_returns_prediction(mnist_dataset, digit_service):
    client = TestClient(create_app(predictor=digit_service))
    image = Image.fromarray(mnist_dataset.x_test[0]).convert("RGBA")

    response = client.post("/api/predict", json={"image": image_to_data_url(image)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["digit"] is not None
    assert payload["confidence"] is not None
    assert len(payload["processed_preview"]) == 28
    assert payload["active_region_id"] is not None
    assert len(payload["regions"]) == 1
    assert len(payload["regions"][0]["processed_preview"]) == 28
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-5


def test_predict_endpoint_accepts_selection_hint(digit_service):
    client = TestClient(create_app(predictor=digit_service))
    image = Image.new("RGBA", (280, 280), (255, 255, 255, 255))
    for x_start in (40, 190):
        for x_coord in range(x_start, x_start + 40):
            for y_coord in range(40, 240):
                image.putpixel((x_coord, y_coord), (0, 0, 0, 255))

    response = client.post(
        "/api/predict",
        json={
            "image": image_to_data_url(image),
            "selection_hint_bbox": {"x": 0.65, "y": 0.1, "width": 0.2, "height": 0.8},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["regions"]) == 2
    assert payload["active_region_id"] == payload["regions"][1]["id"]
    assert all(len(region["processed_preview"]) == 28 for region in payload["regions"])


def test_predict_endpoint_rejects_bad_payload(digit_service):
    client = TestClient(create_app(predictor=digit_service))

    response = client.post("/api/predict", json={"image": "bad-payload"})

    assert response.status_code == 400
