
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.inference import InferenceError
from app.xai import XAIError


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_image_bytes():
    image = Image.new("RGB", (32, 32), color=(120, 160, 80))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def assert_safe_error(response, expected_detail):
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == expected_detail
    assert body["request_id"]
    assert "simulated" not in response.text


def test_predict_inference_failure_returns_controlled_error(
    client, valid_image_bytes, monkeypatch
):
    def fail_prediction(*args, **kwargs):
        raise InferenceError("simulated inference failure")

    monkeypatch.setattr(
        client.app.state.inference_engine,
        "predict",
        fail_prediction,
    )

    response = client.post(
        "/predict",
        files={"file": ("leaf.png", valid_image_bytes, "image/png")},
    )

    assert_safe_error(response, "Prediction failed for this image.")


def test_explain_xai_failure_returns_controlled_error(
    client, valid_image_bytes, monkeypatch
):
    def fail_explanation(*args, **kwargs):
        raise XAIError("simulated XAI failure")

    monkeypatch.setattr(
        client.app.state.xai_explainer,
        "explain",
        fail_explanation,
    )

    response = client.post(
        "/explain",
        files={"file": ("leaf.png", valid_image_bytes, "image/png")},
    )

    assert_safe_error(
        response,
        "Grad-CAM could not be generated for this image.",
    )


def test_analyze_xai_failure_returns_controlled_error(
    client, valid_image_bytes, monkeypatch
):
    def fail_explanation(*args, **kwargs):
        raise XAIError("simulated analyze XAI failure")

    monkeypatch.setattr(
        client.app.state.xai_explainer,
        "explain",
        fail_explanation,
    )

    response = client.post(
        "/analyze",
        files={"file": ("leaf.png", valid_image_bytes, "image/png")},
    )

    assert_safe_error(
        response,
        "Grad-CAM could not be generated for this image.",
    )


def test_api_remains_available_after_simulated_failure(
    client, valid_image_bytes, monkeypatch
):
    def fail_prediction(*args, **kwargs):
        raise InferenceError("simulated failure")

    monkeypatch.setattr(
        client.app.state.inference_engine,
        "predict",
        fail_prediction,
    )

    failed = client.post(
        "/predict",
        files={"file": ("leaf.png", valid_image_bytes, "image/png")},
    )
    assert_safe_error(failed, "Prediction failed for this image.")

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
