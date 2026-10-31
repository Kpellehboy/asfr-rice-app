
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


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


def post_image(client, endpoint, image_bytes):
    return client.post(
        endpoint,
        files={"file": ("leaf.png", image_bytes, "image/png")},
    )


def assert_timeout_response(response):
    assert response.status_code == 504
    body = response.json()
    assert body["detail"] == "Model operation timed out. Please retry."
    assert body["request_id"]


def test_predict_timeout_returns_504_and_api_recovers(
    client, valid_image_bytes, monkeypatch
):
    monkeypatch.setattr(
        "app.main.MODEL_OPERATION_TIMEOUT_SECONDS", 0.05
    )

    engine = client.app.state.inference_engine
    original_predict = engine.predict

    def slow_predict(*args, **kwargs):
        time.sleep(0.25)
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(engine, "predict", slow_predict)

    response = post_image(client, "/predict", valid_image_bytes)
    assert_timeout_response(response)

    time.sleep(0.35)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_explain_timeout_returns_504(
    client, valid_image_bytes, monkeypatch
):
    monkeypatch.setattr(
        "app.main.MODEL_OPERATION_TIMEOUT_SECONDS", 0.05
    )

    explainer = client.app.state.xai_explainer
    original_explain = explainer.explain

    def slow_explain(*args, **kwargs):
        time.sleep(0.25)
        return original_explain(*args, **kwargs)

    monkeypatch.setattr(explainer, "explain", slow_explain)

    response = post_image(client, "/explain", valid_image_bytes)
    assert_timeout_response(response)

    time.sleep(0.35)


def test_analyze_uses_one_overall_deadline(
    client, valid_image_bytes, monkeypatch
):
    monkeypatch.setattr(
        "app.main.MODEL_OPERATION_TIMEOUT_SECONDS", 0.05
    )

    engine = client.app.state.inference_engine
    original_predict = engine.predict

    def slow_predict(*args, **kwargs):
        time.sleep(0.25)
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(engine, "predict", slow_predict)

    response = post_image(client, "/analyze", valid_image_bytes)
    assert_timeout_response(response)

    time.sleep(0.35)
