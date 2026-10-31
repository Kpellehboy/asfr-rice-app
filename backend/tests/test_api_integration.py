
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


REFERENCE_IMAGE = Path(
    r"C:\Users\Menuo\Desktop\KBK\Papers\Capstone_Research"
    r"\Dataset\archive\Rice_Leaf_AUG\Bacterial Leaf Blight"
    r"\20231006_164208.jpg"
)

EXPECTED_CLASS_INDEX = 0
EXPECTED_CLASS_NAME = "Bacterial Leaf Blight"


@pytest.fixture(scope="module")
def client():
    if not REFERENCE_IMAGE.is_file():
        pytest.fail(f"Reference image not found: {REFERENCE_IMAGE}")

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def reference_bytes():
    return REFERENCE_IMAGE.read_bytes()


def upload(client, endpoint, image_bytes, filename="reference.jpg", **params):
    return client.post(
        endpoint,
        files={"file": (filename, image_bytes, "image/jpeg")},
        params=params or None,
    )


def assert_nine_probabilities(data):
    probabilities = data["probabilities"]

    assert len(probabilities) == 9
    assert [item["class_index"] for item in probabilities] == list(range(9))
    assert all(item["class_name"] for item in probabilities)
    assert all(0.0 <= item["probability"] <= 1.0 for item in probabilities)
    assert sum(item["probability"] for item in probabilities) == pytest.approx(
        1.0, abs=1e-4
    )


def test_real_checkpoint_predicts_reference_image(client, reference_bytes):
    response = upload(client, "/predict", reference_bytes)

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["predicted_class_index"] == EXPECTED_CLASS_INDEX
    assert data["predicted_class"] == EXPECTED_CLASS_NAME
    assert data["confidence"] == pytest.approx(
        next(
            item["probability"]
            for item in data["probabilities"]
            if item["class_index"] == EXPECTED_CLASS_INDEX
        ),
        abs=1e-6,
    )
    assert_nine_probabilities(data)
    assert data["model_version"]
    assert data["request_id"]


def test_analyze_returns_prediction_gradcam_overlay_and_gate(
    client, reference_bytes
):
    response = upload(
        client, "/analyze", reference_bytes,
        target_class_index=EXPECTED_CLASS_INDEX,
    )

    assert response.status_code == 200, response.text
    data = response.json()

    prediction = data["prediction"]
    explanation = data["explanation"]
    gate = data["asfr_gate"]

    assert prediction["predicted_class_index"] == EXPECTED_CLASS_INDEX
    assert prediction["predicted_class"] == EXPECTED_CLASS_NAME
    assert_nine_probabilities(prediction)

    assert explanation["explained_class_index"] == EXPECTED_CLASS_INDEX
    assert explanation["explained_class"] == EXPECTED_CLASS_NAME
    assert explanation["target_layer"] == "asfr_fusion_relu"
    assert explanation["method"] == "Grad-CAM"

    heatmap_bytes = base64.b64decode(
        explanation["heatmap_png_base64"], validate=True
    )
    overlay_bytes = base64.b64decode(
        explanation["overlay_png_base64"], validate=True
    )
    gate_bytes = base64.b64decode(
        gate["gate_png_base64"], validate=True
    )

    with Image.open(io.BytesIO(heatmap_bytes)) as image:
        assert image.format == "PNG"
        assert image.mode in ("L", "RGB")
        assert image.size == (Image.open(io.BytesIO(reference_bytes)).width,
                              Image.open(io.BytesIO(reference_bytes)).height)

    with Image.open(io.BytesIO(overlay_bytes)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"

    with Image.open(io.BytesIO(gate_bytes)) as image:
        assert image.format == "PNG"
        assert image.mode == "L"

    assert gate["target_layer"] == "asfr_spatial_gate"
    assert data["request_id"]


def test_explain_respects_explicit_target_class(client, reference_bytes):
    target_class_index = 1

    response = upload(
        client, "/explain", reference_bytes,
        target_class_index=target_class_index,
    )

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["explained_class_index"] == target_class_index
    assert data["explained_class"] == "Brown Spot"
    assert data["target_layer"] == "asfr_fusion_relu"
    assert base64.b64decode(data["heatmap_png_base64"], validate=True)
    assert base64.b64decode(data["overlay_png_base64"], validate=True)


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("notes.txt", b"not an image", 400),
        ("broken.jpg", b"\xff\xd8broken image", 400),
        ("empty.png", b"", 400),
    ],
)
def test_invalid_uploads_are_rejected(
    client, filename, content, expected_status
):
    response = upload(client, "/predict", content, filename=filename)

    assert response.status_code == expected_status
    assert response.json().get("request_id")


def test_upload_byte_limit_is_enforced(client):
    from app.main import MAX_IMAGE_BYTES

    oversized = b"x" * (MAX_IMAGE_BYTES + 1)
    response = upload(client, "/predict", oversized, filename="large.jpg")

    assert response.status_code == 413
    assert response.json()["detail"] == "Image exceeds the 10 MiB upload limit."
    assert response.json().get("request_id")


def test_concurrent_predictions_return_valid_responses(
    client, reference_bytes
):
    def make_request(_):
        response = upload(client, "/predict", reference_bytes)
        return response.status_code, response.json()

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(make_request, range(3)))

    for status_code, data in results:
        assert status_code == 200, data
        assert data["predicted_class_index"] == EXPECTED_CLASS_INDEX
        assert data["predicted_class"] == EXPECTED_CLASS_NAME
        assert_nine_probabilities(data)
