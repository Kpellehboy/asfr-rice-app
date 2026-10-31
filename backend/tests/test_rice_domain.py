
import io

import numpy as np
import pytest
from PIL import Image

from app.screening.rice_domain import (
    RiceDomainClassifier,
    RiceDomainError,
)


class MockDomainModel:
    """Simulates the output of a trained binary classifier."""

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, image, verbose=0):
        return np.array([[self.prediction]], dtype=np.float32)


def create_test_image(
    image_format="JPEG",
    size=(224, 224),
    color=(100, 150, 100),
):
    """Generate valid image bytes for testing."""

    buffer = io.BytesIO()

    image = Image.new("RGB", size, color=color)
    image.save(buffer, format=image_format)

    return buffer.getvalue()


def create_classifier(prediction):
    classifier = RiceDomainClassifier(
        model_path="mock_domain_model.keras",
        threshold=0.5,
        image_size=224,
        normalization="resnet50v2",
    )

    classifier._model = MockDomainModel(prediction)

    return classifier


def test_preprocessing_output_shape():
    image_bytes = create_test_image()

    classifier = create_classifier(0.95)

    result = classifier.preprocess(image_bytes)

    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.float32


def test_resnet_normalization_range():
    image_bytes = create_test_image(
        color=(255, 255, 255)
    )

    classifier = create_classifier(0.95)

    result = classifier.preprocess(image_bytes)

    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_rice_leaf_is_accepted():
    classifier = create_classifier(0.95)

    result = classifier.predict(create_test_image())

    assert result.is_rice_leaf is True
    assert result.status == "ACCEPTED"
    assert result.confidence == pytest.approx(0.95, abs=0.0001)


def test_non_rice_image_is_rejected():
    classifier = create_classifier(0.10)

    result = classifier.predict(create_test_image())

    assert result.is_rice_leaf is False
    assert result.status == "UNSUPPORTED_IMAGE"
    assert result.confidence == pytest.approx(0.10, abs=0.0001)


def test_threshold_boundary_is_accepted():
    classifier = create_classifier(0.50)

    result = classifier.predict(create_test_image())

    assert result.is_rice_leaf is True
    assert result.status == "ACCEPTED"


def test_invalid_model_output_is_rejected():
    classifier = create_classifier(1.5)

    with pytest.raises(RiceDomainError) as error:
        classifier.predict(create_test_image())

    assert error.value.code == "INVALID_DOMAIN_CONFIDENCE"


def test_missing_model_raises_error():
    classifier = RiceDomainClassifier(
        model_path="missing_domain_model.keras"
    )

    with pytest.raises(RiceDomainError) as error:
        classifier.load_model()

    assert error.value.code == "DOMAIN_MODEL_NOT_FOUND"


def test_corrupt_image_is_rejected():
    classifier = create_classifier(0.95)

    with pytest.raises(RiceDomainError) as error:
        classifier.predict(b"not-a-real-image")

    assert error.value.code == "IMAGE_DECODE_FAILED"
