import io

import numpy as np
import pytest

from PIL import Image

from app.screening.screening_service import ScreeningService
from app.screening.image_validator import (
    ValidatedImage,
)
from app.screening.rice_domain import (
    RiceDomainResult,
    RiceDomainError,
)
from app.screening.ood_detector import (
    OODResult,
    OODDetectionError,
)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

def create_valid_image():
    buffer = io.BytesIO()

    Image.new(
        "RGB",
        (224, 224),
        color="green",
    ).save(buffer, format="JPEG")

    return buffer.getvalue()


class MockRiceClassifier:

    def __init__(
        self,
        is_rice_leaf=True,
        should_fail=False,
    ):
        self.is_rice_leaf = is_rice_leaf
        self.should_fail = should_fail

    def predict(self, image_bytes):

        if self.should_fail:
            raise RiceDomainError(
                "Simulated domain failure.",
                code="DOMAIN_MODEL_LOAD_FAILED",
            )

        return RiceDomainResult(
            is_rice_leaf=self.is_rice_leaf,
            confidence=0.95 if self.is_rice_leaf else 0.10,
            threshold=0.5,
            status=(
                "ACCEPTED"
                if self.is_rice_leaf
                else "UNSUPPORTED_IMAGE"
            ),
            message="Mock domain result.",
        )


class MockOODDetector:

    def __init__(
        self,
        is_in_distribution=True,
        should_fail=False,
    ):
        self.is_in_distribution = is_in_distribution
        self.should_fail = should_fail

    def detect(self, image_bytes):

        if self.should_fail:
            raise OODDetectionError(
                "Simulated OOD failure.",
                code="OOD_CALIBRATION_NOT_FOUND",
            )

        return OODResult(
            is_in_distribution=self.is_in_distribution,
            distance=1.0 if self.is_in_distribution else 100.0,
            threshold=10.0,
            status=(
                "IN_DISTRIBUTION"
                if self.is_in_distribution
                else "OOD_WARNING"
            ),
            message="Mock OOD result.",
        )


def create_service(
    rice_leaf=True,
    in_distribution=True,
    domain_failure=False,
    ood_failure=False,
):
    return ScreeningService(
        rice_classifier=MockRiceClassifier(
            is_rice_leaf=rice_leaf,
            should_fail=domain_failure,
        ),
        ood_detector=MockOODDetector(
            is_in_distribution=in_distribution,
            should_fail=ood_failure,
        ),
    )


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def test_valid_rice_image_is_accepted():
    service = create_service()

    result = service.screen(
        create_valid_image(),
        content_type="image/jpeg",
    )

    assert result.approved_for_inference is True
    assert result.status == "ACCEPTED"


def test_invalid_image_is_rejected():
    service = create_service()

    result = service.screen(
        b"not-an-image",
        content_type="image/jpeg",
    )

    assert result.approved_for_inference is False
    assert result.status == "INVALID_IMAGE"


def test_non_rice_image_is_rejected():
    service = create_service(
        rice_leaf=False,
    )

    result = service.screen(
        create_valid_image(),
        content_type="image/jpeg",
    )

    assert result.approved_for_inference is False
    assert result.status == "UNSUPPORTED_IMAGE"
    assert result.ood_result is None


def test_ood_image_is_rejected():
    service = create_service(
        rice_leaf=True,
        in_distribution=False,
    )

    result = service.screen(
        create_valid_image(),
        content_type="image/jpeg",
    )

    assert result.approved_for_inference is False
    assert result.status == "OOD_WARNING"


def test_domain_model_failure_blocks_inference():
    service = create_service(
        domain_failure=True,
    )

    result = service.screen(
        create_valid_image(),
        content_type="image/jpeg",
    )

    assert result.approved_for_inference is False
    assert result.status == "SCREENING_UNAVAILABLE"
    assert result.error_code == "DOMAIN_MODEL_LOAD_FAILED"


def test_ood_failure_blocks_inference():
    service = create_service(
        ood_failure=True,
    )

    result = service.screen(
        create_valid_image(),
        content_type="image/jpeg",
    )

    assert result.approved_for_inference is False
    assert result.status == "SCREENING_UNAVAILABLE"
    assert result.error_code == "OOD_CALIBRATION_NOT_FOUND"


def test_screening_preserves_image_metadata():
    service = create_service()

    result = service.screen(
        create_valid_image(),
        content_type="image/jpeg",
    )

    assert isinstance(
        result.image_metadata,
        ValidatedImage,
    )

    assert result.image_metadata.width == 224
    assert result.image_metadata.height == 224


def test_screening_stops_after_domain_rejection():
    service = create_service(
        rice_leaf=False,
    )

    result = service.screen(
        create_valid_image(),
        content_type="image/jpeg",
    )

    assert result.status == "UNSUPPORTED_IMAGE"
    assert result.ood_result is None


def test_screening_stops_after_invalid_image():
    service = create_service()

    result = service.screen(
        b"invalid",
        content_type="image/jpeg",
    )

    assert result.status == "INVALID_IMAGE"
    assert result.rice_domain is None
    assert result.ood_result is None


def test_all_screening_stages_return_results():
    service = create_service()

    result = service.screen(
        create_valid_image(),
        content_type="image/jpeg",
    )

    assert result.approved_for_inference is True
    assert result.image_metadata is not None
    assert result.rice_domain is not None
    assert result.ood_result is not None