import numpy as np
import pytest

from app.screening.ood_detector import (
    OODDetector,
    OODDetectionError,
)


# ---------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------

@pytest.fixture
def embeddings():
    """
    Generate deterministic synthetic feature embeddings.

    These are software-test data, not real ASFR embeddings.
    """

    rng = np.random.default_rng(42)

    reference = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(200, 8),
    )

    calibration = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(100, 8),
    )

    return reference, calibration


@pytest.fixture
def calibrated_detector(embeddings, tmp_path):
    reference, calibration = embeddings

    return OODDetector.fit(
        reference_embeddings=reference,
        calibration_embeddings=calibration,
        artifact_path=tmp_path / "ood_calibration.npz",
    )


# ---------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------

def test_detector_calibrates_successfully(embeddings, tmp_path):
    reference, calibration = embeddings

    detector = OODDetector.fit(
        reference,
        calibration,
        artifact_path=tmp_path / "calibration.npz",
    )

    assert detector.mean is not None
    assert detector.inverse_covariance is not None
    assert detector.threshold is not None
    assert detector.feature_dimension == 8
    assert detector.threshold > 0


def test_calibration_threshold_is_finite(calibrated_detector):
    assert np.isfinite(calibrated_detector.threshold)


def test_invalid_quantile_is_rejected(embeddings):
    reference, calibration = embeddings

    with pytest.raises(OODDetectionError) as error:
        OODDetector.fit(
            reference,
            calibration,
            quantile=1.0,
        )

    assert error.value.code == "INVALID_CALIBRATION_QUANTILE"


def test_mismatched_embedding_dimensions_are_rejected():
    reference = np.random.default_rng(1).normal(size=(20, 8))
    calibration = np.random.default_rng(2).normal(size=(20, 6))

    with pytest.raises(OODDetectionError) as error:
        OODDetector.fit(reference, calibration)

    assert error.value.code == "EMBEDDING_DIMENSION_MISMATCH"


def test_nonfinite_reference_embeddings_are_rejected():
    reference = np.random.default_rng(1).normal(size=(20, 4))
    calibration = np.random.default_rng(2).normal(size=(20, 4))

    reference[0, 0] = np.nan

    with pytest.raises(OODDetectionError) as error:
        OODDetector.fit(reference, calibration)

    assert error.value.code == "INVALID_EMBEDDING_VALUES"


def test_insufficient_reference_embeddings_are_rejected():
    reference = np.array([[0.1, 0.2]])
    calibration = np.random.default_rng(2).normal(size=(20, 2))

    with pytest.raises(OODDetectionError) as error:
        OODDetector.fit(reference, calibration)

    assert error.value.code == "INSUFFICIENT_EMBEDDINGS"


# ---------------------------------------------------------
# Distance and classification tests
# ---------------------------------------------------------

def test_mahalanobis_distance_is_nonnegative(calibrated_detector):
    embedding = np.zeros(8)

    distance = calibrated_detector.score_embedding(embedding)

    assert distance >= 0
    assert np.isfinite(distance)


def test_known_distribution_embedding_is_accepted(
    calibrated_detector,
    embeddings,
):
    _, calibration = embeddings

    # A calibration sample should be within the calibrated range.
    result = calibrated_detector.detect_embedding(calibration[0])

    assert result.is_in_distribution is True
    assert result.status == "IN_DISTRIBUTION"
    assert result.distance <= result.threshold


def test_far_embedding_is_rejected(calibrated_detector):
    # Deliberately distant synthetic embedding.
    unknown_embedding = np.full(8, 100.0)

    result = calibrated_detector.detect_embedding(
        unknown_embedding
    )

    assert result.is_in_distribution is False
    assert result.status == "OOD_WARNING"
    assert result.distance > result.threshold


def test_invalid_embedding_dimension_is_rejected(
    calibrated_detector,
):
    invalid_embedding = np.zeros(5)

    with pytest.raises(OODDetectionError) as error:
        calibrated_detector.detect_embedding(invalid_embedding)

    assert error.value.code == "EMBEDDING_DIMENSION_MISMATCH"


def test_nonfinite_input_embedding_is_rejected(
    calibrated_detector,
):
    invalid_embedding = np.zeros(8)
    invalid_embedding[0] = np.inf

    with pytest.raises(OODDetectionError) as error:
        calibrated_detector.detect_embedding(invalid_embedding)

    assert error.value.code == "INVALID_INPUT_EMBEDDING"


# ---------------------------------------------------------
# Calibration artifact tests
# ---------------------------------------------------------

def test_missing_calibration_artifact_is_rejected(tmp_path):
    detector = OODDetector(
        artifact_path=tmp_path / "missing.npz"
    )

    with pytest.raises(OODDetectionError) as error:
        detector.load()

    assert error.value.code == "OOD_CALIBRATION_NOT_FOUND"


def test_save_and_load_calibration_artifact(
    calibrated_detector,
    tmp_path,
    embeddings,
):
    artifact_path = tmp_path / "ood_calibration.npz"

    calibrated_detector.artifact_path = artifact_path
    calibrated_detector.save()

    loaded_detector = OODDetector(
        artifact_path=artifact_path
    )

    loaded_detector.load()

    assert np.allclose(
        calibrated_detector.mean,
        loaded_detector.mean,
    )

    assert np.allclose(
        calibrated_detector.inverse_covariance,
        loaded_detector.inverse_covariance,
    )

    assert calibrated_detector.threshold == loaded_detector.threshold

    assert (
        calibrated_detector.feature_dimension
        == loaded_detector.feature_dimension
    )


def test_save_without_calibration_is_rejected(tmp_path):
    detector = OODDetector(
        artifact_path=tmp_path / "uncalibrated.npz"
    )

    with pytest.raises(OODDetectionError) as error:
        detector.save()

    assert error.value.code == "OOD_DETECTOR_NOT_READY"


# ---------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------

def test_detect_uses_configured_feature_extractor(
    calibrated_detector,
):
    calibrated_detector.feature_extractor = (
        lambda image_bytes: np.zeros(8)
    )

    result = calibrated_detector.detect(b"synthetic-image")

    assert result.status == "IN_DISTRIBUTION"


def test_detect_without_feature_extractor_is_rejected(
    calibrated_detector,
):
    with pytest.raises(OODDetectionError) as error:
        calibrated_detector.detect(b"synthetic-image")

    assert error.value.code == "FEATURE_EXTRACTOR_NOT_CONFIGURED"


def test_feature_extraction_failure_is_handled(
    calibrated_detector,
):
    def failing_extractor(image_bytes):
        raise RuntimeError("Simulated feature extraction failure")

    calibrated_detector.feature_extractor = failing_extractor

    with pytest.raises(OODDetectionError) as error:
        calibrated_detector.detect(b"synthetic-image")

    assert error.value.code == "FEATURE_EXTRACTION_FAILED"


def test_empty_image_is_rejected(calibrated_detector):
    calibrated_detector.feature_extractor = (
        lambda image_bytes: np.zeros(8)
    )

    with pytest.raises(OODDetectionError) as error:
        calibrated_detector.detect(b"")

    assert error.value.code == "EMPTY_IMAGE"