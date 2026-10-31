from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OOD_ARTIFACT_PATH = Path(
    "model_artifacts/ood_calibration.npz"
)

# Covariance regularization prevents numerical instability.
COVARIANCE_REGULARIZATION = 1e-5

# Initial target coverage for known rice-leaf validation data.
# This is a calibration setting, not a guaranteed OOD accuracy.
CALIBRATION_QUANTILE = 0.99


# ---------------------------------------------------------
# Result structure
# ---------------------------------------------------------

@dataclass(frozen=True)
class OODResult:
    is_in_distribution: bool
    distance: float
    threshold: float
    status: str
    message: str


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------

class OODDetectionError(Exception):
    """Raised when OOD detection cannot be completed."""

    def __init__(
        self,
        message: str,
        code: str = "OOD_DETECTION_ERROR",
    ):
        self.message = message
        self.code = code
        super().__init__(message)


# ---------------------------------------------------------
# OOD detector
# ---------------------------------------------------------

class OODDetector:
    """
    Mahalanobis-distance OOD detector.

    Reference embeddings:
        Embeddings from known rice-leaf training images.

    Calibration embeddings:
        Separate in-distribution validation embeddings used
        to establish the rejection threshold.

    Feature extractor:
        Callable accepting image bytes and returning a
        one-dimensional feature embedding.

    The detector does not train or modify the ASFR model.
    """

    def __init__(
        self,
        artifact_path: Path = OOD_ARTIFACT_PATH,
        feature_extractor: Optional[
            Callable[[bytes], np.ndarray]
        ] = None,
    ):
        self.artifact_path = Path(artifact_path)
        self.feature_extractor = feature_extractor

        self.mean: Optional[np.ndarray] = None
        self.inverse_covariance: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.feature_dimension: Optional[int] = None

    # -----------------------------------------------------
    # Input validation
    # -----------------------------------------------------

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
        name: str,
    ) -> np.ndarray:

        embeddings = np.asarray(
            embeddings,
            dtype=np.float64,
        )

        if embeddings.ndim != 2:
            raise OODDetectionError(
                f"{name} must be a two-dimensional array.",
                code="INVALID_EMBEDDING_SHAPE",
            )

        if embeddings.shape[0] < 2:
            raise OODDetectionError(
                f"{name} must contain at least two samples.",
                code="INSUFFICIENT_EMBEDDINGS",
            )

        if embeddings.shape[1] < 1:
            raise OODDetectionError(
                f"{name} has no feature dimensions.",
                code="EMPTY_EMBEDDING",
            )

        if not np.all(np.isfinite(embeddings)):
            raise OODDetectionError(
                f"{name} contains NaN or infinite values.",
                code="INVALID_EMBEDDING_VALUES",
            )

        return embeddings

    # -----------------------------------------------------
    # Calibration
    # -----------------------------------------------------

    @classmethod
    def fit(
        cls,
        reference_embeddings: np.ndarray,
        calibration_embeddings: np.ndarray,
        artifact_path: Path = OOD_ARTIFACT_PATH,
        quantile: float = CALIBRATION_QUANTILE,
        regularization: float = COVARIANCE_REGULARIZATION,
    ) -> "OODDetector":
        """
        Fit the reference distribution and calibrate threshold.

        reference_embeddings:
            Known rice-leaf training embeddings.

        calibration_embeddings:
            Separate known rice-leaf validation embeddings.

        quantile:
            Validation distance quantile used as rejection threshold.

        Returns:
            A calibrated OODDetector.
        """

        if not 0.0 < quantile < 1.0:
            raise OODDetectionError(
                "Calibration quantile must be between 0 and 1.",
                code="INVALID_CALIBRATION_QUANTILE",
            )

        if regularization <= 0:
            raise OODDetectionError(
                "Covariance regularization must be positive.",
                code="INVALID_REGULARIZATION",
            )

        reference = cls._validate_embeddings(
            reference_embeddings,
            "Reference embeddings",
        )

        calibration = cls._validate_embeddings(
            calibration_embeddings,
            "Calibration embeddings",
        )

        if reference.shape[1] != calibration.shape[1]:
            raise OODDetectionError(
                "Reference and calibration embeddings "
                "must have matching feature dimensions.",
                code="EMBEDDING_DIMENSION_MISMATCH",
            )

        # Estimate the known-data feature distribution.
        mean = np.mean(reference, axis=0)

        covariance = np.cov(
            reference,
            rowvar=False,
        )

        # Ensure covariance is a 2D matrix, even for one feature.
        covariance = np.atleast_2d(covariance)

        if covariance.shape != (
            reference.shape[1],
            reference.shape[1],
        ):
            raise OODDetectionError(
                "Unexpected covariance matrix dimensions.",
                code="INVALID_COVARIANCE_SHAPE",
            )

        # Scale regularization relative to covariance magnitude.
        scale = float(
            np.trace(covariance) / reference.shape[1]
        )

        if not np.isfinite(scale) or scale <= 0:
            raise OODDetectionError(
                "Reference embeddings have invalid covariance.",
                code="INVALID_COVARIANCE",
            )

        regularized_covariance = covariance + (
            regularization * scale * np.eye(
                reference.shape[1]
            )
        )

        try:
            inverse_covariance = np.linalg.pinv(
                regularized_covariance
            )

        except np.linalg.LinAlgError as exc:
            raise OODDetectionError(
                "Unable to calculate inverse covariance.",
                code="COVARIANCE_INVERSION_FAILED",
            ) from exc

        # Build a temporary detector with the reference statistics.
        # Calibration distances are computed before a threshold exists,
        # so use the internal distance calculation directly.
        detector = cls(artifact_path=artifact_path)

        detector.mean = mean
        detector.inverse_covariance = inverse_covariance
        detector.feature_dimension = reference.shape[1]

        calibration_distances = np.array([
            detector._calculate_distance(embedding)
            for embedding in calibration
        ], dtype=np.float64)

        if (
            calibration_distances.size == 0
            or not np.all(np.isfinite(calibration_distances))
        ):
            raise OODDetectionError(
                "Calibration distances contain invalid values.",
                code="INVALID_CALIBRATION_DISTANCES",
            )

        threshold = float(
            np.quantile(
                calibration_distances,
                quantile,
            )
        )

        if not np.isfinite(threshold) or threshold <= 0:
            raise OODDetectionError(
                "Calibration produced an invalid threshold.",
                code="INVALID_OOD_THRESHOLD",
            )

        detector.threshold = threshold

        return detector

    # -----------------------------------------------------
    # Save calibration artifact
    # -----------------------------------------------------

    def save(self) -> None:
        """Save calibrated detector statistics."""

        self._ensure_calibrated()

        self.artifact_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            self.artifact_path,
            mean=self.mean,
            inverse_covariance=self.inverse_covariance,
            threshold=np.array(self.threshold),
            feature_dimension=np.array(
                self.feature_dimension
            ),
        )

    # -----------------------------------------------------
    # Load calibration artifact
    # -----------------------------------------------------

    def load(self) -> "OODDetector":
        """Load previously calibrated detector statistics."""

        if not self.artifact_path.is_file():
            raise OODDetectionError(
                f"OOD calibration artifact not found: "
                f"{self.artifact_path}",
                code="OOD_CALIBRATION_NOT_FOUND",
            )

        try:
            with np.load(
                self.artifact_path,
                allow_pickle=False,
            ) as artifact:

                mean = np.asarray(
                    artifact["mean"],
                    dtype=np.float64,
                )

                inverse_covariance = np.asarray(
                    artifact["inverse_covariance"],
                    dtype=np.float64,
                )

                threshold = float(
                    artifact["threshold"]
                )

                feature_dimension = int(
                    artifact["feature_dimension"]
                )

        except Exception as exc:
            raise OODDetectionError(
                "Unable to read OOD calibration artifact.",
                code="OOD_CALIBRATION_LOAD_FAILED",
            ) from exc

        if (
            mean.ndim != 1
            or inverse_covariance.ndim != 2
            or inverse_covariance.shape != (
                feature_dimension,
                feature_dimension,
            )
            or mean.shape[0] != feature_dimension
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(inverse_covariance))
            or not np.isfinite(threshold)
            or threshold <= 0
        ):
            raise OODDetectionError(
                "OOD calibration artifact contains invalid statistics.",
                code="INVALID_OOD_CALIBRATION",
            )

        self.mean = mean
        self.inverse_covariance = inverse_covariance
        self.threshold = threshold
        self.feature_dimension = feature_dimension

        return self

    # -----------------------------------------------------
    # Calibration check
    # -----------------------------------------------------

    def _ensure_calibrated(self) -> None:

        if (
            self.mean is None
            or self.inverse_covariance is None
            or self.threshold is None
            or self.feature_dimension is None
        ):
            raise OODDetectionError(
                "OOD detector is not calibrated or loaded.",
                code="OOD_DETECTOR_NOT_READY",
            )

    # -----------------------------------------------------
    # Distance calculation
    # -----------------------------------------------------

    def _calculate_distance(
        self,
        embedding: np.ndarray,
    ) -> float:
        """
        Calculate squared Mahalanobis distance from fitted
        reference statistics. This helper does not require
        a calibrated threshold, so it can also be used in fit().
        """

        if (
            self.mean is None
            or self.inverse_covariance is None
            or self.feature_dimension is None
        ):
            raise OODDetectionError(
                "Reference statistics are not fitted.",
                code="OOD_REFERENCE_NOT_FITTED",
            )

        embedding = np.asarray(
            embedding,
            dtype=np.float64,
        )

        if embedding.ndim != 1:
            raise OODDetectionError(
                "Input embedding must be one-dimensional.",
                code="INVALID_INPUT_EMBEDDING",
            )

        if embedding.shape[0] != self.feature_dimension:
            raise OODDetectionError(
                "Input embedding dimension does not match "
                "the fitted feature dimension.",
                code="EMBEDDING_DIMENSION_MISMATCH",
            )

        if not np.all(np.isfinite(embedding)):
            raise OODDetectionError(
                "Input embedding contains invalid values.",
                code="INVALID_INPUT_EMBEDDING",
            )

        difference = embedding - self.mean

        distance_squared = float(
            difference.T
            @ self.inverse_covariance
            @ difference
        )

        if (
            not np.isfinite(distance_squared)
            or distance_squared < -1e-8
        ):
            raise OODDetectionError(
                "Mahalanobis distance calculation failed.",
                code="INVALID_OOD_DISTANCE",
            )

        return max(distance_squared, 0.0)

    def score_embedding(
        self,
        embedding: np.ndarray,
    ) -> float:
        """
        Calculate squared Mahalanobis distance for a calibrated
        detector.

        Lower distance means closer to the reference
        rice-leaf feature distribution.
        """

        self._ensure_calibrated()
        return self._calculate_distance(embedding)

    # -----------------------------------------------------
    # Detect from embedding
    # -----------------------------------------------------

    def detect_embedding(
        self,
        embedding: np.ndarray,
    ) -> OODResult:

        self._ensure_calibrated()

        distance = self.score_embedding(embedding)

        is_in_distribution = distance <= self.threshold

        if is_in_distribution:
            return OODResult(
                is_in_distribution=True,
                distance=round(distance, 6),
                threshold=round(self.threshold, 6),
                status="IN_DISTRIBUTION",
                message="Image is within the calibrated feature distribution.",
            )

        return OODResult(
            is_in_distribution=False,
            distance=round(distance, 6),
            threshold=round(self.threshold, 6),
            status="OOD_WARNING",
            message="Image is outside the calibrated feature distribution.",
        )

    # -----------------------------------------------------
    # Detect from uploaded image
    # -----------------------------------------------------

    def detect(self, image_bytes: bytes) -> OODResult:
        """
        Extract features and perform OOD detection.

        The feature extractor must accept image bytes and return
        a one-dimensional embedding.

        It must use the same preprocessing and feature layer
        used to generate the calibration embeddings.
        """

        self._ensure_calibrated()

        if not image_bytes:
            raise OODDetectionError(
                "No image data provided.",
                code="EMPTY_IMAGE",
            )

        if self.feature_extractor is None:
            raise OODDetectionError(
                "No ASFR feature extractor has been configured.",
                code="FEATURE_EXTRACTOR_NOT_CONFIGURED",
            )

        try:
            embedding = self.feature_extractor(image_bytes)

        except Exception as exc:
            raise OODDetectionError(
                "Unable to extract ASFR features.",
                code="FEATURE_EXTRACTION_FAILED",
            ) from exc

        return self.detect_embedding(embedding)