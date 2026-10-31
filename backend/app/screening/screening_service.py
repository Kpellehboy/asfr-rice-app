from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.screening.image_validator import (
    validate_image_bytes,
    ValidatedImage,
    ImageValidationError,
)

from app.screening.rice_domain import (
    RiceDomainClassifier,
    RiceDomainResult,
    RiceDomainError,
)

from app.screening.ood_detector import (
    OODDetector,
    OODResult,
    OODDetectionError,
)


# ---------------------------------------------------------
# Screening decision
# ---------------------------------------------------------

@dataclass(frozen=True)
class ScreeningResult:
    """
    Final decision produced by the screening pipeline.

    approved_for_inference:
        True only when every screening stage passes.
    """

    approved_for_inference: bool
    status: str
    message: str
    error_code: Optional[str] = None

    image_metadata: Optional[ValidatedImage] = None
    rice_domain: Optional[RiceDomainResult] = None
    ood_result: Optional[OODResult] = None


# ---------------------------------------------------------
# Screening service
# ---------------------------------------------------------

class ScreeningService:
    """
    Coordinates image validation, rice-domain screening,
    and out-of-distribution detection.

    The service does not perform disease classification.

    Only images that pass all screening stages are approved
    for subsequent ASFR inference.
    """

    def __init__(
        self,
        rice_classifier: RiceDomainClassifier,
        ood_detector: OODDetector,
    ):
        self.rice_classifier = rice_classifier
        self.ood_detector = ood_detector

    # -----------------------------------------------------
    # Main screening pipeline
    # -----------------------------------------------------

    def screen(
        self,
        image_bytes: bytes,
        content_type: Optional[str] = None,
    ) -> ScreeningResult:
        """
        Run the complete screening pipeline.

        Pipeline:
            1. Image validation
            2. Rice-domain classification
            3. OOD detection
            4. Final inference eligibility decision

        Returns:
            ScreeningResult

        The caller must invoke ASFR inference only when
        approved_for_inference is True.
        """

        # -------------------------------------------------
        # Stage 1: Image validation
        # -------------------------------------------------

        try:
            image_metadata = validate_image_bytes(
                image_bytes=image_bytes,
                content_type=content_type,
            )

        except ImageValidationError as exc:
            return ScreeningResult(
                approved_for_inference=False,
                status="INVALID_IMAGE",
                message=exc.message,
                error_code=exc.code,
            )

        # -------------------------------------------------
        # Stage 2: Rice-domain screening
        # -------------------------------------------------

        try:
            rice_result = self.rice_classifier.predict(
                image_bytes
            )

        except RiceDomainError as exc:
            return ScreeningResult(
                approved_for_inference=False,
                status="SCREENING_UNAVAILABLE",
                message=exc.message,
                error_code=exc.code,
                image_metadata=image_metadata,
            )

        except Exception:
            return ScreeningResult(
                approved_for_inference=False,
                status="SCREENING_UNAVAILABLE",
                message="Rice-domain screening encountered an unexpected error.",
                error_code="DOMAIN_SCREENING_FAILED",
                image_metadata=image_metadata,
            )

        if not rice_result.is_rice_leaf:
            return ScreeningResult(
                approved_for_inference=False,
                status="UNSUPPORTED_IMAGE",
                message=rice_result.message,
                error_code="NON_RICE_IMAGE",
                image_metadata=image_metadata,
                rice_domain=rice_result,
            )

        # -------------------------------------------------
        # Stage 3: OOD detection
        # -------------------------------------------------

        try:
            ood_result = self.ood_detector.detect(
                image_bytes
            )

        except OODDetectionError as exc:
            return ScreeningResult(
                approved_for_inference=False,
                status="SCREENING_UNAVAILABLE",
                message=exc.message,
                error_code=exc.code,
                image_metadata=image_metadata,
                rice_domain=rice_result,
            )

        except Exception:
            return ScreeningResult(
                approved_for_inference=False,
                status="SCREENING_UNAVAILABLE",
                message="OOD screening encountered an unexpected error.",
                error_code="OOD_SCREENING_FAILED",
                image_metadata=image_metadata,
                rice_domain=rice_result,
            )

        if not ood_result.is_in_distribution:
            return ScreeningResult(
                approved_for_inference=False,
                status="OOD_WARNING",
                message=ood_result.message,
                error_code="OUT_OF_DISTRIBUTION",
                image_metadata=image_metadata,
                rice_domain=rice_result,
                ood_result=ood_result,
            )

        # -------------------------------------------------
        # Stage 4: Final approval
        # -------------------------------------------------

        return ScreeningResult(
            approved_for_inference=True,
            status="ACCEPTED",
            message=(
                "Image passed validation, rice-domain screening, "
                "and OOD assessment. It may proceed to ASFR inference."
            ),
            image_metadata=image_metadata,
            rice_domain=rice_result,
            ood_result=ood_result,
        )