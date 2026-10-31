from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DOMAIN_MODEL_PATH = Path(
    os.getenv(
        "RICE_DOMAIN_MODEL_PATH",
        "model_artifacts/rice_domain_classifier.keras",
    )
)

IMAGE_SIZE = int(os.getenv("RICE_DOMAIN_IMAGE_SIZE", "224"))

DOMAIN_THRESHOLD = float(
    os.getenv("RICE_DOMAIN_THRESHOLD", "0.5")
)

NORMALIZATION = os.getenv(
    "RICE_DOMAIN_NORMALIZATION",
    "resnet50v2",
).lower()

# Supported normalization:
# resnet50v2 -> pixel values normalized to [-1, 1]
# zero_one   -> pixel values normalized to [0, 1]


# ---------------------------------------------------------
# Result structure
# ---------------------------------------------------------

@dataclass(frozen=True)
class RiceDomainResult:
    is_rice_leaf: bool
    confidence: float
    threshold: float
    status: str
    message: str


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------

class RiceDomainError(Exception):
    """Raised when rice-domain screening cannot be completed."""

    def __init__(
        self,
        message: str,
        code: str = "RICE_DOMAIN_ERROR",
    ):
        self.message = message
        self.code = code
        super().__init__(message)


# ---------------------------------------------------------
# Rice-domain classifier
# ---------------------------------------------------------

class RiceDomainClassifier:
    """
    Separate binary classifier for rice-leaf domain screening.

    Expected model:
        Input:  (batch, IMAGE_SIZE, IMAGE_SIZE, 3)
        Output: (batch, 1)

    Output interpretation:
        0.0 = non-rice image
        1.0 = rice-leaf image

    The model must be trained separately using labelled
    rice-leaf and non-rice images.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold: float = DOMAIN_THRESHOLD,
        image_size: int = IMAGE_SIZE,
        normalization: str = NORMALIZATION,
    ):
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        if image_size <= 0:
            raise ValueError(
                "Image size must be greater than zero."
            )

        if normalization not in {"resnet50v2", "zero_one"}:
            raise ValueError(
                "Normalization must be 'resnet50v2' or 'zero_one'."
            )

        self.model_path = Path(model_path or DOMAIN_MODEL_PATH)
        self.threshold = threshold
        self.image_size = image_size
        self.normalization = normalization

        self._model = None

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    def load_model(self):
        """
        Load the separately trained rice-domain model.

        Model loading is lazy: it happens on first prediction.
        """

        if self._model is not None:
            return self._model

        if not self.model_path.is_file():
            raise RiceDomainError(
                f"Rice-domain model not found: {self.model_path}. "
                "Train and provide the binary domain classifier first.",
                code="DOMAIN_MODEL_NOT_FOUND",
            )

        try:
            from tensorflow import keras

            self._model = keras.models.load_model(
                self.model_path,
                compile=False,
            )

            return self._model

        except ImportError as exc:
            raise RiceDomainError(
                "TensorFlow is required to load the rice-domain model.",
                code="TENSORFLOW_NOT_AVAILABLE",
            ) from exc

        except Exception as exc:
            raise RiceDomainError(
                f"Unable to load rice-domain model: {exc}",
                code="DOMAIN_MODEL_LOAD_FAILED",
            ) from exc

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert uploaded image bytes into model input.

        The preprocessing must match the binary classifier's
        training preprocessing exactly.
        """

        if not image_bytes:
            raise RiceDomainError(
                "No image data provided.",
                code="EMPTY_IMAGE",
            )

        try:
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")

                image = image.resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.BILINEAR,
                )

                image_array = np.asarray(
                    image,
                    dtype=np.float32,
                )

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RiceDomainError(
                "Unable to decode image for rice-domain screening.",
                code="IMAGE_DECODE_FAILED",
            ) from exc

        # Apply the normalization used during model training.
        if self.normalization == "resnet50v2":
            image_array = (image_array / 127.5) - 1.0

        elif self.normalization == "zero_one":
            image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        return image_array

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    def predict(self, image_bytes: bytes) -> RiceDomainResult:
        """
        Determine whether an image belongs to the rice-leaf domain.

        Returns a RiceDomainResult.

        This method does not perform disease classification.
        """

        model = self.load_model()

        model_input = self.preprocess(image_bytes)

        try:
            prediction = model.predict(
                model_input,
                verbose=0,
            )

            prediction = np.asarray(prediction)

            # This implementation expects one sigmoid output.
            if prediction.size != 1:
                raise RiceDomainError(
                    "Expected a binary classifier with one sigmoid output.",
                    code="INVALID_DOMAIN_MODEL_OUTPUT",
                )

            confidence = float(prediction.reshape(-1)[0])

            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise RiceDomainError(
                    "Domain classifier returned an invalid probability.",
                    code="INVALID_DOMAIN_CONFIDENCE",
                )

        except RiceDomainError:
            raise

        except Exception as exc:
            raise RiceDomainError(
                f"Rice-domain prediction failed: {exc}",
                code="DOMAIN_PREDICTION_FAILED",
            ) from exc

        is_rice_leaf = confidence >= self.threshold

        if is_rice_leaf:
            return RiceDomainResult(
                is_rice_leaf=True,
                confidence=round(confidence, 4),
                threshold=self.threshold,
                status="ACCEPTED",
                message="Image passed rice-leaf domain screening.",
            )

        return RiceDomainResult(
            is_rice_leaf=False,
            confidence=round(confidence, 4),
            threshold=self.threshold,
            status="UNSUPPORTED_IMAGE",
            message="Image does not appear to belong to the rice-leaf domain.",
        )