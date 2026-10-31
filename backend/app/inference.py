
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import keras

from app.preprocessing import preprocess_image

logger = logging.getLogger(__name__)

PROJECT_ROOT = (
    Path("/app")
    if Path("/app/model_artifacts").is_dir()
    else Path(__file__).resolve().parents[2]
)
ARTIFACT_DIR = PROJECT_ROOT / "model_artifacts"

MODEL_PATH = ARTIFACT_DIR / "best_model.keras"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

EXPECTED_INPUT_SHAPE = (224, 224, 3)
EXPECTED_CLASS_COUNT = 9


class ModelLoadError(RuntimeError):
    """The saved inference model or its metadata could not be loaded."""


class InferenceError(RuntimeError):
    """Inference failed or returned invalid model output."""


class InferenceEngine:
    """Loads the verified model once and exposes read-only prediction."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        class_names_path: Path = CLASS_NAMES_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.metadata_path = Path(metadata_path)

        self.model: Any = None
        self.class_names: list[str] = []
        self.model_version: str | None = None

    def load(self) -> None:
        """Load and validate model artifacts once during initialization."""
        if self.model is not None:
            return

        try:
            for path in (
                self.model_path,
                self.class_names_path,
                self.metadata_path,
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Required model artifact is missing: {path.name}"
                    )

            with self.class_names_path.open(
                "r", encoding="utf-8"
            ) as file:
                class_data = json.load(file)

            class_names = class_data.get("class_names")

            if (
                not isinstance(class_names, list)
                or len(class_names) != EXPECTED_CLASS_COUNT
                or any(
                    not isinstance(name, str) or not name.strip()
                    for name in class_names
                )
                or len(set(class_names)) != EXPECTED_CLASS_COUNT
            ):
                raise ValueError("Invalid nine-class mapping")

            with self.metadata_path.open(
                "r", encoding="utf-8"
            ) as file:
                metadata = json.load(file)

            if not isinstance(metadata, dict):
                raise ValueError("Model metadata must be a JSON object.")

            # Hash the checkpoint before loading so the reported model
            # identity is tied to the exact artifact on disk.
            digest = hashlib.sha256()
            with self.model_path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)

            checkpoint_hash = digest.hexdigest()
            declared_hash = metadata.get("sha256")

            if declared_hash and declared_hash != checkpoint_hash:
                raise ValueError("Checkpoint SHA-256 does not match metadata.")

            model = keras.models.load_model(
                str(self.model_path),
                compile=False,
            )

            input_shape = tuple(model.input_shape)
            output_shape = tuple(model.output_shape)

            if input_shape != (None, *EXPECTED_INPUT_SHAPE):
                raise ValueError(
                    f"Unexpected model input shape: {input_shape}"
                )

            if output_shape != (None, EXPECTED_CLASS_COUNT):
                raise ValueError(
                    f"Unexpected model output shape: {output_shape}"
                )

            declared_version = (
                metadata.get("model_version")
                or metadata.get("version")
            )

            self.model = model
            self.class_names = list(class_names)
            self.model_version = (
                str(declared_version)
                if declared_version
                else f"sha256:{checkpoint_hash}"
            )

            logger.info(
                "Inference model loaded: classes=%d model_version=%s",
                len(self.class_names),
                self.model_version,
            )

        except Exception as exc:
            self.model = None
            self.class_names = []
            self.model_version = None
            logger.exception("Inference model initialization failed")
            raise ModelLoadError(
                "The inference model could not be initialized."
            ) from exc

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Predict one image and return all nine class probabilities."""
        if self.model is None:
            raise ModelLoadError("The inference model is not loaded.")

        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("Image input must be non-empty bytes.")

        try:
            image = preprocess_image(image_bytes)
            image = np.asarray(image, dtype=np.float32)

            if image.shape != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"Preprocessing returned invalid shape: {image.shape}"
                )

            if not np.isfinite(image).all():
                raise ValueError(
                    "Preprocessing returned non-finite pixel values."
                )

            batch = np.expand_dims(image, axis=0)

            # Inference only: no training or weight updates.
            raw_output = self.model(batch, training=False)
            probabilities = np.asarray(raw_output, dtype=np.float64)

            if probabilities.shape != (1, EXPECTED_CLASS_COUNT):
                raise InferenceError(
                    f"Invalid model output shape: {probabilities.shape}"
                )

            probabilities = probabilities[0]

            if not np.isfinite(probabilities).all():
                raise InferenceError(
                    "Model output contains non-finite values."
                )

            if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
                raise InferenceError(
                    "Model output is outside the probability range."
                )

            if not np.isclose(
                probabilities.sum(), 1.0, atol=1e-5
            ):
                raise InferenceError(
                    "Model probabilities do not sum to approximately one."
                )

            predicted_index = int(np.argmax(probabilities))

            probability_list = [
                {
                    "class_index": index,
                    "class_name": name,
                    "probability": float(probabilities[index]),
                }
                for index, name in enumerate(self.class_names)
            ]

            return {
                "predicted_class_index": predicted_index,
                "predicted_class": self.class_names[predicted_index],
                "predicted_probability": float(
                    probabilities[predicted_index]
                ),
                "probabilities": probability_list,
                "model_version": self.model_version,
            }

        except (ModelLoadError, InferenceError, ValueError):
            raise
        except Exception as exc:
            logger.exception("Inference execution failed")
            raise InferenceError(
                "Prediction failed. Check the image and retry."
            ) from exc
