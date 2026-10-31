
from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from keras import Model

from app.preprocessing import preprocess_image


TARGET_LAYER = "asfr_fusion_relu"
GATE_LAYER = "asfr_spatial_gate"
EXPECTED_CLASS_COUNT = 9
EXPECTED_INPUT_SHAPE = (224, 224, 3)
EXPECTED_TARGET_SHAPE = (14, 14, 256)
EXPECTED_GATE_SHAPE = (14, 14, 1)
NUMERICAL_EPSILON = 1e-12


class XAIError(RuntimeError):
    """Raised when an explanation cannot be generated reliably."""


class GradCAMExplainer:
    """Generate class-specific Grad-CAM and separate ASFR gate maps."""

    def __init__(self, inference_engine: Any):
        model = getattr(inference_engine, "model", None)
        class_names = getattr(inference_engine, "class_names", None)

        if model is None or not class_names:
            raise XAIError("A loaded inference engine is required.")

        try:
            target = model.get_layer(TARGET_LAYER)
            gate = model.get_layer(GATE_LAYER)

            self.model = model
            self.class_names = list(class_names)
            self.model_version = getattr(
                inference_engine, "model_version", "unknown"
            )
            self.target_layer = TARGET_LAYER

            if len(self.class_names) != EXPECTED_CLASS_COUNT:
                raise ValueError("Expected exactly nine class labels.")

            if tuple(model.input_shape[1:]) != EXPECTED_INPUT_SHAPE:
                raise ValueError("Unexpected model input shape.")

            if tuple(target.output.shape[1:]) != EXPECTED_TARGET_SHAPE:
                raise ValueError("Unexpected Grad-CAM target shape.")

            if tuple(gate.output.shape[1:]) != EXPECTED_GATE_SHAPE:
                raise ValueError("Unexpected ASFR gate shape.")

            if int(model.output_shape[-1]) != EXPECTED_CLASS_COUNT:
                raise ValueError("Unexpected model output class count.")

            # Both outputs remain connected to the original saved model.
            self.grad_model = Model(
                inputs=model.inputs,
                outputs=[target.output, model.output],
                name="asfr_gradcam_model",
            )

            self.gate_model = Model(
                inputs=model.inputs,
                outputs=gate.output,
                name="asfr_gate_extractor",
            )

            # Read-only feature extractor for OOD calibration and inference.
            # It reuses the original ASFR checkpoint and does not add weights.
            self.ood_feature_model = Model(
                inputs=model.inputs,
                outputs=target.output,
                name="asfr_ood_feature_extractor",
            )

        except Exception as exc:
            raise XAIError(
                "Could not initialize XAI models from the saved checkpoint."
            ) from exc

    @staticmethod
    def _decode_original(image_bytes: bytes) -> np.ndarray:
        """Decode an image as RGB without changing its original dimensions."""
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("Image must be supplied as non-empty bytes.")

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                return np.asarray(image.convert("RGB"), dtype=np.uint8)

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(
                "Input is not a supported valid image."
            ) from exc

    def _prepare(
        self, image_bytes: bytes
    ) -> tuple[np.ndarray, tf.Tensor]:
        """Apply the same preprocessing used by the inference pipeline."""
        original_rgb = self._decode_original(image_bytes)

        try:
            processed = np.asarray(
                preprocess_image(image_bytes), dtype=np.float32
            )
        except Exception as exc:
            raise XAIError("Image preprocessing failed.") from exc

        if processed.shape != EXPECTED_INPUT_SHAPE:
            raise XAIError(
                f"Preprocessing returned unexpected shape: {processed.shape}"
            )

        if not np.isfinite(processed).all():
            raise XAIError(
                "Preprocessed image contains non-finite values."
            )

        batch = tf.convert_to_tensor(
            processed[None, ...], dtype=tf.float32
        )

        return original_rgb, batch

    @staticmethod
    def _resize_map(
        values: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Resize a normalized float map without 8-bit quantization."""
        values = np.asarray(values, dtype=np.float32)

        if values.ndim != 2 or not np.isfinite(values).all():
            raise XAIError("Explanation map is invalid.")

        values = np.clip(values, 0.0, 1.0)

        # PIL mode F preserves floating-point values during resizing.
        image = Image.fromarray(values, mode="F")
        resized = image.resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )

        result = np.asarray(resized, dtype=np.float32)
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _make_overlay(
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Create an RGB overlay using a simple blue-to-red heat palette."""
        heat_rgb = np.stack(
            [
                heatmap,
                np.zeros_like(heatmap),
                1.0 - heatmap,
            ],
            axis=-1,
        )

        original_float = original_rgb.astype(np.float32) / 255.0
        overlay = (
            (1.0 - alpha) * original_float + alpha * heat_rgb
        )

        return np.clip(
            np.rint(overlay * 255.0), 0, 255
        ).astype(np.uint8)

    def explain(
        self,
        image_bytes: bytes,
        target_class_index: int | None = None,
        overlay_alpha: float = 0.40,
    ) -> dict[str, Any]:
        """Return Grad-CAM, aligned heatmap, overlay, and diagnostics.

        A zero heatmap is returned when the selected target has no positive
        Grad-CAM response after ReLU. This represents the computed result;
        it is not replaced with an artificial or random explanation.
        """
        if (
            not isinstance(overlay_alpha, (int, float))
            or isinstance(overlay_alpha, bool)
            or not np.isfinite(overlay_alpha)
            or not 0.0 <= overlay_alpha <= 1.0
        ):
            raise ValueError("overlay_alpha must be between 0 and 1.")

        original_rgb, batch = self._prepare(image_bytes)

        try:
            with tf.GradientTape() as tape:
                activations, predictions = self.grad_model(
                    batch, training=False
                )

                if target_class_index is None:
                    class_index = int(
                        tf.argmax(
                            predictions[0], output_type=tf.int32
                        ).numpy()
                    )
                else:
                    if (
                        not isinstance(target_class_index, int)
                        or isinstance(target_class_index, bool)
                        or not 0 <= target_class_index
                        < len(self.class_names)
                    ):
                        raise ValueError(
                            "Target class index must be an integer from 0 to 8."
                        )

                    class_index = target_class_index

                # This checkpoint exposes softmax probabilities.
                # The explanation score is therefore the selected class
                # probability, not a pre-softmax logit.
                class_score = predictions[:, class_index]

            gradients = tape.gradient(class_score, activations)

            if gradients is None:
                raise XAIError(
                    f"Class score is disconnected from {self.target_layer}."
                )

            gradients_np = gradients.numpy()
            activations_np = activations.numpy()
            predictions_np = predictions.numpy()

            if not np.isfinite(gradients_np).all():
                raise XAIError("Grad-CAM gradients are non-finite.")

            if not np.isfinite(activations_np).all():
                raise XAIError("Grad-CAM activations are non-finite.")

            if not np.isfinite(predictions_np).all():
                raise XAIError("Model predictions contain non-finite values.")

            gradient_magnitude = float(
                np.max(np.abs(gradients_np))
            )

            if gradient_magnitude <= NUMERICAL_EPSILON:
                raise XAIError(
                    "Grad-CAM gradients are effectively zero for "
                    "this input and target class."
                )

            # Standard Grad-CAM channel weights:
            # spatial average of the gradients for each channel.
            channel_weights = np.mean(
                gradients_np[0], axis=(0, 1)
            )

            raw_cam = np.sum(
                activations_np[0]
                * channel_weights[None, None, :],
                axis=-1,
            )

            if not np.isfinite(raw_cam).all():
                raise XAIError("Grad-CAM contains non-finite values.")

            # ReLU retains positive class-associated evidence.
            positive_cam = np.maximum(raw_cam, 0.0)
            positive_max = float(np.max(positive_cam))
            positive_min = float(np.min(positive_cam))

            has_positive_response = (
                positive_max > NUMERICAL_EPSILON
            )

            if has_positive_response:
                normalized_cam = positive_cam / positive_max
                normalization = "positive_max"
                warning = None
            else:
                # A zero map is mathematically faithful to the ReLU result.
                # Do not invent a heatmap or silently switch methods.
                normalized_cam = np.zeros_like(
                    positive_cam, dtype=np.float32
                )
                normalization = "none_empty_positive_response"
                warning = (
                    "No positive Grad-CAM response was found for this "
                    "target class at the selected layer. The displayed "
                    "heatmap is zero-valued and should not be interpreted "
                    "as evidence of a localized region."
                )

            height, width = original_rgb.shape[:2]

            heatmap = self._resize_map(
                normalized_cam, width, height
            )

            overlay = self._make_overlay(
                original_rgb,
                heatmap,
                float(overlay_alpha),
            )

            probability = float(
                predictions_np[0, class_index]
            )

            if (
                not np.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise XAIError(
                    "Selected class probability is invalid."
                )

            return {
                "explained_class_index": class_index,
                "explained_class": self.class_names[class_index],
                "class_probability": probability,
                "method": "Grad-CAM",
                "target_layer": self.target_layer,
                "model_version": self.model_version,
                "heatmap": heatmap.astype(np.float32),
                "overlay_rgb": overlay,
                "warning": warning,
                "metadata": {
                    "input_width": width,
                    "input_height": height,
                    "heatmap_width": width,
                    "heatmap_height": height,
                    "feature_map_shape": list(
                        activations_np.shape[1:3]
                    ),
                    "feature_channels": int(
                        activations_np.shape[-1]
                    ),
                    "gradient_max_abs": gradient_magnitude,
                    "positive_cam_min": positive_min,
                    "positive_cam_max": positive_max,
                    "has_positive_response": has_positive_response,
                    "normalization": normalization,
                    "overlay_alpha": float(overlay_alpha),
                    "score_type": "softmax_probability",
                    "class_specific": True,
                    "method_limitations": (
                        "Grad-CAM is a gradient-based localization "
                        "explanation, not proof of causal feature "
                        "importance. A zero heatmap means no positive "
                        "response survived ReLU at this target layer."
                    ),
                },
            }

        except (ValueError, XAIError):
            raise
        except Exception as exc:
            raise XAIError("Grad-CAM generation failed.") from exc

    def extract_ood_embedding(
        self, image_bytes: bytes
    ) -> dict[str, Any]:
        """Extract a 256-dimensional embedding from ASFR fusion features.

        The original ASFR checkpoint is used in inference mode. The spatial
        feature tensor (14, 14, 256) is reduced with global average pooling
        over its two spatial dimensions, yielding a (256,) embedding.

        This method only extracts features. It does not decide whether an
        image is in-distribution; that decision belongs to the calibrated
        OOD detector.
        """
        _, batch = self._prepare(image_bytes)

        try:
            feature_batch = self.ood_feature_model(
                batch, training=False
            ).numpy()

            expected_batch_shape = (1, *EXPECTED_TARGET_SHAPE)
            if feature_batch.shape != expected_batch_shape:
                raise XAIError(
                    "ASFR feature extractor returned an unexpected shape: "
                    f"{feature_batch.shape}; expected {expected_batch_shape}."
                )

            if not np.isfinite(feature_batch).all():
                raise XAIError(
                    "ASFR feature tensor contains non-finite values."
                )

            # (1, 14, 14, 256) -> (1, 256).
            embedding_batch = np.mean(
                feature_batch, axis=(1, 2), dtype=np.float32
            )

            if embedding_batch.shape != (1, EXPECTED_TARGET_SHAPE[-1]):
                raise XAIError(
                    "ASFR embedding has an unexpected shape: "
                    f"{embedding_batch.shape}."
                )

            embedding = embedding_batch[0].astype(
                np.float32, copy=False
            )

            if not np.isfinite(embedding).all():
                raise XAIError(
                    "ASFR embedding contains non-finite values."
                )

            return {
                "embedding": embedding,
                "target_layer": self.target_layer,
                "model_version": self.model_version,
                "metadata": {
                    "feature_map_shape": list(EXPECTED_TARGET_SHAPE),
                    "embedding_dimension": int(embedding.shape[0]),
                    "pooling": "global_average_pooling",
                    "feature_extractor": "asfr_ood_feature_extractor",
                },
            }

        except (ValueError, XAIError):
            raise
        except Exception as exc:
            raise XAIError(
                "ASFR OOD feature extraction failed."
            ) from exc

    def extract_asfr_gate(
        self, image_bytes: bytes
    ) -> dict[str, Any]:
        """Return the learned ASFR spatial gate independently of Grad-CAM."""
        original_rgb, batch = self._prepare(image_bytes)

        try:
            gate_batch = self.gate_model(
                batch, training=False
            ).numpy()

            gate = gate_batch[0, ..., 0]

            if not np.isfinite(gate).all():
                raise XAIError(
                    "ASFR gate contains non-finite values."
                )

            # This layer is expected to be sigmoid-gated.
            if np.min(gate) < -1e-6 or np.max(gate) > 1.0 + 1e-6:
                raise XAIError(
                    "ASFR gate values fall outside the expected "
                    "sigmoid range [0, 1]."
                )

            gate = np.clip(gate, 0.0, 1.0)

            height, width = original_rgb.shape[:2]
            aligned_gate = self._resize_map(
                gate, width, height
            )

            return {
                "method": "ASFR spatial gate",
                "target_layer": GATE_LAYER,
                "model_version": self.model_version,
                "gate_map": aligned_gate.astype(np.float32),
                "warning": (
                    "The gate shows learned spatial gating values. "
                    "It is not class-specific Grad-CAM and is not, "
                    "by itself, evidence of causal feature importance."
                ),
                "metadata": {
                    "input_width": width,
                    "input_height": height,
                    "gate_width": width,
                    "gate_height": height,
                    "native_gate_shape": list(gate.shape),
                    "gate_min": float(np.min(gate)),
                    "gate_max": float(np.max(gate)),
                    "class_specific": False,
                    "interpretation": (
                        "Learned ASFR spatial gating values; not "
                        "Grad-CAM and not, by themselves, causal "
                        "feature importance."
                    ),
                },
            }

        except XAIError:
            raise
        except Exception as exc:
            raise XAIError(
                "ASFR gate extraction failed."
            ) from exc
