
"""Safe inference preprocessing matching the ASFR training notebook."""

from __future__ import annotations

import io
import warnings

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SIZE = (224, 224)
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

# Supported formats accepted by the TensorFlow decoder.
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "GIF"}

warnings.simplefilter("error", Image.DecompressionBombWarning)


class ImagePreprocessingError(ValueError):
    """Raised when image input is invalid, unsafe, or unsupported."""


def _validate_image_bytes(image_bytes: bytes) -> None:
    """Validate image size, format, dimensions, and file integrity."""

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ImagePreprocessingError(
            "Image content must be non-empty bytes."
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImagePreprocessingError(
            "Image file exceeds the upload size limit."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ImagePreprocessingError(
                    "Unsupported image format."
                )

            width, height = image.size

            if width <= 0 or height <= 0:
                raise ImagePreprocessingError(
                    "Invalid image dimensions."
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise ImagePreprocessingError(
                    "Image exceeds the pixel-count limit."
                )

            # Check the encoded image structure.
            image.verify()

        # Reopen and force decoding to catch malformed image data.
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()

            # Validate the dimensions after applying orientation metadata.
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size

            if width <= 0 or height <= 0:
                raise ImagePreprocessingError(
                    "Invalid image dimensions."
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise ImagePreprocessingError(
                    "Image exceeds the pixel-count limit."
                )

    except ImagePreprocessingError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        ValueError,
        SyntaxError,
    ) as exc:
        raise ImagePreprocessingError(
            "Invalid, corrupted, or unsafe image content."
        ) from exc


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Return a normalized float32 tensor with shape (224, 224, 3).

    Reference pipeline:
      1. Decode using tf.io.decode_image(channels=3,
         expand_animations=False).
      2. Resize to 224x224 using bilinear interpolation with antialiasing.
      3. Apply ResNetV2 preprocess_input.

    No training augmentation is applied.

    Note:
      TensorFlow performs the actual decoding so the image-processing
      operations match the reference pipeline. Pillow is used only for
      preliminary format, integrity, and resource-limit validation.
    """

    _validate_image_bytes(image_bytes)

    try:
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )

        image.set_shape([None, None, 3])

        # Check decoded dimensions before allocating the resized tensor.
        dimensions = tf.shape(image)
        height = int(dimensions[0].numpy())
        width = int(dimensions[1].numpy())

        if height <= 0 or width <= 0:
            raise ImagePreprocessingError(
                "Decoded image has invalid dimensions."
            )

        if height * width > MAX_IMAGE_PIXELS:
            raise ImagePreprocessingError(
                "Decoded image exceeds the pixel-count limit."
            )

        image = tf.image.resize(
            image,
            IMAGE_SIZE,
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=True,
        )

        image = tf.keras.applications.resnet_v2.preprocess_input(
            image
        )

        result = np.asarray(
            image.numpy(),
            dtype=np.float32,
        )

    except ImagePreprocessingError:
        raise
    except (
        tf.errors.OpError,
        ValueError,
        OSError,
        TypeError,
    ) as exc:
        raise ImagePreprocessingError(
            "TensorFlow could not decode or preprocess this image."
        ) from exc

    if result.shape != (224, 224, 3):
        raise ImagePreprocessingError(
            "Unexpected preprocessed tensor shape."
        )

    if result.dtype != np.float32:
        raise ImagePreprocessingError(
            "Unexpected output tensor dtype."
        )

    if not np.isfinite(result).all():
        raise ImagePreprocessingError(
            "Preprocessing produced non-finite values."
        )

    if result.min() < -1.001 or result.max() > 1.001:
        raise ImagePreprocessingError(
            "Output values are outside the expected range."
        )

    return result
