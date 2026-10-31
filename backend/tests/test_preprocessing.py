
import io
import struct
import zlib

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from app.preprocessing import (
    ALLOWED_FORMATS,
    ImagePreprocessingError,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    preprocess_image,
)


IMAGE_SIZE = (224, 224)
PARITY_ATOL = 1e-6


def make_image(mode="RGB", fmt="PNG", size=(64, 48)):
    """Create a small encoded image for testing."""
    image = Image.new(mode, size, color=128)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def notebook_reference_preprocess(image_bytes):
    """Reproduce the documented notebook preprocessing operations."""
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )

    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return image.numpy().astype(np.float32)


@pytest.mark.parametrize(
    ("mode", "fmt"),
    [
        ("RGB", "PNG"),
        ("L", "PNG"),
        ("RGB", "JPEG"),
        ("RGB", "BMP"),
        ("RGB", "GIF"),
    ],
)
def test_supported_images_produce_valid_tensor(mode, fmt):
    payload = make_image(mode=mode, fmt=fmt)
    result = preprocess_image(payload)

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= -1.001
    assert result.max() <= 1.001


@pytest.mark.parametrize(
    ("mode", "fmt"),
    [
        ("RGB", "PNG"),
        ("L", "PNG"),
        ("RGB", "JPEG"),
        ("RGB", "BMP"),
        ("RGB", "GIF"),
    ],
)
def test_numerical_parity_with_notebook_reference(mode, fmt):
    """Compare every output value against the reference operations."""
    payload = make_image(mode=mode, fmt=fmt)

    actual = preprocess_image(payload)
    reference = notebook_reference_preprocess(payload)

    assert actual.shape == reference.shape == (224, 224, 3)
    assert actual.dtype == reference.dtype == np.float32

    difference = np.abs(actual - reference)
    max_abs_error = float(difference.max())
    mean_abs_error = float(difference.mean())

    np.testing.assert_allclose(
        actual,
        reference,
        rtol=0,
        atol=PARITY_ATOL,
    )

    print(
        f"{fmt}/{mode}: "
        f"max_abs_error={max_abs_error:.10g}, "
        f"mean_abs_error={mean_abs_error:.10g}, "
        f"atol={PARITY_ATOL}"
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not an image",
        b"\x89PNG\r\n",
    ],
)
def test_corrupted_images_are_rejected(payload):
    with pytest.raises(ImagePreprocessingError):
        preprocess_image(payload)


def test_unsupported_format_is_rejected():
    payload = make_image("RGB", "TIFF")

    with pytest.raises(
        ImagePreprocessingError,
        match="Unsupported",
    ):
        preprocess_image(payload)


def test_oversized_payload_is_rejected():
    payload = b"x" * (MAX_IMAGE_BYTES + 1)

    with pytest.raises(
        ImagePreprocessingError,
        match="size limit",
    ):
        preprocess_image(payload)


def test_excessive_pixel_count_is_rejected():
    """Reject excessive declared PNG dimensions without allocating image pixels."""
    width = MAX_IMAGE_PIXELS + 1
    height = 1

    def png_chunk(chunk_type, data):
        """Build a PNG chunk with its length and CRC."""
        chunk_data = chunk_type + data
        crc = zlib.crc32(chunk_data) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(data))
            + chunk_data
            + struct.pack(">I", crc)
        )

    # Construct a tiny PNG whose IHDR declares too many pixels.
    # The image data is deliberately minimal: preprocessing should reject
    # the dimensions before attempting to decode the image pixels.
    png_payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + png_chunk(b"IEND", b"")
    )

    with pytest.raises(
        ImagePreprocessingError,
        match="pixel-count",
    ):
        preprocess_image(png_payload)


def test_declared_supported_formats_match_expected_allowlist():
    assert ALLOWED_FORMATS == {"JPEG", "PNG", "BMP", "GIF"}
