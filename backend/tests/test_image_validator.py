import io

import pytest
from PIL import Image

from app.screening.image_validator import (
    ImageValidationError,
    validate_image_bytes,
)


def create_test_image(
    image_format="JPEG",
    size=(224, 224),
):
    buffer = io.BytesIO()

    image = Image.new("RGB", size, color="green")
    image.save(buffer, format=image_format)

    return buffer.getvalue()


def test_valid_jpeg_is_accepted():
    image_bytes = create_test_image("JPEG")

    result = validate_image_bytes(
        image_bytes,
        content_type="image/jpeg",
    )

    assert result.format == "JPEG"
    assert result.width == 224
    assert result.height == 224


def test_valid_png_is_accepted():
    image_bytes = create_test_image("PNG")

    result = validate_image_bytes(
        image_bytes,
        content_type="image/png",
    )

    assert result.format == "PNG"


def test_empty_image_is_rejected():
    with pytest.raises(ImageValidationError) as error:
        validate_image_bytes(b"")

    assert error.value.code == "EMPTY_IMAGE"


def test_corrupt_image_is_rejected():
    with pytest.raises(ImageValidationError):
        validate_image_bytes(b"this is not an image")


def test_unsupported_mime_type_is_rejected():
    image_bytes = create_test_image("JPEG")

    with pytest.raises(ImageValidationError) as error:
        validate_image_bytes(
            image_bytes,
            content_type="application/pdf",
        )

    assert error.value.code == "UNSUPPORTED_MIME_TYPE"


def test_small_image_is_rejected():
    image_bytes = create_test_image(
        "JPEG",
        size=(32, 32),
    )

    with pytest.raises(ImageValidationError) as error:
        validate_image_bytes(image_bytes)

    assert error.value.code == "IMAGE_TOO_SMALL"


def test_mime_format_mismatch_is_rejected():
    image_bytes = create_test_image("JPEG")

    with pytest.raises(ImageValidationError) as error:
        validate_image_bytes(
            image_bytes,
            content_type="image/png",
        )

    assert error.value.code == "MIME_FORMAT_MISMATCH"


def test_oversized_image_is_rejected():
    image_bytes = b"x" * (10 * 1024 * 1024 + 1)

    with pytest.raises(ImageValidationError) as error:
        validate_image_bytes(image_bytes)

    assert error.value.code == "FILE_TOO_LARGE"