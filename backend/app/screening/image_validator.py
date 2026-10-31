from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------
# Validation policy
# ---------------------------------------------------------

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64

MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 16_000_000

ALLOWED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",  # Common alias
    "image/png",
    "image/webp",
}


# ---------------------------------------------------------
# Validation result
# ---------------------------------------------------------

@dataclass(frozen=True)
class ValidatedImage:
    """
    Metadata for an image that has passed validation.

    The original image bytes are not modified.
    """

    format: str
    mime_type: str
    width: int
    height: int
    mode: str
    size_bytes: int

    @property
    def aspect_ratio(self) -> float:
        return round(self.width / self.height, 4)


# ---------------------------------------------------------
# Validation exception
# ---------------------------------------------------------

class ImageValidationError(Exception):
    """
    Raised when an uploaded image fails validation.
    """

    def __init__(
        self,
        message: str,
        code: str = "INVALID_IMAGE",
    ) -> None:
        self.message = message
        self.code = code
        super().__init__(message)


# ---------------------------------------------------------
# Main validation function
# ---------------------------------------------------------

def validate_image_bytes(
    image_bytes: bytes,
    content_type: Optional[str] = None,
) -> ValidatedImage:
    """
    Validate uploaded image bytes before model inference.

    Checks:
        1. Empty upload
        2. Maximum file size
        3. Supported MIME type
        4. Actual image format
        5. MIME/format consistency
        6. Image integrity and decodability
        7. Static image requirement
        8. Image dimensions and pixel count

    Args:
        image_bytes:
            Raw uploaded image bytes.

        content_type:
            MIME type supplied by the upload request.
            This is checked but never trusted as proof of format.

    Returns:
        ValidatedImage:
            Metadata describing the validated image.

    Raises:
        ImageValidationError:
            If any validation check fails.
    """

    # 1. Check whether the upload is empty.
    if not image_bytes:
        raise ImageValidationError(
            "No image data was provided.",
            code="EMPTY_IMAGE",
        )

    # 2. Enforce upload size.
    file_size = len(image_bytes)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError(
            "Image exceeds the maximum allowed upload size of 10 MB.",
            code="FILE_TOO_LARGE",
        )

    # 3. Validate declared MIME type, if provided.
    declared_mime = None

    if content_type:
        declared_mime = content_type.split(";")[0].strip().lower()

        if declared_mime not in ALLOWED_MIME_TYPES:
            raise ImageValidationError(
                "Unsupported image MIME type. "
                "Upload a JPEG, PNG, or WebP image.",
                code="UNSUPPORTED_MIME_TYPE",
            )

    # 4. Inspect actual image content using Pillow.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(BytesIO(image_bytes)) as image:

                actual_format = image.format

                if actual_format not in ALLOWED_FORMATS:
                    raise ImageValidationError(
                        "Unsupported image format. "
                        "Only JPEG, PNG, and WebP are accepted.",
                        code="UNSUPPORTED_IMAGE_FORMAT",
                    )

                # 5. Confirm MIME type matches actual format.
                actual_mime = ALLOWED_FORMATS[actual_format]

                if (
                    declared_mime is not None
                    and declared_mime != actual_mime
                    and not (
                        declared_mime == "image/jpg"
                        and actual_format == "JPEG"
                    )
                ):
                    raise ImageValidationError(
                        "The uploaded MIME type does not match "
                        "the actual image format.",
                        code="MIME_FORMAT_MISMATCH",
                    )

                # 6. Reject animated or multi-frame images.
                if getattr(image, "n_frames", 1) > 1:
                    raise ImageValidationError(
                        "Animated or multi-frame images are not supported.",
                        code="ANIMATED_IMAGE_NOT_SUPPORTED",
                    )

                width, height = image.size

                # 7. Check dimensions.
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    raise ImageValidationError(
                        "Image resolution is too small. "
                        f"Minimum dimensions are "
                        f"{MIN_IMAGE_WIDTH} × {MIN_IMAGE_HEIGHT} pixels.",
                        code="IMAGE_TOO_SMALL",
                    )

                if (
                    width > MAX_IMAGE_WIDTH
                    or height > MAX_IMAGE_HEIGHT
                ):
                    raise ImageValidationError(
                        "Image dimensions exceed the maximum allowed "
                        f"resolution of {MAX_IMAGE_WIDTH} × "
                        f"{MAX_IMAGE_HEIGHT} pixels.",
                        code="IMAGE_DIMENSIONS_TOO_LARGE",
                    )

                pixel_count = width * height

                if pixel_count > MAX_IMAGE_PIXELS:
                    raise ImageValidationError(
                        "Image contains too many pixels.",
                        code="IMAGE_PIXEL_LIMIT_EXCEEDED",
                    )

                # 8. Verify that the image can actually be decoded.
                image.verify()

            # Reopen after verify(), then force full decoding.
            with Image.open(BytesIO(image_bytes)) as decoded_image:
                decoded_image.load()

                return ValidatedImage(
                    format=actual_format,
                    mime_type=actual_mime,
                    width=width,
                    height=height,
                    mode=decoded_image.mode,
                    size_bytes=file_size,
                )

    except ImageValidationError:
        raise

    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            "Image was rejected because its decompressed size is unsafe.",
            code="DECOMPRESSION_BOMB",
        ) from exc

    except Image.DecompressionBombWarning as exc:
        raise ImageValidationError(
            "Image was rejected because its decompressed size is unsafe.",
            code="DECOMPRESSION_BOMB",
        ) from exc

    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            "The uploaded file is not a valid or readable image.",
            code="CORRUPT_OR_INVALID_IMAGE",
        ) from exc