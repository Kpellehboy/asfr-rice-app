
from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from io import BytesIO
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.inference import InferenceEngine, InferenceError, ModelLoadError
from app.xai import GradCAMExplainer, XAIError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("asfr.api")

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_TARGET_CLASS = 8

# Maximum time an HTTP request waits for model operations.
# This does not forcibly terminate a running TensorFlow worker thread.
MODEL_OPERATION_TIMEOUT_SECONDS = 60.0

# TensorFlow model and gradient operations share one lock.
# A timed-out operation retains this lock until its worker finishes.
MODEL_EXECUTION_LOCK = threading.RLock()


class HealthResponse(BaseModel):
    status: str
    stage: str


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str | None
    xai_ready: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    input_shape: list[int | None]
    output_shape: list[int | None]
    class_names: list[str]
    class_count: int


class ClassProbability(BaseModel):
    class_index: int = Field(ge=0, le=MAX_TARGET_CLASS)
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int = Field(ge=0, le=MAX_TARGET_CLASS)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: list[ClassProbability]
    model_version: str
    request_id: str
    filename: str | None = None
    content_type: str | None = None


class ExplanationResponse(BaseModel):
    explained_class_index: int = Field(ge=0, le=MAX_TARGET_CLASS)
    explained_class: str
    class_probability: float = Field(ge=0.0, le=1.0)
    method: str
    target_layer: str
    model_version: str
    warning: str | None = None
    metadata: dict[str, Any]
    heatmap_png_base64: str
    overlay_png_base64: str
    request_id: str


class GateResponse(BaseModel):
    method: str
    target_layer: str
    model_version: str
    warning: str | None = None
    metadata: dict[str, Any]
    gate_png_base64: str
    request_id: str


class AnalyzeResponse(BaseModel):
    prediction: PredictionResponse
    explanation: ExplanationResponse
    asfr_gate: GateResponse
    request_id: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = InferenceEngine()
    engine.load()
    explainer = GradCAMExplainer(engine)

    app.state.inference_engine = engine
    app.state.xai_explainer = explainer
    logger.info("Model and XAI explainer loaded once for this process")

    try:
        yield
    finally:
        app.state.xai_explainer = None
        app.state.inference_engine = None


app = FastAPI(
    title="ASFR Rice Disease API",
    description=(
        "Image-level rice disease classification using the saved ASFR model. "
        "Grad-CAM and the learned ASFR spatial gate are returned separately."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied_id = request.headers.get("X-Request-ID", "").strip()
    request_id = (
        supplied_id
        if supplied_id and len(supplied_id) <= 128
        else str(uuid.uuid4())
    )
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request error request_id=%s path=%s",
            request_id,
            request.url.path,
        )
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error.",
                "request_id": request_id,
            },
        )

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
        headers=exc.headers,
    )


async def _read_image_upload(file: UploadFile) -> bytes:
    content = await file.read(MAX_IMAGE_BYTES + 1)

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MiB upload limit.",
        )

    try:
        with Image.open(BytesIO(content)) as image:
            if image.format not in {"PNG", "JPEG", "BMP", "GIF", "WEBP"}:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format.",
                )

            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image exceeds the pixel-count limit.",
                )

            image.verify()

    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Upload is not a valid supported image.",
        ) from exc

    return content


def _png_base64(image_array: np.ndarray, mode: str) -> str:
    buffer = BytesIO()
    Image.fromarray(image_array, mode=mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _get_engine(request: Request) -> InferenceEngine:
    engine = getattr(request.app.state, "inference_engine", None)

    if engine is None or engine.model is None:
        raise HTTPException(
            status_code=503,
            detail="Inference model is not ready.",
        )

    return engine


def _get_explainer(request: Request) -> GradCAMExplainer:
    explainer = getattr(request.app.state, "xai_explainer", None)

    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Explanation service is not ready.",
        )

    return explainer


def _prediction_response(
    result: dict[str, Any],
    request_id: str,
    filename: str | None = None,
    content_type: str | None = None,
) -> PredictionResponse:
    required_keys = {
        "predicted_class",
        "predicted_class_index",
        "predicted_probability",
        "probabilities",
        "model_version",
    }

    if not required_keys.issubset(result):
        raise InferenceError(
            "Inference result does not match the expected response contract."
        )

    probability_records = result["probabilities"]
    if not isinstance(probability_records, list):
        raise InferenceError("Class probabilities must be a list.")

    return PredictionResponse(
        predicted_class=str(result["predicted_class"]),
        predicted_class_index=int(result["predicted_class_index"]),
        confidence=float(result["predicted_probability"]),
        probabilities=probability_records,
        model_version=str(result["model_version"]),
        request_id=request_id,
        filename=filename,
        content_type=content_type,
    )


def _run_prediction(
    request: Request,
    image_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> PredictionResponse:
    engine = _get_engine(request)

    try:
        with MODEL_EXECUTION_LOCK:
            result = engine.predict(image_bytes)

        return _prediction_response(
            result,
            request_id=request.state.request_id,
            filename=filename,
            content_type=content_type,
        )

    except ModelLoadError as exc:
        raise HTTPException(
            status_code=503,
            detail="Inference model is not ready.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Image could not be processed.",
        ) from exc
    except InferenceError as exc:
        logger.warning(
            "Prediction failed request_id=%s error=%s",
            request.state.request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail="Prediction failed for this image.",
        ) from exc


def _run_gradcam(
    request: Request,
    image_bytes: bytes,
    target_class_index: int | None,
    overlay_alpha: float,
) -> ExplanationResponse:
    explainer = _get_explainer(request)

    try:
        with MODEL_EXECUTION_LOCK:
            result = explainer.explain(
                image_bytes,
                target_class_index=target_class_index,
                overlay_alpha=overlay_alpha,
            )

        heatmap = np.rint(
            np.clip(result["heatmap"], 0.0, 1.0) * 255.0
        ).astype(np.uint8)

        return ExplanationResponse(
            explained_class_index=int(result["explained_class_index"]),
            explained_class=str(result["explained_class"]),
            class_probability=float(result["class_probability"]),
            method=str(result["method"]),
            target_layer=str(result["target_layer"]),
            model_version=str(result["model_version"]),
            warning=result.get("warning"),
            metadata=result.get("metadata", {}),
            heatmap_png_base64=_png_base64(heatmap, mode="L"),
            overlay_png_base64=_png_base64(
                result["overlay_rgb"], mode="RGB"
            ),
            request_id=request.state.request_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid explanation parameters or image.",
        ) from exc
    except XAIError as exc:
        logger.warning(
            "Grad-CAM failed request_id=%s error=%s",
            request.state.request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail="Grad-CAM could not be generated for this image.",
        ) from exc


def _run_gate(request: Request, image_bytes: bytes) -> GateResponse:
    explainer = _get_explainer(request)

    try:
        with MODEL_EXECUTION_LOCK:
            result = explainer.extract_asfr_gate(image_bytes)

        gate_image = np.rint(
            np.clip(result["gate_map"], 0.0, 1.0) * 255.0
        ).astype(np.uint8)

        return GateResponse(
            method=str(result["method"]),
            target_layer=str(result["target_layer"]),
            model_version=str(result["model_version"]),
            warning=result.get("warning"),
            metadata=result.get("metadata", {}),
            gate_png_base64=_png_base64(gate_image, mode="L"),
            request_id=request.state.request_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image for ASFR gate extraction.",
        ) from exc
    except XAIError as exc:
        logger.warning(
            "ASFR gate failed request_id=%s error=%s",
            request.state.request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail="ASFR spatial gate could not be extracted.",
        ) from exc


async def _run_with_timeout(
    func,
    *args,
    timeout_seconds: float,
    request_id: str,
):
    """Bound HTTP wait time without claiming to cancel TensorFlow work."""
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, partial(func, *args))

    try:
        return await asyncio.wait_for(
            asyncio.shield(future),
            timeout=max(0.0, timeout_seconds),
        )
    except asyncio.TimeoutError as exc:
        logger.warning(
            "model_operation_timeout request_id=%s timeout_seconds=%.2f",
            request_id,
            timeout_seconds,
        )

        # The worker may still be running. Consume its eventual exception
        # to avoid an unobserved exception after the request has timed out.
        def consume_late_result(done_future):
            if not done_future.cancelled():
                try:
                    done_future.exception()
                except Exception:
                    pass

        future.add_done_callback(consume_late_result)

        raise HTTPException(
            status_code=504,
            detail="Model operation timed out. Please retry.",
        ) from exc


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", stage="inference-xai-api")


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request):
    engine = getattr(request.app.state, "inference_engine", None)
    explainer = getattr(request.app.state, "xai_explainer", None)
    is_ready = (
        engine is not None
        and engine.model is not None
        and explainer is not None
    )

    return ReadyResponse(
        ready=is_ready,
        model_version=engine.model_version if is_ready else None,
        xai_ready=explainer is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request):
    engine = _get_engine(request)
    model = engine.model

    return ModelInfoResponse(
        model_version=str(engine.model_version),
        input_shape=list(model.input_shape),
        output_shape=list(model.output_shape),
        class_names=list(engine.class_names),
        class_count=len(engine.class_names),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image_bytes = await _read_image_upload(file)

    return await _run_with_timeout(
        _run_prediction,
        request,
        image_bytes,
        file.filename,
        file.content_type,
        timeout_seconds=MODEL_OPERATION_TIMEOUT_SECONDS,
        request_id=request.state.request_id,
    )


@app.post("/explain", response_model=ExplanationResponse)
async def explain(
    request: Request,
    file: UploadFile = File(...),
    target_class_index: int | None = Query(
        default=None, ge=0, le=MAX_TARGET_CLASS
    ),
    overlay_alpha: float = Query(default=0.40, ge=0.0, le=1.0),
):
    image_bytes = await _read_image_upload(file)

    return await _run_with_timeout(
        _run_gradcam,
        request,
        image_bytes,
        target_class_index,
        overlay_alpha,
        timeout_seconds=MODEL_OPERATION_TIMEOUT_SECONDS,
        request_id=request.state.request_id,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    target_class_index: int | None = Query(
        default=None, ge=0, le=MAX_TARGET_CLASS
    ),
    overlay_alpha: float = Query(default=0.40, ge=0.0, le=1.0),
):
    image_bytes = await _read_image_upload(file)
    deadline = time.monotonic() + MODEL_OPERATION_TIMEOUT_SECONDS

    prediction = await _run_with_timeout(
        _run_prediction,
        request,
        image_bytes,
        file.filename,
        file.content_type,
        timeout_seconds=deadline - time.monotonic(),
        request_id=request.state.request_id,
    )

    explanation = await _run_with_timeout(
        _run_gradcam,
        request,
        image_bytes,
        target_class_index,
        overlay_alpha,
        timeout_seconds=deadline - time.monotonic(),
        request_id=request.state.request_id,
    )

    gate = await _run_with_timeout(
        _run_gate,
        request,
        image_bytes,
        timeout_seconds=deadline - time.monotonic(),
        request_id=request.state.request_id,
    )

    return AnalyzeResponse(
        prediction=prediction,
        explanation=explanation,
        asfr_gate=gate,
        request_id=request.state.request_id,
    )


