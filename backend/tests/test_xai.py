
import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from app.inference import InferenceEngine
from app.xai import GradCAMExplainer, XAIError


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "model_artifacts" / "best_model.keras"
CLASS_NAMES_PATH = ROOT / "model_artifacts" / "class_names.json"
METADATA_PATH = ROOT / "model_artifacts" / "model_metadata.json"

EXPECTED_TARGET_LAYER = "asfr_fusion_relu"
EXPECTED_GATE_LAYER = "asfr_spatial_gate"
EXPECTED_CLASS_COUNT = 9


# --------------------------------------------------
# Fixtures and utilities
# --------------------------------------------------

@pytest.fixture(scope="module")
def explainer():
    engine = InferenceEngine(
        model_path=CHECKPOINT_PATH,
        class_names_path=CLASS_NAMES_PATH,
        metadata_path=METADATA_PATH,
    )
    engine.load()
    return GradCAMExplainer(engine)


def make_image(seed: int, width: int = 320, height: int = 240) -> bytes:
    """Generate a deterministic RGB PNG for XAI testing."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )

    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------
# Existing real-checkpoint Grad-CAM tests
# --------------------------------------------------

def test_real_checkpoint_gradcam_gradients_and_alignment(explainer):
    result = explainer.explain(make_image(11), target_class_index=0)

    assert result["method"] == "Grad-CAM"
    assert result["target_layer"] == EXPECTED_TARGET_LAYER
    assert result["explained_class_index"] == 0
    assert result["explained_class"] == explainer.class_names[0]

    heatmap = result["heatmap"]
    overlay = result["overlay_rgb"]

    assert heatmap.shape == (240, 320)
    assert overlay.shape == (240, 320, 3)
    assert heatmap.dtype == np.float32
    assert overlay.dtype == np.uint8
    assert np.isfinite(heatmap).all()
    assert 0.0 <= float(heatmap.min()) <= 1.0
    assert 0.0 <= float(heatmap.max()) <= 1.0
    assert result["metadata"]["gradient_max_abs"] > 1e-12


def test_multiple_target_classes_and_images(explainer):
    first = make_image(21)
    second = make_image(22)

    class_zero = explainer.explain(first, target_class_index=0)
    class_two = explainer.explain(first, target_class_index=2)
    other_image = explainer.explain(second, target_class_index=0)

    assert class_zero["explained_class_index"] == 0
    assert class_two["explained_class_index"] == 2
    assert class_zero["heatmap"].shape == class_two["heatmap"].shape
    assert other_image["heatmap"].shape == (240, 320)

    assert not np.allclose(
        class_zero["heatmap"], class_two["heatmap"], atol=1e-6
    )
    assert not np.allclose(
        class_zero["heatmap"], other_image["heatmap"], atol=1e-6
    )


# --------------------------------------------------
# OOD feature embedding tests
# --------------------------------------------------

def test_ood_embedding_shape_dtype_and_metadata(explainer):
    result = explainer.extract_ood_embedding(make_image(601))

    embedding = result["embedding"]

    assert embedding.shape == (256,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()

    assert result["target_layer"] == EXPECTED_TARGET_LAYER
    assert result["model_version"] == explainer.model_version

    metadata = result["metadata"]
    assert metadata["feature_map_shape"] == [14, 14, 256]
    assert metadata["embedding_dimension"] == 256
    assert metadata["pooling"] == "global_average_pooling"
    assert metadata["feature_extractor"] == "asfr_ood_feature_extractor"


def test_ood_embedding_is_repeatable(explainer):
    image_bytes = make_image(602)

    first = explainer.extract_ood_embedding(image_bytes)
    second = explainer.extract_ood_embedding(image_bytes)

    np.testing.assert_allclose(
        first["embedding"],
        second["embedding"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_ood_embedding_changes_for_different_inputs(explainer):
    first = explainer.extract_ood_embedding(make_image(603))
    second = explainer.extract_ood_embedding(make_image(604))

    assert not np.allclose(
        first["embedding"],
        second["embedding"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_ood_embedding_rejects_invalid_image(explainer):
    with pytest.raises(ValueError):
        explainer.extract_ood_embedding(b"not an image")

    with pytest.raises(ValueError):
        explainer.extract_ood_embedding(b"")


def test_asfr_gate_is_separate_and_aligned(explainer):
    result = explainer.extract_asfr_gate(make_image(31))

    assert result["method"] == "ASFR spatial gate"
    assert result["target_layer"] == EXPECTED_GATE_LAYER
    assert result["metadata"]["class_specific"] is False
    assert result["gate_map"].shape == (240, 320)
    assert result["gate_map"].dtype == np.float32
    assert np.isfinite(result["gate_map"]).all()
    assert result["gate_map"].min() >= 0.0
    assert result["gate_map"].max() <= 1.0
    assert result["metadata"]["native_gate_shape"] == [14, 14]


def test_invalid_image_fails_safely(explainer):
    with pytest.raises(ValueError):
        explainer.explain(b"not an image")

    with pytest.raises(ValueError):
        explainer.extract_asfr_gate(b"")


@pytest.mark.parametrize("target_class", [-1, 9, 1.5, True, "2"])
def test_invalid_target_class_fails_safely(explainer, target_class):
    with pytest.raises(ValueError):
        explainer.explain(
            make_image(41), target_class_index=target_class
        )


# --------------------------------------------------
# Stage 5.4: initialization and parameter validation
# --------------------------------------------------

def test_unloaded_engine_fails_during_initialization():
    engine = SimpleNamespace(model=None, class_names=[])

    with pytest.raises(XAIError, match="loaded inference engine"):
        GradCAMExplainer(engine)


def test_missing_xai_layer_fails_during_initialization(explainer):
    class ModelWithoutLayers:
        def get_layer(self, name):
            raise ValueError(f"No such layer: {name}")

    engine = SimpleNamespace(
        model=ModelWithoutLayers(),
        class_names=explainer.class_names,
        model_version="test",
    )

    with pytest.raises(
        XAIError, match="initialize XAI models"
    ):
        GradCAMExplainer(engine)


@pytest.mark.parametrize(
    "alpha",
    [-0.01, 1.01, float("nan"), float("inf"), True, "0.4"],
)
def test_invalid_overlay_alpha_fails_safely(explainer, alpha):
    with pytest.raises(ValueError, match="overlay_alpha"):
        explainer.explain(make_image(501), overlay_alpha=alpha)


# --------------------------------------------------
# Stage 5.4: repeated-output stability and integrity
# --------------------------------------------------

def test_repeated_explanations_are_stable(explainer):
    image_bytes = make_image(502)

    first = explainer.explain(image_bytes, target_class_index=0)
    second = explainer.explain(image_bytes, target_class_index=0)

    np.testing.assert_allclose(
        first["heatmap"],
        second["heatmap"],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        first["overlay_rgb"], second["overlay_rgb"]
    )

    assert first["explained_class_index"] == second["explained_class_index"]
    assert first["class_probability"] == pytest.approx(
        second["class_probability"], rel=1e-6, abs=1e-7
    )


def test_nonfinite_and_malformed_maps_are_rejected():
    with pytest.raises(XAIError, match="Explanation map is invalid"):
        GradCAMExplainer._resize_map(
            np.array([[0.0, np.nan]], dtype=np.float32),
            width=20,
            height=10,
        )

    with pytest.raises(XAIError, match="Explanation map is invalid"):
        GradCAMExplainer._resize_map(
            np.zeros((2, 2, 1), dtype=np.float32),
            width=20,
            height=10,
        )


def test_missing_gradients_fail_safely(explainer, monkeypatch):
    def return_no_gradients(self, target, sources, *args, **kwargs):
        return None

    monkeypatch.setattr(
        tf.GradientTape, "gradient", return_no_gradients
    )

    with pytest.raises(XAIError, match="disconnected"):
        explainer.explain(make_image(503), target_class_index=0)


def test_explanation_output_integrity(explainer):
    result = explainer.explain(make_image(504), target_class_index=3)

    assert result["method"] == "Grad-CAM"
    assert result["target_layer"] == EXPECTED_TARGET_LAYER
    assert result["explained_class_index"] == 3
    assert result["explained_class"] == explainer.class_names[3]

    heatmap = result["heatmap"]
    overlay = result["overlay_rgb"]
    metadata = result["metadata"]

    assert heatmap.shape == (240, 320)
    assert overlay.shape == (240, 320, 3)
    assert heatmap.dtype == np.float32
    assert overlay.dtype == np.uint8
    assert np.isfinite(heatmap).all()
    assert np.isfinite(overlay).all()
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0

    assert metadata["input_width"] == metadata["heatmap_width"] == 320
    assert metadata["input_height"] == metadata["heatmap_height"] == 240
    assert 0.0 <= result["class_probability"] <= 1.0


# --------------------------------------------------
# Existing targeted weight-randomization sanity test
# --------------------------------------------------

def test_gradcam_changes_after_target_layer_weight_randomization(explainer):
    """Perturb an upstream layer in memory and restore its exact weights."""
    perturbed_layer = explainer.model.get_layer("asfr_deep_projection")

    checkpoint_hash_before = sha256_file(CHECKPOINT_PATH)
    original_weights = [
        weight.copy() for weight in perturbed_layer.get_weights()
    ]

    assert original_weights, "Perturbation layer contains no weights."

    image_bytes = make_image(73)
    target_class = 0
    baseline_map = None

    try:
        baseline = explainer.explain(
            image_bytes, target_class_index=target_class
        )
        baseline_map = baseline["heatmap"].copy()

        rng = np.random.default_rng(2026)
        randomized_weights = []

        for weight in original_weights:
            if np.issubdtype(weight.dtype, np.floating):
                randomized_weights.append(
                    rng.normal(
                        loc=0.0,
                        scale=0.05,
                        size=weight.shape,
                    ).astype(weight.dtype)
                )
            else:
                randomized_weights.append(weight.copy())

        perturbed_layer.set_weights(randomized_weights)

        randomized = explainer.explain(
            image_bytes, target_class_index=target_class
        )
        randomized_map = randomized["heatmap"]

        mean_absolute_difference = float(
            np.mean(np.abs(baseline_map - randomized_map))
        )

        print(
            "\nRandomization diagnostic:"
            f"\n  layer={perturbed_layer.name}"
            f"\n  mean_absolute_map_difference="
            f"{mean_absolute_difference:.8f}"
        )

        assert np.isfinite(mean_absolute_difference)
        assert mean_absolute_difference > 1e-6, (
            "Grad-CAM did not change sufficiently after randomizing "
            "asfr_deep_projection."
        )

    finally:
        perturbed_layer.set_weights(original_weights)

    assert baseline_map is not None

    restored = explainer.explain(
        image_bytes, target_class_index=target_class
    )

    np.testing.assert_allclose(
        restored["heatmap"],
        baseline_map,
        rtol=1e-5,
        atol=1e-6,
    )

    assert sha256_file(CHECKPOINT_PATH) == checkpoint_hash_before
