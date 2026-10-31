import numpy as np
import pytest

from app.inference import (
    InferenceEngine,
    InferenceError,
    ModelLoadError,
)


CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Narrow Brown Leaf Spot",
    "Neck_Blast",
    "Rice Hispa",
    "Sheath Blight",
]


class FakeModel:
    input_shape = (None, 224, 224, 3)
    output_shape = (None, 9)

    def __init__(self):
        self.weights = [np.array([1.0, 2.0], dtype=np.float32)]
        self.calls = 0

    def __call__(self, batch, training=False):
        assert training is False
        assert batch.shape == (1, 224, 224, 3)
        self.calls += 1

        output = np.array(
            [[0.01, 0.02, 0.80, 0.03, 0.02, 0.02, 0.03, 0.04, 0.03]],
            dtype=np.float32,
        )
        return output


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(
        "app.inference.preprocess_image",
        lambda image_bytes: np.zeros(
            (224, 224, 3), dtype=np.float32
        ),
    )

    instance = InferenceEngine()
    instance.model = FakeModel()
    instance.class_names = CLASS_NAMES.copy()
    instance.model_version = "test-model"
    return instance


def test_prediction_shape_and_probability_validity(engine):
    result = engine.predict(b"valid-image-placeholder")

    assert len(result["probabilities"]) == 9
    values = np.array(
        [item["probability"] for item in result["probabilities"]]
    )

    assert np.isfinite(values).all()
    assert np.all((values >= 0) & (values <= 1))
    assert np.isclose(values.sum(), 1.0, atol=1e-5)


def test_predicted_class_and_index_mapping(engine):
    result = engine.predict(b"image")

    assert result["predicted_class_index"] == 2
    assert result["predicted_class"] == "Healthy Rice Leaf"
    assert result["predicted_probability"] == pytest.approx(0.80)

    assert [
        item["class_name"] for item in result["probabilities"]
    ] == CLASS_NAMES


def test_multiple_images_return_consistent_predictions(engine):
    first = engine.predict(b"image-one")
    second = engine.predict(b"image-two")

    assert first == second
    assert engine.model.calls == 2


def test_inference_does_not_modify_weights(engine):
    before = [weight.copy() for weight in engine.model.weights]

    engine.predict(b"image")

    for original, current in zip(before, engine.model.weights):
        np.testing.assert_array_equal(original, current)


@pytest.mark.parametrize("invalid_input", [b"", None, "not-bytes", 123])
def test_invalid_inputs_fail_safely(engine, invalid_input):
    with pytest.raises((ValueError, ModelLoadError)):
        engine.predict(invalid_input)


def test_unloaded_engine_rejects_prediction():
    instance = InferenceEngine()

    with pytest.raises(ModelLoadError):
        instance.predict(b"image")