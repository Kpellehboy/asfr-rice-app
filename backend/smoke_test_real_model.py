
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from app.inference import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else None
MODEL_PATH = ROOT / "model_artifacts" / "best_model.keras"
CLASS_NAMES_PATH = ROOT / "model_artifacts" / "class_names.json"
METADATA_PATH = ROOT / "model_artifacts" / "model_metadata.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if IMAGE_PATH is None:
        raise SystemExit(
            "Usage: python backend/smoke_test_real_model.py "
            "test_images/rice_leaf.jpg"
        )

    image_path = IMAGE_PATH.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    print("Loading actual saved checkpoint...")
    engine = InferenceEngine(
        model_path=MODEL_PATH,
        class_names_path=CLASS_NAMES_PATH,
        metadata_path=METADATA_PATH,
    )
    engine.load()

    assert engine.model is not None
    assert len(engine.class_names) == 9
    assert tuple(engine.model.input_shape[1:]) == (224, 224, 3)
    assert tuple(engine.model.output_shape[1:]) == (9,)

    weights_before = [
        np.array(weight, copy=True)
        for weight in engine.model.get_weights()
    ]
    checkpoint_hash_before = sha256(MODEL_PATH)

    result = engine.predict(image_path.read_bytes())

    probabilities = result["probabilities"]
    assert len(probabilities) == 9

    values = np.asarray(
        [item["probability"] for item in probabilities],
        dtype=np.float64,
    )
    assert np.isfinite(values).all()
    assert np.all(values >= 0.0) and np.all(values <= 1.0)
    assert np.isclose(values.sum(), 1.0, atol=1e-5)

    predicted_index = result["predicted_class_index"]
    assert 0 <= predicted_index < 9
    assert result["predicted_class"] == engine.class_names[predicted_index]
    assert probabilities[predicted_index]["class_name"] == result["predicted_class"]
    assert np.isclose(
        result["predicted_probability"],
        values[predicted_index],
        atol=1e-7,
    )

    weights_after = engine.model.get_weights()
    assert len(weights_before) == len(weights_after)
    assert all(
        np.array_equal(before, after)
        for before, after in zip(weights_before, weights_after)
    )
    assert sha256(MODEL_PATH) == checkpoint_hash_before

    print("\nREAL-CHECKPOINT SMOKE TEST PASSED")
    print(f"Checkpoint SHA-256: {checkpoint_hash_before}")
    print(f"Model version: {result['model_version']}")
    print(f"Image: {image_path.name}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Predicted index: {predicted_index}")
    print(f"Predicted probability: {result['predicted_probability']:.6f}")
    print(f"Probability sum: {values.sum():.8f}")
    print("\nAll class probabilities:")
    print(json.dumps(probabilities, indent=2))
    print("\nModel weights unchanged: PASS")
    print("Checkpoint file unchanged: PASS")


if __name__ == "__main__":
    main()
