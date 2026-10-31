
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = Path(
    r"C:\Users\Menuo\Desktop\KBK\Papers\Capstone_Research"
    r"\Dataset\archive\Rice_Leaf_AUG\Bacterial Leaf Blight"
    r"\20231006_164208.jpg"
)
MASK_PATH = ROOT / "backend" / "tests" / "assets" / "leaf_background_mask.png"
MODEL_PATH = ROOT / "model_artifacts" / "best_model.keras"

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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_image(rgb):
    resized = Image.fromarray(rgb).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32)
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def main():
    for path in (IMAGE_PATH, MASK_PATH, MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    original = np.asarray(Image.open(IMAGE_PATH).convert("RGB"))
    mask_image = Image.open(MASK_PATH).convert("L")
    mask = np.asarray(mask_image)

    print("Original dimensions (W,H):",
          (original.shape[1], original.shape[0]))
    print("Mask dimensions (W,H):", mask_image.size)

    if mask_image.size != (original.shape[1], original.shape[0]):
        raise ValueError(
            "Mask dimensions do not match original image. "
            "Do not resize a misaligned mask automatically."
        )

    # Accept only black background and white leaf labels.
    unique = np.unique(mask)
    if not np.isin(unique, [0, 255]).all():
        raise ValueError(
            f"Mask must be binary (0/255). Values found: {unique[:20]}"
        )

    leaf = mask == 255
    background = ~leaf
    leaf_fraction = float(leaf.mean())

    print("Leaf pixel fraction:", round(leaf_fraction, 4))
    print("Background pixel fraction:", round(float(background.mean()), 4))

    # Reject masks that are clearly empty or cover nearly the whole image.
    if leaf_fraction < 0.01 or leaf_fraction > 0.95:
        raise ValueError(
            "Mask coverage is implausible. Inspect the mask before testing."
        )

    checkpoint_hash_before = sha256(MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    def predict(rgb):
        probabilities = np.asarray(
            model(prepare_image(rgb), training=False)
        )[0]
        if probabilities.shape != (len(CLASS_NAMES),):
            raise ValueError(
                f"Unexpected model output shape: {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("Model returned non-finite probabilities.")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
            raise ValueError(
                f"Probabilities do not sum to 1: {probabilities.sum()}"
            )
        return probabilities

    # Baseline: original pixels.
    baseline = predict(original)
    target_index = int(np.argmax(baseline))

    # Variant A: replace background with neutral gray, keep leaf pixels.
    gray_variant = original.copy()
    gray_variant[background] = [127, 127, 127]
    gray_probs = predict(gray_variant)

    # Variant B: replace background with a different neutral color.
    alternate_variant = original.copy()
    alternate_variant[background] = [200, 200, 200]
    alternate_probs = predict(alternate_variant)

    # Control: replace the annotated leaf region, keep background.
    leaf_control = original.copy()
    leaf_control[leaf] = [127, 127, 127]
    control_probs = predict(leaf_control)

    checkpoint_hash_after = sha256(MODEL_PATH)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError("Checkpoint hash changed during the test.")

    def describe(name, probs):
        pred = int(np.argmax(probs))
        return {
            "variant": name,
            "predicted_class": CLASS_NAMES[pred],
            "predicted_index": pred,
            "predicted_probability": float(probs[pred]),
            "baseline_target_probability": float(probs[target_index]),
            "baseline_target_probability_change": float(
                probs[target_index] - baseline[target_index]
            ),
            "top_class_changed": bool(pred != target_index),
        }

    results = {
        "image": str(IMAGE_PATH),
        "mask": str(MASK_PATH),
        "mask_leaf_fraction": leaf_fraction,
        "baseline_target_class": CLASS_NAMES[target_index],
        "baseline": describe("original", baseline),
        "background_gray": describe("background_gray_127", gray_probs),
        "background_light_gray": describe(
            "background_light_gray_200", alternate_probs
        ),
        "leaf_region_control": describe(
            "leaf_region_gray_control", control_probs
        ),
        "background_gray_probability_vector": gray_probs.tolist(),
        "background_light_gray_probability_vector": alternate_probs.tolist(),
        "leaf_control_probability_vector": control_probs.tolist(),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_unchanged": True,
        "interpretation_note": (
            "Single-image diagnostic only. Mask quality and the chosen "
            "replacement colors affect the result; this is not causal proof."
        ),
    }

    output_path = ROOT / "backend" / "tests" / "background_dependence_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nBACKGROUND-DEPENDENCE RESULTS")
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
