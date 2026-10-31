
from pathlib import Path
from io import BytesIO
import hashlib

import numpy as np
from PIL import Image

from app.inference import InferenceEngine
from app.preprocessing import preprocess_image
from app.xai import GradCAMExplainer


ROOT = Path(__file__).resolve().parents[2]

REAL_IMAGE = Path(
    r"C:\Users\Menuo\Desktop\KBK\Papers\Capstone_Research"
    r"\Dataset\archive\Rice_Leaf_AUG\Bacterial Leaf Blight"
    r"\20231006_164208.jpg"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_to_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def predict_probabilities(engine, image_bytes):
    tensor = np.asarray(preprocess_image(image_bytes), dtype=np.float32)
    predictions = engine.model.predict(tensor[None, ...], verbose=0)
    probabilities = np.asarray(predictions[0], dtype=np.float32)

    assert probabilities.shape == (9,)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-4)

    return probabilities


def make_region_mask(image_rgb, region_mask):
    """Replace the selected region with neutral mid-gray."""
    result = image_rgb.copy()
    result[region_mask] = np.array([127, 127, 127], dtype=np.uint8)
    return result


def test_gradcam_region_mask_vs_control_region():
    assert REAL_IMAGE.is_file(), f"Real image not found: {REAL_IMAGE}"

    model_path = ROOT / "model_artifacts" / "best_model.keras"
    class_names_path = ROOT / "model_artifacts" / "class_names.json"
    metadata_path = ROOT / "model_artifacts" / "model_metadata.json"

    checkpoint_hash_before = sha256_file(model_path)

    engine = InferenceEngine(
        model_path=model_path,
        class_names_path=class_names_path,
        metadata_path=metadata_path,
    )
    engine.load()
    explainer = GradCAMExplainer(engine)

    image_bytes = REAL_IMAGE.read_bytes()

    with Image.open(BytesIO(image_bytes)) as source:
        original_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)

    height, width = original_rgb.shape[:2]

    explanation = explainer.explain(image_bytes)
    heatmap = explanation["heatmap"]
    target_class = explanation["explained_class_index"]

    assert heatmap.shape == (height, width)
    assert np.isfinite(heatmap).all()

    # Compare equal-area regions: top 20% of pixels by heatmap score
    # versus bottom 20%, with deterministic tie handling.
    flat_heatmap = heatmap.reshape(-1)
    pixel_count = flat_heatmap.size
    region_count = max(1, int(round(pixel_count * 0.20)))

    order = np.argsort(flat_heatmap, kind="stable")
    control_indices = order[:region_count]
    highlighted_indices = order[-region_count:]

    highlighted_mask = np.zeros(pixel_count, dtype=bool)
    control_mask = np.zeros(pixel_count, dtype=bool)
    highlighted_mask[highlighted_indices] = True
    control_mask[control_indices] = True

    highlighted_mask = highlighted_mask.reshape(height, width)
    control_mask = control_mask.reshape(height, width)

    # Avoid a degenerate comparison if the heatmap is constant.
    assert not np.allclose(heatmap.max(), heatmap.min()), (
        "Grad-CAM heatmap is constant; region masking is not informative."
    )

    original_probabilities = predict_probabilities(engine, image_bytes)

    highlighted_image = make_region_mask(original_rgb, highlighted_mask)
    control_image = make_region_mask(original_rgb, control_mask)

    highlighted_bytes = image_to_bytes(Image.fromarray(highlighted_image))
    control_bytes = image_to_bytes(Image.fromarray(control_image))

    highlighted_probabilities = predict_probabilities(
        engine, highlighted_bytes
    )
    control_probabilities = predict_probabilities(engine, control_bytes)

    original_score = float(original_probabilities[target_class])
    highlighted_score = float(highlighted_probabilities[target_class])
    control_score = float(control_probabilities[target_class])

    highlighted_drop = original_score - highlighted_score
    control_drop = original_score - control_score

    print("\nGrad-CAM region-masking results")
    print(f"Image: {REAL_IMAGE.name}")
    print(f"Explained class: {explanation['explained_class']}")
    print(f"Target class index: {target_class}")
    print(f"Original target probability: {original_score:.6f}")
    print(f"Highlighted-region probability: {highlighted_score:.6f}")
    print(f"Control-region probability: {control_score:.6f}")
    print(f"Highlighted-region score drop: {highlighted_drop:.6f}")
    print(f"Control-region score drop: {control_drop:.6f}")
    print(f"Difference in score drops: {highlighted_drop - control_drop:.6f}")

    # Sanity assertions: valid probabilities and equal-sized disjoint masks.
    assert np.isfinite(
        [original_score, highlighted_score, control_score]
    ).all()
    assert highlighted_mask.sum() == control_mask.sum()
    assert not np.any(highlighted_mask & control_mask)

    # Do not require the highlighted region to win: that is the empirical
    # result being measured, not a condition to force the test to pass.
    assert sha256_file(model_path) == checkpoint_hash_before
