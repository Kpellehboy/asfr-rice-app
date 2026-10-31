
from pathlib import Path
import hashlib
import keras

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model_artifacts" / "best_model.keras"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
print("Checkpoint:", MODEL_PATH)
print("SHA-256:", digest)

model = keras.models.load_model(MODEL_PATH, compile=False)

print("\nMODEL INPUT:", model.input_shape)
print("MODEL OUTPUT:", model.output_shape)
print("TOTAL PARAMETERS:", model.count_params())

print("\n=== ALL LAYERS ===")
for index, layer in enumerate(model.layers):
    try:
        output_shape = layer.output.shape
    except Exception:
        output_shape = "unavailable"

    print(
        f"{index:03d} | {layer.name:45s} | "
        f"{layer.__class__.__name__:25s} | {output_shape}"
    )

print("\n=== XAI CANDIDATES ===")
keywords = (
    "conv3_block4_out",
    "conv4_block6_out",
    "asfr",
    "gate",
    "spatial",
    "routing",
    "fusion",
    "upsampl",
    "sigmoid",
    "global_average",
    "predictions",
)

for layer in model.layers:
    name = layer.name.lower()
    if any(keyword in name for keyword in keywords):
        try:
            shape = layer.output.shape
        except Exception:
            shape = "unavailable"
        print(
            f"{layer.name} | {layer.__class__.__name__} | {shape}"
        )
