import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "model_artifacts"

def test_class_mapping_has_nine_unique_labels():
    data = json.loads((ARTIFACTS / "class_names.json").read_text(encoding="utf-8"))
    names = data["class_names"]
    assert len(names) == 9
    assert len(set(names)) == 9

def test_model_checksum_matches_metadata():
    meta = json.loads((ARTIFACTS / "model_metadata.json").read_text(encoding="utf-8"))
    path = ARTIFACTS / meta["artifact"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == meta["artifact_sha256"]

def test_model_metadata_shape_and_class_count():
    meta = json.loads((ARTIFACTS / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["input"]["shape"] == [None, 224, 224, 3]
    assert meta["output"]["shape"] == [None, 9]
    assert meta["output"]["classes"] == 9
