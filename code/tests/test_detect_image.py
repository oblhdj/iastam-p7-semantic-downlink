"""scripts/detect_image.py: the single-image path runs end to end and labels its evaluation honestly."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "scripts"))


@pytest.mark.inference
def test_single_image_without_ground_truth(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    import detect_image
    tile = REPO / "demo" / "quickstart" / "tiles" / "03_synthetic_00f34434e.jpg"
    if detect_image.find_weights() is None or not tile.exists():
        pytest.skip("weights or quickstart tiles not present")
    monkeypatch.setattr(sys, "argv", ["detect_image.py", "--image", str(tile), "--out", str(tmp_path)])
    assert detect_image.main() == 0
    rec = json.loads((tmp_path / f"{tile.stem}.json").read_text())
    assert rec["mode"] == "whole" and rec["input"]["raw_bytes"] == 768 * 768 * 3
    assert rec["evaluation"]["ground_truth"] is False            # counts only, never "accuracy"
    assert rec["detections"] and (tmp_path / f"{tile.stem}_detections.png").exists()


def test_rejects_a_non_image(tmp_path, monkeypatch):
    import detect_image
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"not an image")
    monkeypatch.setattr(sys, "argv", ["detect_image.py", "--image", str(bad), "--out", str(tmp_path)])
    assert detect_image.main() == 2
