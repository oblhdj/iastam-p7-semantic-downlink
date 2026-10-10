"""demo/quickstart: the CPU-only demo must stay faithful to the repo it summarises.

Pins: the ONNX decode follows ultralytics' rules (what wp1_predictions.csv was written with); the
bundled tiles are the synthetic ones the manifest names and regenerate from their generator; the
packet uses sat7.priority unchanged (P0-P3 levels, WP4 sizes, one shared coastal context tile);
the reproducibility cross-check counts decision flips correctly. The end-to-end run needs
onnxruntime, which the analysis .venv does not carry -- it skips there (install
demo/quickstart/requirements.txt to run it).
"""
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
QS = REPO / "demo" / "quickstart"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"qs_{name}", QS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod               # @dataclass resolves its module through sys.modules
    spec.loader.exec_module(mod)
    return mod


qs = _load("run_demo")


def _head(boxes):
    """A fake single-class YOLOv8 head output, shape (5, N), from (cx, cy, w, h, score) rows."""
    return np.array(boxes, np.float32).T


# ------------------------------------------------------------------------------ decode

def test_decode_is_strict_on_conf_and_suppresses_above_iou():
    out = _head([(100, 100, 20, 20, 0.9), (101, 100, 20, 20, 0.8),     # IoU ~0.9 -> suppressed
                 (300, 300, 10, 10, 0.05), (400, 400, 10, 10, 0.06)])   # 0.05 is not > 0.05
    dets = qs.decode(out, conf=0.05, iou=0.7, w=768, h=768)
    assert [round(d[4], 2) for d in dets] == [0.9, 0.06]


def test_decode_clips_boxes_to_the_image_like_ultralytics():
    (cx, cy, w, h, _), = qs.decode(_head([(5, 700, 20, 140, 0.5)]), 0.05, 0.7, 768, 768)
    assert (cx, w) == pytest.approx((7.5, 15.0))       # x0 -5 -> 0
    assert (cy, h) == pytest.approx((699.0, 138.0))    # y1 770 -> 768


def test_nms_keeps_a_box_at_exactly_the_threshold():
    # torchvision suppresses only IoU > thr; two 10x10 boxes offset to IoU exactly 1/3
    xyxy = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], np.float32)
    keep = qs.nms_xyxy(xyxy, np.array([0.9, 0.8], np.float32), thr=1 / 3)
    assert len(keep) == 2


# ------------------------------------------------------------------------------ bundled tiles

def _manifest():
    return json.loads((QS / "tiles" / "manifest.json").read_text())


def test_bundled_tiles_are_the_synthetic_ones_in_the_manifest():
    m = _manifest()
    assert len(m["tiles"]) == 9 and all(e["synthetic"] for e in m["tiles"])
    for e in m["tiles"]:
        assert hashlib.sha256((QS / "tiles" / e["file"]).read_bytes()).hexdigest() == e["sha256"]


def test_stand_in_ids_are_committed_test_tiles_with_matching_context():
    with (REPO / "code" / "results" / "wp6_tiles.csv").open(newline="") as f:
        ctx = {r["image"]: r["context"] for r in csv.DictReader(f)}
    for e in _manifest()["tiles"]:
        assert ctx[e["stands_in_for"]] == e["reference_context"]


def test_generator_reproduces_the_bundled_tiles(tmp_path):
    gen = _load("make_synthetic_tiles")
    gen.OUT = tmp_path
    gen.main()
    for e in _manifest()["tiles"]:
        a = cv2.imread(str(QS / "tiles" / e["file"])).astype(np.int16)
        b = cv2.imread(str(tmp_path / e["file"])).astype(np.int16)
        # byte-identical on the machine that made them; allow JPEG/SIMD jitter elsewhere
        assert np.abs(a - b).mean() < 0.5, e["file"]


def test_synthetic_tiles_get_the_reference_prefilter_context():
    from sat7.prefilter import run_prefilter
    for e in _manifest()["tiles"]:
        img = cv2.imread(str(QS / "tiles" / e["file"]))
        assert run_prefilter(img).context == e["reference_context"], e["file"]


# ------------------------------------------------------------------------------ packet

def test_packet_uses_sat7_priority_levels_and_sizes():
    from sat7.priority import P0, P1, P2, P3, PriorityConfig, encode_priority
    from sat7.scheduler import COAST_TILE_BYTES_MEASURED, LoDConfig

    lod = LoDConfig(conf_low=qs.DET_THR, size_model=qs.load_size_model())
    pcfg = PriorityConfig(p1_conf=lod.conf_high)
    dets = [[(10, 10, 40, 8, 0.9), (50, 50, 20, 10, 0.5), (90, 90, 6, 6, 0.1)],   # ships
            [(10, 10, 60, 20, 0.9), (40, 40, 30, 12, 0.4)],                        # coast
            [(10, 10, 30, 30, 0.9)]]                                                # cloud
    ctxs = ["ships", "coast", "cloud"]
    wl, tile_of = qs.build_workload(ctxs, dets)
    levels = [qs.level_of(s, ctxs[tile_of[i]], pcfg, lod) for i, s in enumerate(wl.ships)]
    # sat7.priority escalates EVERY above-threshold coastal detection to P3, confident ones too
    assert levels == [P1, P2, P0, P3, P3, P0]
    items = encode_priority(wl, lod, pcfg)
    by = {k: [it for it in items if it.kind == k] for k in ("P1", "P2", "P3")}
    assert len(by["P1"]) == 4 and all(it.size == lod.l0_bytes for it in by["P1"])
    assert sorted(it.size for it in by["P2"]) == sorted(
        lod.size_model.l1(L) for L in (20.0, 60.0, 30.0))
    # one shared context tile; a hand-built workload carries no per-tile size -> the measured mean
    assert [it.size for it in by["P3"]] == [COAST_TILE_BYTES_MEASURED]
    assert not any(tile_of[it.ships[0]] == 2 for it in items)          # cloud tile sends nothing


def test_cross_check_counts_threshold_flips_and_unpaired_boxes():
    live = [(100, 100, 20, 20, 0.26), (500, 500, 20, 20, 0.30)]
    ref = [(100, 100, 20, 20, 0.24)]
    c = qs.cross_check(live, ref, {"det_thr": 0.25})
    assert c["paired"] == 1 and c["max_dconf"] == pytest.approx(0.02)
    assert c["flips"]["det_thr"] == 2          # one crossing + one unpaired box above the cut


# ------------------------------------------------------------------------------ end to end

def test_end_to_end_on_bundled_tiles(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--out", str(tmp_path)])
    assert qs.main() == 0
    pkt = json.loads((tmp_path / "packet.json").read_text())
    assert pkt["label"] == "SYNTH" and pkt["synthetic_tiles"] is True
    assert {d["level"] for d in pkt["detections"]} == {0, 1, 2, 3}
    assert 0 < pkt["semantic_bytes_measured"] < pkt["raw_bytes"]
    # the packet total is the size of the real serialized stream, not an estimate
    assert (tmp_path / "downlink.bin").stat().st_size == pkt["semantic_bytes_measured"]
    assert (tmp_path / "swath_annotated.jpg").exists()
