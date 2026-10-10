import numpy as np
import pytest

from sat7.perception import PerceptionConfig, detect_image, slice_count


def _stub(box_per_crop):
    """A detect_fn that returns the same local box in every crop."""
    return lambda crops: [list(box_per_crop) for _ in crops]


def test_whole_mode_is_one_call_and_passes_boxes_through():
    img = np.zeros((768, 768, 3), np.uint8)
    cfg = PerceptionConfig(mode="whole")
    assert slice_count(768, 768, cfg) == 1
    dets = detect_image(img, _stub([(100, 100, 20, 20, 0.9)]), cfg)
    assert len(dets) == 1 and dets[0][:2] == (100.0, 100.0)


def test_sahi_slice_counts_match_wp16():
    assert slice_count(3072, 3072, PerceptionConfig(mode="sahi", window=768)) == 25
    assert slice_count(3072, 3072, PerceptionConfig(mode="sahi", window=512)) == 64


def test_sahi_global_coords_lift_by_slice_origin():
    # one ship at local (50,50) in every window; the second window starts at x=614 (stride 768*0.8)
    img = np.zeros((768, 1400, 3), np.uint8)
    dets = detect_image(img, _stub([(50, 50, 10, 10, 0.9)]),
                        PerceptionConfig(mode="sahi", window=768, overlap=0.2, nms_iou=0.5))
    xs = sorted(round(d[0]) for d in dets)
    assert 50 in xs                      # the first window's ship stays near x=50 (global)
    assert max(xs) > 600                 # a later window's ship is lifted well to the right


def test_fusion_merges_overlap_duplicate_but_keeps_distinct():
    # two windows both see a ship at the SAME global point -> fuse to 1; a far ship stays separate
    slices_img = np.zeros((768, 820, 3), np.uint8)   # windows at x=0 and x=52 (stride 614 clamped)

    def detect(crops):
        # crop 0 (origin 0): ship at local 400; crop 1 (origin 52): ship at local 348 -> both global 400
        return [[(400, 400, 40, 40, 0.9)], [(348, 400, 40, 40, 0.8)]]
    fused = detect_image(slices_img, detect, PerceptionConfig(mode="sahi", window=768, overlap=0.2))
    assert len(fused) == 1               # the duplicate merged (highest conf kept)
    assert abs(fused[0][4] - 0.9) < 1e-9


def test_without_fusion_ablation_keeps_duplicates():
    img = np.zeros((768, 820, 3), np.uint8)

    def detect(crops):
        return [[(400, 400, 40, 40, 0.9)], [(348, 400, 40, 40, 0.8)]]
    merged = detect_image(img, detect, PerceptionConfig(mode="sahi", window=768, overlap=0.2, fuse=True))
    kept = detect_image(img, detect, PerceptionConfig(mode="sahi", window=768, overlap=0.2, fuse=False))
    assert len(kept) > len(merged)       # the ablation exposes the double count


def test_zero_overlap_ablation_reduces_slice_count():
    overlapped = slice_count(3072, 3072, PerceptionConfig(mode="sahi", window=768, overlap=0.2))
    abutting = slice_count(3072, 3072, PerceptionConfig(mode="sahi", window=768, overlap=0.0))
    assert abutting < overlapped         # no overlap -> fewer windows (and uncovered seams)


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        PerceptionConfig(mode="nope")
    with pytest.raises(ValueError):
        PerceptionConfig(overlap=1.0)


# ---------------------------------------------------------------- audit B2 fixes + global coordinates
def test_whole_mode_returns_the_detector_output_untouched():
    # two distinct ships the detector kept (IoU 0.67, under its own 0.7 NMS) must both survive:
    # before the fix a second NMS at 0.5 merged them, so "whole" != the raw B1 detector output
    img = np.zeros((768, 768, 3), np.uint8)
    dets = detect_image(img, _stub([(100, 100, 40, 40, 0.9), (108, 100, 40, 40, 0.8)]),
                        PerceptionConfig(mode="whole"))
    assert len(dets) == 2


def _bright(crops):
    """Stub detector: a 6 px box on every pixel of value 255, in the crop's own coordinates."""
    out = []
    for c in crops:
        ys, xs = np.nonzero(c[..., 0] == 255)
        out.append([(float(x), float(y), 6.0, 6.0, 0.9) for x, y in zip(xs, ys)])
    return out


@pytest.mark.parametrize("cfg", [PerceptionConfig(mode="sahi", window=768, overlap=0.2),
                                 PerceptionConfig(mode="sahi", window=768, overlap=0.0, fuse=False),
                                 PerceptionConfig(mode="sahi", window=512, overlap=0.2)])
def test_global_coordinates_are_window_origin_plus_local(cfg):
    # targets near the clamped last windows and on the far image edges: x_global = x0 + x_local
    img = np.zeros((1000, 1700, 3), np.uint8)
    truth = {(10, 10), (700, 500), (1650, 990), (1699, 0), (900, 640), (0, 999)}
    for x, y in truth:
        img[y, x] = 255
    got = {(round(d[0]), round(d[1])) for d in detect_image(img, _bright, cfg)}
    assert got == truth


def test_image_smaller_than_the_window_is_one_crop():
    img = np.zeros((200, 300, 3), np.uint8)
    img[150, 250] = 255
    cfg = PerceptionConfig(mode="sahi", window=768)
    assert slice_count(300, 200, cfg) == 1
    assert [(round(d[0]), round(d[1])) for d in detect_image(img, _bright, cfg)] == [(250, 150)]


def test_letterbox_matches_ultralytics_geometry_and_decode_inverts_it():
    from sat7.perception import decode_yolo, letterbox
    out, gain, pad = letterbox(np.zeros((768, 768, 3), np.uint8), 768)
    assert out.shape == (768, 768, 3) and gain == 1.0 and pad == (0, 0)        # native: identity
    out, gain, pad = letterbox(np.zeros((384, 768, 3), np.uint8), 768)
    assert out.shape == (768, 768, 3) and gain == 1.0 and pad == (0, 192)      # centre pad
    out, gain, pad = letterbox(np.zeros((300, 300, 3), np.uint8), 768)
    assert out.shape == (768, 768, 3) and gain == pytest.approx(2.56)         # scaleup, like predict
    head = np.array([[384.0], [384.0], [256.0], [128.0], [0.9]], np.float32)  # cx cy w h score
    (cx, cy, w, h, _), = decode_yolo(head, 0.05, 0.7, 300, 300, gain=gain, pad=pad)
    assert (cx, cy, w, h) == pytest.approx((150.0, 150.0, 100.0, 50.0), abs=1e-3)


def test_load_detector_rejects_missing_and_unknown_weights(tmp_path):
    from sat7.perception import load_detector
    with pytest.raises(FileNotFoundError):
        load_detector(tmp_path / "nope.onnx")
    bad = tmp_path / "w.h5"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        load_detector(bad)


def test_onnx_detector_maps_a_letterboxed_crop_back_to_the_tile():
    pytest.importorskip("onnxruntime")
    import cv2
    from pathlib import Path
    from sat7.b2_sahi_fusion import _iou
    from sat7.perception import load_detector
    repo = Path(__file__).resolve().parents[2]
    model = next((p for p in (repo / "code/runs/ships/weights/best.onnx",
                              repo / "demo/quickstart/model/best.onnx") if p.exists()), None)
    tile_p = repo / "demo/quickstart/tiles/01_synthetic_00113a75c.jpg"
    if model is None or not tile_p.exists():
        pytest.skip("ONNX weights or quickstart tiles not present")
    det = load_detector(model)
    tile = cv2.imread(str(tile_p))
    full = max(det([tile])[0], key=lambda d: d[4])               # the most confident ship
    x0 = int(min(max(full[0] - 192, 0), 768 - 384))
    y0 = int(min(max(full[1] - 192, 0), 768 - 384))
    crop = np.ascontiguousarray(tile[y0:y0 + 384, x0:x0 + 384])  # upscaled x2 by the letterbox
    lifted = [(x + x0, y + y0, w, h, c) for (x, y, w, h, c) in det([crop])[0]]
    assert max(_iou(full[:4], d[:4]) for d in lifted) >= 0.5
