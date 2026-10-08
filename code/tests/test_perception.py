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
