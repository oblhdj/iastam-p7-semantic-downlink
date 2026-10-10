"""sat7.campaign: B0-B4 are switches on one chain, on one shared scene."""
import numpy as np
import pytest

from sat7.campaign import (DETECTIONS_ONLY, MODES, TILE, Scene, SceneTile, downlink, perceive, render,
                           sample_scenes)
from sat7.priority import P1, P3
from sat7.semantic import decode_downlink, raw_packet_bytes


class Stub:
    """A detect_fn that counts its crops and reports one fixed box per crop."""
    def __init__(self):
        self.calls = 0

    def __call__(self, crops):
        self.calls += len(crops)
        return [[(c.shape[1] / 2, c.shape[0] / 2, 40.0, 20.0, 0.5)] for c in crops]


def _scene():
    ctx = ["ships", "coast", "cloud", "ships"]
    tiles = [SceneTile(k // 2, k % 2, f"t{k}.jpg", ctx[k], ctx[k]) for k in range(4)]
    return Scene(0, 2, tiles)


def _img():
    return np.random.default_rng(0).integers(0, 255, (2 * TILE, 2 * TILE, 3), np.uint8)


def test_the_five_modes_differ_only_in_their_switches():
    sw = {n: (m.perception, m.payload, m.relay) for n, m in MODES.items()}
    assert sw == {"B0": ("none", "raw_image", False), "B1": ("whole", "detections", False),
                  "B2": ("sahi", "detections", False), "B3": ("sahi", "semantic", False),
                  "B4": ("sahi", "semantic", True)}


def test_perception_switch_b0_no_detector_b1_one_call_b2_sahi_windows():
    img, det = _img(), Stub()
    assert perceive(img, MODES["B0"], det) == [] and det.calls == 0
    perceive(img, MODES["B1"], det)
    assert det.calls == 1                                   # the whole original image, no slicing
    det.calls = 0
    perceive(img, MODES["B2"], det)
    assert det.calls == 9                                   # 768 px windows, 20% overlap on 1536 px


def test_b0_payload_is_the_original_image():
    img, s = _img(), _scene()
    dl = downlink(s, img, [], MODES["B0"])
    stream = b"".join(dl.products[0].packets)
    assert len(stream) == raw_packet_bytes(2 * TILE, 2 * TILE)
    assert np.array_equal(decode_downlink(stream)["raw_images"][0], img)


def test_detection_output_is_one_record_per_detection_above_the_cut():
    img, s = _img(), _scene()
    dets = [(100, 100, 40, 20, 0.9), (1000, 100, 40, 20, 0.3), (100, 1000, 40, 20, 0.1)]
    dl = downlink(s, img, dets, MODES["B2"])
    assert [p.kind for p in dl.products] == ["P1", "P1"]    # no ROI, no context, the 0.1 box dropped
    assert dl.levels == [P1, P1, 0] and DETECTIONS_ONLY.coast_to_p3 is False


def test_semantic_payload_uses_each_tiles_context():
    img, s = _img(), _scene()
    dets = [(100, 100, 40, 20, 0.9),        # tile 0, ships, confident  -> P1
            (1000, 100, 40, 20, 0.9),       # tile 1, coast             -> P3 + the tile as context
            (100, 1000, 40, 20, 0.9)]       # tile 2, cloud             -> P0, nothing sent
    dl = downlink(s, img, dets, MODES["B3"])
    assert dl.levels == [P1, P3, 0]
    ctx = [p for p in dl.products if p.kind == "P3"]
    assert len(ctx) == 1
    g = decode_downlink(b"".join(q for p in dl.products for q in p.packets))
    c = next(iter(g["contexts"].values()))
    assert (c["x0"], c["y0"], c["w"], c["h"]) == (TILE, 0, TILE, TILE)   # the coastal SOURCE tile
    assert len(g["records"]) == 2 and {round(r.cx) for r in g["records"]} == {100, 1000}


def test_b3_and_b4_share_perception_and_payload():
    assert (MODES["B3"].perception, MODES["B3"].payload) == (MODES["B4"].perception, MODES["B4"].payload)
    assert MODES["B4"].relay and not MODES["B3"].relay


def test_scenes_are_deterministic_and_carry_shifted_ground_truth():
    pools = {"ships": ["a.jpg"], "coast": ["b.jpg"], "cloud": ["c.jpg"], "empty": ["d.jpg"]}
    onboard = {"a.jpg": "ships", "b.jpg": "coast", "c.jpg": "cloud", "d.jpg": "ships"}
    gts = {"a.jpg": [(10.0, 20.0, 5.0, 5.0)]}
    s1 = sample_scenes(pools, onboard, lambda n: gts.get(n, []), 3, per_side=2, seed=4)
    s2 = sample_scenes(pools, onboard, lambda n: gts.get(n, []), 3, per_side=2, seed=4)
    assert [[t.image for t in s.tiles] for s in s1] == [[t.image for t in s.tiles] for s in s2]
    for s in s1:
        for g, k in zip(s.gts, s.gt_tile):
            t = s.tiles[k]
            assert t.image == "a.jpg" and g[:2] == (10.0 + t.col * TILE, 20.0 + t.row * TILE)
            assert s.tile_index(g[0], g[1]) == k
    img = render(s1[0], lambda n: np.full((TILE, TILE, 3), ord(n[0]), np.uint8))
    assert img.shape == (2 * TILE, 2 * TILE, 3) and img[0, 0, 0] == ord(s1[0].tiles[0].image[0])
