"""B2 -- SAHI sliced inference with global-coordinate fusion (onboard perception stage).

The accepted paper's B2 config is "SAHI + YOLO": slice a wide swath into overlapping windows,
run the detector on each, then lift every detection back into swath-global coordinates and merge
the duplicates that the overlap produces. `../docs/START_HERE.md` section 3 listed the
"global-coordinate transform + detection fusion" as NOT implemented; this module is it,
production form, lifted from the version proven correct in `scripts/wp16_swath_policies.py`
(whose sanity gate confirmed the fused whole-swath recall equals per-tile recall to 0.0000).

Design:
  * Detector-agnostic. This module does no inference and imports no torch; the caller passes a
    `detect_fn(crops) -> per-crop detections` callable. So sat7 stays importable in the non-torch
    .venv, and the fusion is unit-testable without a GPU.
  * Parameterised by window size. `window=768` is the detector's native input (no upscaling) and
    the recommended setting for this detector; `window=512` is the paper's literal spec and is
    reproducible by passing it. Report 16 measured that at 512 the window cannot hold a large
    straddling ship intact and the slice count triples (64 vs 25 on a 3072 swath), so 768 is the
    default here -- but nothing is hardcoded to it.

Box convention everywhere: (cx, cy, w, h, conf) in pixels, centre form -- the same the detector
and the WP scripts use. Slice-local on the way in, swath-global on the way out.

    from sat7.b2_sahi_fusion import SahiConfig, run_sahi
    dets = run_sahi(swath_bgr, detect_fn, SahiConfig(window=768))

where detect_fn(list_of_crops) returns, per crop, a list/array of (cx, cy, w, h, conf) in that
crop's own pixel coordinates (exactly what a YOLO predict wrapper yields).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SahiConfig:
    window: int = 768          # slice size in px; 768 = detector-native (recommended), 512 = paper
    overlap: float = 0.20      # fraction of overlap between neighbouring windows
    nms_iou: float = 0.50      # IoU above which two fused detections are merged (highest conf wins)
    min_conf: float = 0.0      # drop detections below this before fusing (detector usually pre-cuts)


def plan_slices(width: int, height: int, cfg: SahiConfig) -> list[tuple[int, int, int, int]]:
    """SAHI window grid covering the whole image: list of (x, y, w, h) in image pixels.

    Windows step by stride = window*(1-overlap); the last window on each axis is clamped to the
    edge so it stays `window`-sized (standard SAHI, and the convention wp16 validated). On a
    3072 px swath this yields 25 windows at 768/20% and 64 at 512/20%.
    """
    win = cfg.window
    stride = max(1, int(round(win * (1 - cfg.overlap))))

    def origins(extent: int) -> list[int]:
        if extent <= win:
            return [0]
        xs = sorted({min(x, extent - win) for x in range(0, extent, stride)})
        return xs

    xs, ys = origins(width), origins(height)
    return [(x, y, min(win, width), min(win, height)) for y in ys for x in xs]


def _iou(a, b) -> float:
    """IoU of two centre-form boxes (cx, cy, w, h). Identical rule to wp12_seams / wp16."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, iou_thr: float = 0.5):
    """Greedy NMS, highest confidence first -- the cross-slice duplicate merge proven in wp16."""
    keep = []
    for b in sorted(boxes, key=lambda z: -z[4]):
        if all(_iou(b[:4], k[:4]) < iou_thr for k in keep):
            keep.append(b)
    return keep


def fuse(per_slice, slices, cfg: SahiConfig):
    """Lift slice-local detections to swath-global coords and merge overlap duplicates.

    This is the exact step wp16's sanity gate validated: adding each slice's (x, y) origin to a
    detection's (cx, cy) places it in swath-global space; greedy NMS then removes the duplicates
    that the window overlap creates. `per_slice[i]` are the detections for `slices[i]`.
    """
    glob = []
    for (x, y, _w, _h), dets in zip(slices, per_slice):
        for d in dets:
            cx, cy, w, h, conf = float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])
            if conf >= cfg.min_conf:
                glob.append((cx + x, cy + y, w, h, conf))
    return nms(glob, cfg.nms_iou)


def run_sahi(image, detect_fn, cfg: SahiConfig | None = None):
    """Full B2: plan windows, crop, detect, fuse to swath-global detections.

    `image` is an (H, W, C) array; `detect_fn(crops)` takes a list of window crops and returns,
    per crop, an iterable of (cx, cy, w, h, conf) in that crop's own pixel coordinates.
    Returns the fused global detections as a list of (cx, cy, w, h, conf).
    """
    cfg = cfg or SahiConfig()
    h, w = image.shape[:2]
    slices = plan_slices(w, h, cfg)
    crops = [image[y:y + hh, x:x + ww] for (x, y, ww, hh) in slices]
    per_slice = detect_fn(crops)
    if len(per_slice) != len(slices):
        raise ValueError(f"detect_fn returned {len(per_slice)} results for {len(slices)} crops")
    return fuse(per_slice, slices, cfg)


if __name__ == "__main__":
    # Lightweight self-check (no detector): slice counts match wp16, and fusion merges a duplicate
    # that two overlapping windows produce for one ship while keeping two genuinely distinct ships.
    for win, expect in ((768, 25), (512, 64)):
        n = len(plan_slices(3072, 3072, SahiConfig(window=win)))
        print(f"plan_slices 3072x3072 @ {win}px/20% -> {n} windows (wp16 expected {expect})")
        assert n == expect, f"slice count {n} != {expect}"
    # two windows each see the same ship at global ~ (400,400); a third sees a distinct ship far away
    slices = [(0, 0, 768, 768), (300, 0, 768, 768)]
    per_slice = [[(400, 400, 40, 40, 0.9)], [(100, 400, 40, 40, 0.8), (700, 700, 30, 30, 0.7)]]
    fused = fuse(per_slice, slices, SahiConfig())
    # slice0 ship at global (400,400); slice1 first ship at (400,400) duplicate -> merged;
    # slice1 second ship at global (1000,700) distinct -> kept. Expect 2 detections.
    print(f"fuse merged overlap duplicate -> {len(fused)} detections (expected 2)")
    assert len(fused) == 2, fused
    print("b2_sahi_fusion self-check OK")
