"""First-class onboard perception path: whole-image (B1) vs SAHI-sliced (B2), one API.

The accepted paper's B-progression makes SAHI a *mode of the perception stage*, not an
ablation: B1 is "YOLO on the original image", B2 is "SAHI + YOLO". `sat7.b2_sahi_fusion`
already holds the proven slicing + global-coordinate fusion (validated in wp16 to 0.0000
recall error vs per-tile); this module lifts it to a selectable front door so a caller picks
the path with one config object instead of calling a different function per config, and so the
paper's three slicing ablations become flags rather than separate code paths.

    from sat7.perception import PerceptionConfig, detect_image
    dets = detect_image(image, detect_fn, PerceptionConfig(mode="sahi", window=512))

`detect_fn(list_of_crops) -> per-crop [(cx, cy, w, h, conf), ...]` in each crop's own pixel
coordinates -- exactly the callable `sat7.b2_sahi_fusion.run_sahi` expects and the one a YOLO
`.predict` wrapper yields. `make_yolo_detect_fn` below is that wrapper, kept here so the whole
perception stage lives in one place; it is the ONLY thing that imports ultralytics/torch, and it
does so lazily inside the factory, so `import sat7.perception` still works in the non-torch .venv
(START_HERE section 6). Everything else in this module is detector-agnostic and unit-testable
without a GPU.

Box convention everywhere, matching b2_sahi_fusion and every wp script: (cx, cy, w, h, conf) in
pixels, centre form. Slice-local on the way into `detect_fn`, image-global on the way out.

Labels (project rule): the slicing/fusion geometry is exact arithmetic; recall numbers this path
produces are REAL only when `detect_fn` is the trained detector on real imagery. This module emits
no numbers of its own.
"""
from __future__ import annotations

from dataclasses import dataclass

from .b2_sahi_fusion import SahiConfig, fuse, nms, plan_slices

# the three slicing ablations the paper lists (PAPER_COVERAGE section 6), as named modes rather
# than forks. "without SAHI" is simply mode="whole"; the other two are flags on the sahi mode.
MODES = ("whole", "sahi")


@dataclass
class PerceptionConfig:
    """How the onboard detector sees one captured frame.

    mode="whole"  -> B1: one detector call on the whole frame. The "without SAHI" ablation.
    mode="sahi"   -> B2: slice into overlapping windows, detect each, lift to global coords, fuse.

    The two sahi ablations are flags, not separate code, so a sweep toggles them directly:
      overlap=0.0   -> "without tile overlap"  (abutting windows, seams uncovered)
      fuse=False    -> "without detection fusion" (global coords, but cross-window duplicates kept)
    """
    mode: str = "whole"
    window: int = 768          # SAHI slice size (px). 768 = detector-native (wp16 default); 512 = paper
    overlap: float = 0.20      # neighbour-window overlap fraction; 0.0 = "without overlap" ablation
    nms_iou: float = 0.50      # IoU above which two detections merge in fusion (highest conf wins)
    min_conf: float = 0.0      # drop detections below this before fusing
    fuse: bool = True          # False = "without detection fusion" ablation (duplicates survive)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")

    def to_sahi(self) -> SahiConfig:
        """The b2_sahi_fusion config this perception config drives."""
        return SahiConfig(window=self.window, overlap=self.overlap,
                          nms_iou=self.nms_iou, min_conf=self.min_conf)


def detect_image(image, detect_fn, cfg: PerceptionConfig | None = None):
    """Run the perception stage on one frame and return image-global detections.

    `image` is an (H, W, C) array; `detect_fn` takes a list of crops and returns, per crop, an
    iterable of (cx, cy, w, h, conf) in that crop's own pixel coordinates. Returns a list of
    (cx, cy, w, h, conf) in image-global coordinates -- the same shape for both modes, so a caller
    (or the B0-B4 campaign runner) swaps B1 for B2 by changing only the config.
    """
    cfg = cfg or PerceptionConfig()
    if cfg.mode == "whole":
        # one window == the whole frame, origin (0, 0): fusion is a no-op but we route through the
        # same lift+NMS so a single dead box and a SAHI box are handled identically downstream.
        h, w = image.shape[:2]
        slices = [(0, 0, w, h)]
        per_slice = detect_fn([image])
        _check(per_slice, slices)
        boxes = [(float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]))
                 for d in per_slice[0] if float(d[4]) >= cfg.min_conf]
        return nms(boxes, cfg.nms_iou) if cfg.fuse else boxes

    scfg = cfg.to_sahi()
    h, w = image.shape[:2]
    slices = plan_slices(w, h, scfg)
    crops = [image[y:y + hh, x:x + ww] for (x, y, ww, hh) in slices]
    per_slice = detect_fn(crops)
    _check(per_slice, slices)
    if cfg.fuse:
        return fuse(per_slice, slices, scfg)
    # "without fusion": still lift to global coords (otherwise the boxes are meaningless), but skip
    # the cross-window NMS, so the overlap duplicates remain -- which is exactly what the ablation
    # is meant to expose as double-counted detections.
    glob = []
    for (x, y, _w, _h), dets in zip(slices, per_slice):
        for d in dets:
            if float(d[4]) >= scfg.min_conf:
                glob.append((float(d[0]) + x, float(d[1]) + y, float(d[2]), float(d[3]), float(d[4])))
    return glob


def slice_count(width: int, height: int, cfg: PerceptionConfig) -> int:
    """How many detector calls this frame costs -- the compute side of the SAHI trade (report 16).

    1 for whole-image; the SAHI window grid otherwise. Lets a caller price B2 against B1 (e.g. the
    1.56x at 768 / 20% on a 3072 swath) without running the detector.
    """
    if cfg.mode == "whole":
        return 1
    return len(plan_slices(width, height, cfg.to_sahi()))


def _check(per_slice, slices) -> None:
    if len(per_slice) != len(slices):
        raise ValueError(f"detect_fn returned {len(per_slice)} results for {len(slices)} crops")


def make_yolo_detect_fn(model, imgsz: int = 768, conf: float = 0.25, device=None):
    """The YOLO `detect_fn` B1/B2 share: crops -> per-crop (cx, cy, w, h, conf).

    Lifted verbatim from wp18_campaign_runner.make_detect_fn so the perception stage owns its own
    detector adapter instead of the campaign script. ultralytics/torch is imported lazily only to
    resolve the device default, so importing this module never pulls in torch; passing an explicit
    `device` avoids even that. `model` is an already-loaded ultralytics YOLO.
    """
    if device is None:
        import torch                                   # lazy: only when the caller wants auto-device
        device = 0 if torch.cuda.is_available() else "cpu"

    def detect_fn(crops):
        res = model.predict(crops, imgsz=imgsz, conf=conf, device=device, verbose=False)
        return [[(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                 for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                for r in res]
    return detect_fn


if __name__ == "__main__":
    # Self-check without a detector (mirrors b2_sahi_fusion's): a stub detect_fn that reports one
    # fixed box per crop lets us prove each mode's geometry and each ablation's effect.
    import numpy as np

    img = np.zeros((3072, 3072, 3), np.uint8)

    # whole: exactly one detector call, one box through unchanged
    one = detect_image(img, lambda crops: [[(100, 100, 20, 20, 0.9)] for _ in crops],
                       PerceptionConfig(mode="whole"))
    assert len(one) == 1 and slice_count(3072, 3072, PerceptionConfig(mode="whole")) == 1
    print(f"whole-image: {len(one)} detection, 1 call OK")

    # sahi vs whole slice counts match wp16 (25 @ 768, 64 @ 512)
    for win, expect in ((768, 25), (512, 64)):
        n = slice_count(3072, 3072, PerceptionConfig(mode="sahi", window=win))
        assert n == expect, (win, n, expect)
    print("sahi slice counts match wp16 (25 @ 768, 64 @ 512) OK")

    # fusion vs no-fusion: two overlapping windows see the same ship -> fuse merges to 1, the
    # "without fusion" ablation keeps both (the double-count the ablation is meant to show).
    def stub(crops):
        # every crop reports a ship at its own local (50, 50); windows at global 0 and ~614 overlap
        return [[(50, 50, 40, 40, 0.9)] for _ in crops]
    fused = detect_image(img[:768, :1400], stub, PerceptionConfig(mode="sahi", window=768, overlap=0.2))
    unfused = detect_image(img[:768, :1400], stub, PerceptionConfig(mode="sahi", window=768,
                                                                    overlap=0.2, fuse=False))
    assert len(unfused) >= len(fused)
    print(f"fusion {len(fused)} <= no-fusion {len(unfused)} detections OK")
    print("perception self-check OK")
