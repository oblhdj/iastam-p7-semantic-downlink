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

import ast
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .b2_sahi_fusion import SahiConfig, fuse, plan_slices

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
    `fuse`/`nms_iou` act on the sahi mode only: a whole frame is one window, so there is nothing
    to fuse and the detector's own output is returned as-is.
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
        # one window == the whole frame, origin (0, 0): the lift is the identity and there is no
        # second window to duplicate a box, so the detector's own output IS the B1 answer. No
        # cross-window NMS here (audit B2): a second NMS at nms_iou=0.5 on top of the detector's
        # own NMS (0.7) would merge distinct, overlapping ships the detector deliberately kept,
        # making "whole" differ from the raw detector output that B1 / wp1_predictions report.
        # The same reason wp16 does not fuse its non-overlapping tilings.
        h, w = image.shape[:2]
        slices = [(0, 0, w, h)]
        per_slice = detect_fn([image])
        _check(per_slice, slices)
        return [(float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]))
                for d in per_slice[0] if float(d[4]) >= cfg.min_conf]

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


def make_yolo_detect_fn(model, imgsz: int = 768, conf: float = 0.25, device=None,
                        iou: float = 0.70):
    """The YOLO `detect_fn` B1/B2 share: crops -> per-crop (cx, cy, w, h, conf).

    Lifted verbatim from wp18_campaign_runner.make_detect_fn so the perception stage owns its own
    detector adapter instead of the campaign script. ultralytics/torch is imported lazily only to
    resolve the device default, so importing this module never pulls in torch; passing an explicit
    `device` avoids even that. `model` is an already-loaded ultralytics YOLO. `iou` is the detector's
    own NMS threshold (0.70 = ultralytics' predict default, what every wp script ran with).
    """
    if device is None:
        import torch                                   # lazy: only when the caller wants auto-device
        device = 0 if torch.cuda.is_available() else "cpu"

    def detect_fn(crops):
        res = model.predict(crops, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        return [[(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                 for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                for r in res]
    return detect_fn


# ------------------------------------------------------------------ torch-free detector (ONNX)
# The trained detector, exported by wp1 as ONNX FP32 (measured lossless vs PyTorch in every size
# bucket, results/wp1_export.json), run through onnxruntime on CPU: no torch, no CUDA. Its
# post-processing mirrors ultralytics' predict so boxes are comparable with wp1_predictions.csv
# (verified: 40/40 boxes paired on 9 real test tiles, max |dconf| 3.4e-4, 0 decision flips --
# demo/quickstart/README.md): ultralytics LetterBox (centre pad, grey 114) -> RGB/255 -> score >
# conf -> greedy NMS (suppress IoU > iou, torchvision rule) -> at most 300 boxes -> undo the
# letterbox -> clip to the crop. A 768x768 crop letterboxes to itself, so tiles and SAHI windows
# take the exact path that was verified.

DEFAULT_RAW_CONF = 0.05   # the raw dump threshold wp1_predictions.csv was written at; the onboard
                          # 0.25 cut is applied downstream, so P0 / sub-threshold boxes stay visible
DEFAULT_NMS_IOU = 0.70    # ultralytics predict default


def letterbox(img: np.ndarray, size: int, color: int = 114):
    """Ultralytics LetterBox(auto=False, center=True, scaleup=True): resize keeping aspect, pad to
    size x size. Returns (padded, gain, (pad_left, pad_top))."""
    import cv2
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(color, color, color))
    return img, r, (left, top)


def nms_xyxy(xyxy: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """Greedy NMS, highest score first; a box is suppressed when IoU > thr (torchvision rule)."""
    x0, y0, x1, y1 = xyxy.T
    areas = (x1 - x0) * (y1 - y0)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        iw = np.clip(np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]), 0, None)
        ih = np.clip(np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]), 0, None)
        inter = iw * ih
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= thr]
    return np.asarray(keep, dtype=int)


def decode_yolo(out: np.ndarray, conf: float, iou: float, w: int, h: int, max_det: int = 300,
                max_nms: int = 30000, gain: float = 1.0, pad: tuple[float, float] = (0.0, 0.0)):
    """YOLOv8 head output (4 + n_classes, anchors) -> [(cx, cy, w, h, conf)] in crop pixels.

    `gain`/`pad` undo the letterbox (ultralytics scale_boxes); the boxes are then clipped to the
    w x h crop. Class-agnostic: the repo's detector has one class (ship) and the box convention
    carries no class id.
    """
    scores = out[4:].max(0)
    keep = scores > conf
    if not keep.any():
        return []
    cx, cy, bw, bh = (out[k][keep] for k in range(4))
    s = scores[keep]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    if len(s) > max_nms:
        top = np.argsort(-s)[:max_nms]
        xyxy, s = xyxy[top], s[top]
    idx = nms_xyxy(xyxy, s, iou)[:max_det]
    xyxy, s = xyxy[idx].copy(), s[idx]
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy /= gain
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    return [(float((a + c) / 2), float((b + d) / 2), float(c - a), float(d - b), float(sc))
            for (a, b, c, d), sc in zip(xyxy, s)]


class OnnxDetector:
    """The trained detector as an ONNX FP32 file on CPU; a `detect_fn` (list of BGR crops in,
    per-crop (cx, cy, w, h, conf) out), so it drops into detect_image / run_sahi unchanged."""

    def __init__(self, path, conf: float = DEFAULT_RAW_CONF, iou: float = DEFAULT_NMS_IOU):
        import onnxruntime as ort                       # lazy: importing sat7 never needs it
        self.path = Path(path)
        self.ort_version = ort.__version__
        self.sess = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 768
        n_out = self.sess.get_outputs()[0].shape[1]
        if isinstance(n_out, int) and n_out != 5:
            raise ValueError(f"{self.path.name}: {n_out - 4} classes; this adapter expects the repo's "
                             f"single-class (ship) detector -- the box convention carries no class id")
        meta = self.sess.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(meta.get("names", "{}"))
            self.names = {int(k): str(v) for k, v in names.items()} or {0: "ship"}
        except (ValueError, SyntaxError, AttributeError):
            self.names = {0: "ship"}
        self.conf, self.iou = conf, iou
        self.calls, self.ms = 0, 0.0

    def __call__(self, crops):
        return [self.detect(c) for c in crops]

    def detect(self, bgr: np.ndarray) -> list[tuple]:
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.shape[0] == 0 or bgr.shape[1] == 0:
            raise ValueError(f"expected a non-empty H x W x 3 BGR crop, got shape {bgr.shape}")
        h, w = bgr.shape[:2]
        img, gain, pad = letterbox(bgr, self.size)
        x = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        t = time.perf_counter()
        out = self.sess.run(None, {self.input: x})[0][0]
        self.ms += 1e3 * (time.perf_counter() - t)
        self.calls += 1
        return decode_yolo(out, self.conf, self.iou, w, h, gain=gain, pad=pad)


def load_detector(weights, conf: float = DEFAULT_RAW_CONF, iou: float = DEFAULT_NMS_IOU,
                  imgsz: int = 768, device=None):
    """The configured detector as a `detect_fn`, from its file: `.onnx` -> onnxruntime on CPU (no
    torch); `.pt` -> ultralytics (torch, GPU if available). Both expose `.names` ({class id: name}).

    The weights are not in git (code/runs/ is ignored); a CPU copy of the ONNX export ships with
    the quickstart demo at demo/quickstart/model/best.onnx.
    """
    p = Path(weights)
    if not p.exists():
        raise FileNotFoundError(f"detector weights not found: {p} (code/runs/ is not in git; the "
                                f"ONNX copy is demo/quickstart/model/best.onnx)")
    if p.suffix.lower() == ".onnx":
        return OnnxDetector(p, conf=conf, iou=iou)
    if p.suffix.lower() == ".pt":
        from ultralytics import YOLO                     # lazy: torch only on this branch
        model = YOLO(str(p))
        fn = make_yolo_detect_fn(model, imgsz=imgsz, conf=conf, device=device, iou=iou)
        fn.names = dict(model.names)
        return fn
    raise ValueError(f"unsupported weights format {p.suffix!r}: use .onnx (CPU) or .pt (ultralytics)")


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
