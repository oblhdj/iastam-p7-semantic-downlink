"""Run the onboard perception stage on ONE satellite image and show what it found.

    python scripts/detect_image.py --image path/to/scene.jpg
    python scripts/detect_image.py --image data/yolo_ships/images/test/00113a75c.jpg   # GT auto-found
    python scripts/detect_image.py --image big_scene.png --mode sahi --window 768 --overlap 0.2

The single-image path through the existing sat7 stages (nothing reimplemented):

  1. input       sat7.imagery.load_image -- validate container/decode/8-bit/size; print dimensions,
                 file bytes and raw (H x W x 3) bytes, the B0 definition
  2. model       sat7.perception.load_detector -- .onnx on CPU via onnxruntime (no torch) or .pt via
                 ultralytics; first found of runs/ships/weights/best.onnx,
                 ../demo/quickstart/model/best.onnx, runs/ships/weights/best.pt unless --model
  3. detection   sat7.perception.detect_image -- "whole" (one call; B1) or "sahi" (overlapping
                 windows, global coords x0 + x_local / y0 + y_local, NMS fusion; B2). "auto" picks
                 whole when the image fits one window, SAHI otherwise. Defaults are the repo's:
                 window 768 (detector-native, report 16), overlap 0.20, fusion NMS IoU 0.50
  4. evaluation  sat7.evaluation -- with ground truth (a YOLO label file: --labels, or found next
                 to the image in the images/ -> labels/ layout) precision / recall / F1 / AP50 /
                 AP50-95 at IoU 0.5; without it, counts and confidences only, never called accuracy
  5. output      <out>/<stem>_detections.png (original | detections side by side) + <stem>.json

Raw boxes are kept down to conf 0.05 (the wp1 dump threshold) so the AP curve is meaningful; the
operating cut --conf (0.25, onboard) decides what is drawn as a detection and scored as P/R/F1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from sat7.b2_sahi_fusion import plan_slices
from sat7.datasets import parse_yolo_lines
from sat7.evaluation import describe_detections, evaluate_detections
from sat7.imagery import load_image
from sat7.perception import (DEFAULT_RAW_CONF, PerceptionConfig, detect_image, load_detector,
                             slice_count)

WEIGHT_CANDIDATES = (ROOT / "runs" / "ships" / "weights" / "best.onnx",
                     ROOT.parent / "demo" / "quickstart" / "model" / "best.onnx",
                     ROOT / "runs" / "ships" / "weights" / "best.pt")


def find_weights() -> Path | None:
    return next((p for p in WEIGHT_CANDIDATES if p.exists()), None)


def find_labels(image: Path) -> Path | None:
    """The YOLO layout keeps <root>/labels/<split>/<stem>.txt beside <root>/images/<split>/."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:-1], image.stem + ".txt")
            return cand if cand.exists() else None
    return None


def xyxy(b):
    return (b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)


def render(img, info, dets, conf_thr, gts, windows, label):
    """Original | detections, side by side, with the dimensions and sizes in the header."""
    left, right = img.copy(), img.copy()
    for (x, y, w, h) in windows:                                   # SAHI windows (sahi mode)
        cv2.rectangle(right, (x + 1, y + 1), (x + w - 2, y + h - 2), (200, 200, 120), 1)
    for g in gts:                                                  # ground truth, blue
        a = [int(round(v)) for v in xyxy(g)]
        cv2.rectangle(right, (a[0], a[1]), (a[2], a[3]), (255, 140, 0), 2)
    for d in sorted(dets, key=lambda z: z[4]):
        a = [int(round(v)) for v in xyxy(d)]
        if d[4] >= conf_thr:
            cv2.rectangle(right, (a[0], a[1]), (a[2], a[3]), (60, 220, 60), 2)
            cv2.putText(right, f"{label} {d[4]:.2f}", (a[0], max(12, a[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(right, (a[0], a[1]), (a[2], a[3]), (150, 150, 150), 1)
    panel = np.hstack([left, np.full((img.shape[0], 8, 3), 255, np.uint8), right])
    scale = min(1.0, 2400 / panel.shape[1])
    if scale < 1.0:
        panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    head = np.full((64, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"original: {info.width} x {info.height} px | file {info.file_bytes:,} B | "
                      f"raw (HxWx3) {info.raw_bytes:,} B", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1, cv2.LINE_AA)
    n_kept = sum(d[4] >= conf_thr for d in dets)
    cv2.putText(head, f"right: {n_kept} detections >= {conf_thr} (green), below cut grey"
                      + (", ground truth blue" if gts else ""), (8, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, panel])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None, help="detector weights (.onnx or .pt)")
    ap.add_argument("--mode", choices=("auto", "whole", "sahi"), default="auto")
    ap.add_argument("--window", type=int, default=768, help="SAHI window px (repo default 768)")
    ap.add_argument("--overlap", type=float, default=0.20, help="SAHI overlap (repo default 0.20)")
    ap.add_argument("--nms-iou", type=float, default=0.50, help="fusion NMS IoU (repo default 0.50)")
    ap.add_argument("--conf", type=float, default=0.25, help="operating cut (onboard 0.25)")
    ap.add_argument("--labels", type=Path, default=None, help="YOLO label file; default: auto-find")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "demo" / "_live" / "detect_image",
                    help="output dir (default: gitignored demo/_live/detect_image)")
    args = ap.parse_args()

    # ---- 1. input
    try:
        img, info = load_image(args.image)
    except (ValueError, FileNotFoundError) as e:
        print(f"INPUT REJECTED: {e}")
        return 2
    print(f"[1] INPUT   {info.summary()}")
    for n in info.notes:
        print(f"            note: {n}")

    # ---- 2. model
    weights = args.model or find_weights()
    if weights is None:
        print("MODEL NOT FOUND: pass --model; weights are not in git (see demo/quickstart/model/)")
        return 2
    t0 = time.perf_counter()
    det = load_detector(weights)
    backend = (f"onnxruntime {det.ort_version}, CPU" if hasattr(det, "ort_version")
               else "ultralytics (torch)")
    names = getattr(det, "names", {0: "ship"})
    label = names.get(0, "ship")
    print(f"[2] MODEL   {weights} ({backend}); classes {names}; loaded in {time.perf_counter() - t0:.2f} s")

    # ---- 3. detection
    mode = args.mode
    if mode == "auto":
        mode = "whole" if info.width <= args.window and info.height <= args.window else "sahi"
    cfg = PerceptionConfig(mode=mode, window=args.window, overlap=args.overlap, nms_iou=args.nms_iou)
    windows = plan_slices(info.width, info.height, cfg.to_sahi()) if mode == "sahi" else []
    t0 = time.perf_counter()
    dets = detect_image(img, det, cfg)                       # global (cx, cy, w, h, conf)
    dt = time.perf_counter() - t0
    kept = sorted((d for d in dets if d[4] >= args.conf), key=lambda d: -d[4])
    print(f"[3] DETECT  mode {mode}: {slice_count(info.width, info.height, cfg)} detector call(s)"
          + (f" ({args.window} px windows, {args.overlap:.0%} overlap, fusion NMS IoU {args.nms_iou})"
             if mode == "sahi" else "") + f" in {dt:.2f} s")
    print(f"            {len(dets)} raw boxes >= {DEFAULT_RAW_CONF}; {len(kept)} at or above the "
          f"{args.conf} operating cut")
    if kept:
        print(f"            {'#':>3}  {'class':<6}{'conf':>6}  {'cx':>7}{'cy':>7}{'w':>6}{'h':>6}   "
              f"{'x0':>7}{'y0':>7}{'x1':>7}{'y1':>7}")
        for i, d in enumerate(kept, 1):
            a = xyxy(d)
            print(f"            {i:>3}  {label:<6}{d[4]:6.3f}  {d[0]:7.1f}{d[1]:7.1f}{d[2]:6.1f}{d[3]:6.1f}   "
                  f"{a[0]:7.1f}{a[1]:7.1f}{a[2]:7.1f}{a[3]:7.1f}")

    # ---- 4. evaluation
    lbl = args.labels or find_labels(args.image)
    gts = []
    if lbl is not None and lbl.exists():
        gts = parse_yolo_lines(lbl.read_text().splitlines(), info.width, info.height, classes={0})
        ev = evaluate_detections([(dets, gts)], conf_thr=args.conf)
        print(f"[4] EVAL    ground truth: {lbl} ({len(gts)} ships)")
        print(f"            @conf {args.conf}, IoU 0.5: TP {ev['TP']}  FP {ev['FP']}  FN {ev['FN']}  "
              f"precision {ev['precision']}  recall {ev['recall']}  F1 {ev['F1']}")
        print(f"            AP50 {ev['AP50']}  AP50-95 {ev['AP50-95']}  (one image: a spot check, "
              f"not a benchmark -- the measured figures are in results/)")
    else:
        ev = describe_detections(dets, args.conf)
        c = ev[f"confidence_at_or_above_{args.conf}"]
        print("[4] EVAL    no ground truth found -> counts and confidences only (NOT accuracy)")
        print(f"            {ev['n_at_or_above_thr']} detections >= {args.conf}"
              + (f"; confidence min {c['min']} / median {c['median']} / max {c['max']}" if c else ""))

    # ---- 5. output
    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / f"{args.image.stem}_detections.png"
    cv2.imwrite(str(png), render(img, info, dets, args.conf, gts, windows, label))
    rec = {"image": str(args.image), "input": info.__dict__, "model": str(weights), "backend": backend,
           "mode": mode, "window": args.window, "overlap": args.overlap, "nms_iou": args.nms_iou,
           "conf": args.conf, "detections": [{"class": label, "conf": round(d[4], 4),
                                              "cx": round(d[0], 2), "cy": round(d[1], 2),
                                              "w": round(d[2], 2), "h": round(d[3], 2)} for d in dets],
           "evaluation": ev}
    (args.out / f"{args.image.stem}.json").write_text(json.dumps(rec, indent=2))
    print(f"[5] OUTPUT  {png}  (+ {args.image.stem}.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
