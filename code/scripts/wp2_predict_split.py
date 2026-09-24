"""WP2 step 1: dump detector predictions for a split (normally val).

Calibration must be *fitted* on data the evaluation never sees. WP1 only dumped predictions
for the test split, so every calibration fitted on it would be marking its own homework.
The val split (5,320 leakage-free tiles, 8,173 ships) has never been used for anything --
this fills that gap now that the 3.12 environment works.

Matching rule is identical to WP1 (greedy, IoU >= 0.5, each prediction used once), so the
val dumps are directly comparable with results/wp1_predictions.csv.

    .venv312/Scripts/python.exe scripts/wp2_predict_split.py --split val

Outputs: results/wp2_<split>_predictions.csv, results/wp2_<split>_gt_matched.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def gt_boxes(label_path: Path, w: int = 768, h: int = 768) -> list[tuple[float, float, float, float]]:
    """YOLO label file -> [(cx, cy, w, h)] in pixels."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            out.append((float(p[1]) * w, float(p[2]) * h, float(p[3]) * w, float(p[4]) * h))
    return out


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix, iy = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.05, help="keep every box above this")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    dev = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.weights))
    img_dir, lbl_dir = args.data / "images" / args.split, args.data / "labels" / args.split
    files = sorted(img_dir.glob("*.jpg"))
    print(f"{len(files)} tiles in split '{args.split}', device {dev}")

    rows, matched = [], []
    t0 = time.perf_counter()
    for i in range(0, len(files), args.batch):
        batch = files[i:i + args.batch]
        results = model.predict([str(x) for x in batch], imgsz=args.imgsz, conf=args.conf,
                                device=dev, verbose=False)
        for f, res in zip(batch, results):
            preds = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                     for b, c in zip(res.boxes.xywh.cpu().numpy(), res.boxes.conf.cpu().numpy())]
            for cx, cy, w, h, conf in preds:
                rows.append({"image": f.name, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf})
            used = set()
            for g in gt_boxes(lbl_dir / (f.stem + ".txt")):
                best, best_i = 0.0, -1
                for k, p in enumerate(preds):
                    if k in used:
                        continue
                    v = iou(g, p[:4])
                    if v > best:
                        best, best_i = v, k
                if best >= 0.5:
                    used.add(best_i)
                matched.append({"image": f.name, "size_px": max(g[2], g[3]), "found": best >= 0.5,
                                "conf": preds[best_i][4] if best_i >= 0 and best >= 0.5 else 0.0})
        if i % (args.batch * 20) == 0:
            print(f"  {i + len(batch)}/{len(files)}", flush=True)

    dt = time.perf_counter() - t0
    pred, gtm = pd.DataFrame(rows), pd.DataFrame(matched)
    args.out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out / f"wp2_{args.split}_predictions.csv", index=False)
    gtm.to_csv(args.out / f"wp2_{args.split}_gt_matched.csv", index=False)
    print(f"\n{len(pred)} boxes, {len(gtm)} ground-truth ships, recall {gtm.found.mean():.3f}")
    print(f"{len(files)} tiles in {dt:.0f}s ({1000 * dt / max(1, len(files)):.0f} ms/tile)")
    print(f"Saved wp2_{args.split}_predictions.csv and wp2_{args.split}_gt_matched.csv in {args.out}")


if __name__ == "__main__":
    main()
