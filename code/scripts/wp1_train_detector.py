"""WP1 -- train the ship detector on the leakage-free dataset, then measure it properly.

    python scripts/wp1_train_detector.py --epochs 12 --imgsz 768 --batch 8

Steps
  1. train YOLOv8n on data/yolo_ships (built by WP0, no duplicate leaks),
  2. evaluate on the TEST split: mAP50, precision, recall,
  3. break the recall down BY SHIP SIZE (small ships are what suffers),
  4. export ONNX (FP32) and INT8, and time both on the CPU,
  5. write predictions.csv (every box with score >= 0.05) -- needed for calibration (WP2)
     and to feed real detections into the scheduler (WP5).

Outputs: results/wp1_*.csv|png|json, runs/detect/... (weights)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZE_BINS = [(0, 32, "small (<32 px)"), (32, 96, "medium (32-96 px)"), (96, 10_000, "large (>96 px)")]


def gt_boxes(label_file: Path, w: int = 768, h: int = 768) -> list[tuple[float, float, float, float]]:
    out = []
    if label_file.exists():
        for line in label_file.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                cx, cy, bw, bh = (float(v) for v in p[1:])
                out.append((cx * w, cy * h, bw * w, bh * h))
    return out


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix, iy = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships" / "data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.05, help="keep every box above this for calibration")
    ap.add_argument("--skip-train", action="store_true", help="reuse the last weights")
    ap.add_argument("--resume", action="store_true", help="continue from runs/ships/weights/last.pt")
    ap.add_argument("--weights", type=Path, help="explicit weights for evaluation")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from ultralytics import YOLO
    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ---------------------------------------------------------------- 1. train
    weights = args.weights
    if args.resume:
        last = ROOT / "runs" / "ships" / "weights" / "last.pt"
        print(f"resuming from {last}")
        r = YOLO(str(last)).train(resume=True)          # epochs/imgsz/batch come from the checkpoint
        weights = Path(r.save_dir) / "weights" / "best.pt"
    elif not args.skip_train:
        model = YOLO(args.model)
        r = model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                        workers=args.workers, device=dev, project=str(ROOT / "runs"), name="ships",
                        exist_ok=True, patience=5, seed=0, val=True, plots=True, cache=False)
        weights = Path(r.save_dir) / "weights" / "best.pt"
    weights = weights or (ROOT / "runs" / "ships" / "weights" / "best.pt")
    print(f"weights: {weights}")
    model = YOLO(str(weights))

    # ---------------------------------------------------------------- 2. test metrics
    m = model.val(data=str(args.data), split="test", imgsz=args.imgsz, device=dev, verbose=False)
    overall = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "precision": float(m.box.mp),
               "recall": float(m.box.mr)}
    print("\nTEST:", {k: round(v, 4) for k, v in overall.items()})

    # ---------------------------------------------------------------- 3. predictions + size breakdown
    img_dir = args.data.parent / "images" / "test"
    lbl_dir = args.data.parent / "labels" / "test"
    files = sorted(img_dir.glob("*.jpg"))
    rows, matched_rows = [], []
    t0 = time.perf_counter()
    for i in range(0, len(files), 32):
        batch = files[i:i + 32]
        for f, res in zip(batch, model.predict([str(x) for x in batch], imgsz=args.imgsz, conf=args.conf,
                                               device=dev, verbose=False)):
            preds = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                     for b, c in zip(res.boxes.xywh.cpu().numpy(), res.boxes.conf.cpu().numpy())]
            for cx, cy, w, h, conf in preds:
                rows.append({"image": f.name, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf})
            gts = gt_boxes(lbl_dir / (f.stem + ".txt"))
            used = set()
            for g in gts:
                best, best_i = 0.0, -1
                for k, p in enumerate(preds):
                    if k in used:
                        continue
                    v = iou(g, p[:4])
                    if v > best:
                        best, best_i = v, k
                if best >= 0.5:
                    used.add(best_i)
                matched_rows.append({"image": f.name, "size_px": max(g[2], g[3]), "found": best >= 0.5,
                                     "conf": preds[best_i][4] if best_i >= 0 and best >= 0.5 else 0.0})
        if i % 640 == 0:
            print(f"  predicted {i + len(batch)}/{len(files)} tiles", flush=True)
    dt = time.perf_counter() - t0
    pred = pd.DataFrame(rows)
    gtm = pd.DataFrame(matched_rows)
    pred.to_csv(args.out / "wp1_predictions.csv", index=False)
    gtm.to_csv(args.out / "wp1_gt_matched.csv", index=False)

    by_size = []
    for lo, hi, name in SIZE_BINS:
        sub = gtm[(gtm.size_px >= lo) & (gtm.size_px < hi)]
        by_size.append({"bucket": name, "ships": len(sub), "recall": float(sub.found.mean()) if len(sub) else np.nan})
    bs = pd.DataFrame(by_size)
    print("\nrecall by ship size (IoU 0.5, conf >= %.2f):" % args.conf)
    print(bs.round(3).to_string(index=False))
    print(f"\ninference: {len(files)} tiles in {dt:.0f}s on {'GPU' if dev == 0 else 'CPU'} "
          f"({1000 * dt / max(1, len(files)):.0f} ms/tile)")

    # ---------------------------------------------------------------- 4. export + CPU timing
    exports = {}
    for fmt, kwargs in (("onnx", {}), ("onnx-int8", {"int8": True, "dynamic": False})):
        try:
            t = time.perf_counter()
            p = model.export(format="onnx", imgsz=args.imgsz, opset=13, **({} if fmt == "onnx" else kwargs))
            exports[fmt] = {"path": str(p), "MB": round(Path(p).stat().st_size / 1e6, 1),
                            "export_s": round(time.perf_counter() - t, 1)}
        except Exception as e:                      # keep going: the report can live without it
            exports[fmt] = {"error": str(e)[:200]}
    print("\nexports:", json.dumps(exports, indent=2))

    (args.out / "wp1_metrics.json").write_text(json.dumps(
        {"weights": str(weights), "test": overall, "by_size": by_size,
         "inference_ms_per_tile_gpu": 1000 * dt / max(1, len(files)), "exports": exports,
         "epochs": args.epochs, "imgsz": args.imgsz}, indent=2))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(bs.bucket, bs.recall, color=["#C0392B", "#E07A1F", "#2E8B57"])
    for i, (r_, n) in enumerate(zip(bs.recall, bs.ships)):
        ax.text(i, r_ + 0.01, f"{r_:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("recall @ IoU 0.5")
    ax.set_title("Detector recall by ship size (test split, no leakage)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "wp1_recall_by_size.png", dpi=160)
    print(f"\nSaved wp1_metrics.json, wp1_predictions.csv, wp1_recall_by_size.png in {args.out}")


if __name__ == "__main__":
    main()
