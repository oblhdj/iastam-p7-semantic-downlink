"""WP6 step 1: turn the real test tiles + real detections into a tile catalogue.

The scheduler experiment (WP5) used a synthetic workload: contexts drawn from
assumed probabilities, detector confidence drawn from a Beta(5,2). This script
replaces every one of those draws with a measurement on the 5,320 held-out test
tiles:

  * context        classic pre-filter (sat7.prefilter) -> cloud / land_coast / sea
  * ships per tile ground truth (data/yolo_ships/labels/test)
  * ship size      ground-truth box, px
  * detected?      real detector output (results/wp1_gt_matched.csv)
  * confidence     real detector confidence of the matching box
  * false alarms   predictions not matched to any ground-truth ship, per tile

Outputs (results/):
  wp6_tiles.csv       one row per test tile  (context, cloud_frac, n_ships, n_fp_<thr>)
  wp6_ships.csv       one row per GT ship    (tile, size_px, found, conf, context)
  wp6_catalogue.json  summary + the false-alarm rate as a function of threshold

    python scripts/wp6_build_catalogue.py            # all 5,320 test tiles
    python scripts/wp6_build_catalogue.py --n 200    # quick check
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from sat7.prefilter import CLOUD, LAND, PrefilterConfig, run_prefilter

FP_THRESHOLDS = (0.05, 0.25, 0.40)     # report the false-alarm rate at several cut-offs


def _classify(path_str: str):
    """Run the classic pre-filter on one tile (worker process)."""
    path = Path(path_str)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return path.name, "unreadable", 0.0, 0.0, 0.0, 0.0
    r = run_prefilter(img, PrefilterConfig())
    # n_candidates: how many bright objects the cheap classic stage saw. Compared against
    # what the network confirmed, it is an onboard uncertainty signal (WP7 adaptive coast).
    return (path.name, r.context, r.cloud_frac, r.edge_density, r.big_blob_frac,
            len(r.boxes), r.runtime_ms)


def _gt_boxes(label_path: Path, w: int = 768, h: int = 768) -> np.ndarray:
    """YOLO label file -> (n, 4) array of xyxy boxes in pixels."""
    if not label_path.exists():
        return np.zeros((0, 4))
    rows = [l.split() for l in label_path.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4))
    a = np.array([[float(v) for v in r[1:5]] for r in rows])
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def flag_predictions(preds: pd.DataFrame, labels_dir: Path, tiles: list[str],
                     iou_thr: float = 0.3) -> pd.DataFrame:
    """Tag every prediction as a true positive or a false alarm.

    A prediction is a true positive if it overlaps a ground-truth ship by IoU >= 0.3
    (the rule used in the detector evaluation); everything else is a false alarm the
    downlink would have to carry. Flagging each box, rather than counting at a few
    fixed cut-offs, lets the workload count false alarms at *any* threshold later.
    """
    want = set(tiles)
    out = []
    for name, p in preds.groupby("image"):
        if name not in want:
            continue
        gt = _gt_boxes(labels_dir / str(name).replace(".jpg", ".txt"))
        pb = np.stack([p.cx - p.w / 2, p.cy - p.h / 2, p.cx + p.w / 2, p.cy + p.h / 2], axis=1)
        hit = (_iou_matrix(pb, gt) >= iou_thr).any(axis=1) if gt.size else np.zeros(len(pb), bool)
        out.append(pd.DataFrame({"image": name, "conf": p.conf.to_numpy(), "false_alarm": ~hit}))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["image", "conf", "false_alarm"]))


def count_false_alarms(flags: pd.DataFrame, tiles: list[str]) -> pd.DataFrame:
    """Per-tile false-alarm / true-positive counts at the reporting thresholds."""
    base = pd.DataFrame({"image": tiles})
    for t in FP_THRESHOLDS:
        k = flags[flags.conf >= t]
        fp = k[k.false_alarm].groupby("image").size().rename(f"n_fp_{t:g}")
        tp = k[~k.false_alarm].groupby("image").size().rename(f"n_tp_{t:g}")
        base = base.join(fp, on="image").join(tp, on="image")
    return base.fillna(0).astype({c: int for c in base.columns if c != "image"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--preds", type=Path, default=ROOT / "results" / "wp1_predictions.csv")
    ap.add_argument("--matched", type=Path, default=ROOT / "results" / "wp1_gt_matched.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0, help="only the first N tiles (debug)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reuse-context", action="store_true",
                    help="keep the pre-filter contexts already in wp6_tiles.csv (skips the slow part)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    img_dir = args.data / "images" / args.split
    lab_dir = args.data / "labels" / args.split
    tiles = sorted(p.name for p in img_dir.glob("*.jpg"))
    if args.n:
        tiles = tiles[:args.n]
    print(f"{len(tiles)} {args.split} tiles")

    # ---- 1. context of every tile: classic pre-filter, the same code that runs onboard
    keep = ["image", "context", "cloud_frac", "edge_density", "big_blob_frac", "n_candidates",
            "prefilter_ms"]
    cache = args.out / "wp6_tiles.csv"
    if args.reuse_context and cache.exists():
        ctx = pd.read_csv(cache)[keep]
        ctx = ctx[ctx.image.isin(set(tiles))]
        print(f"reusing pre-filter contexts for {len(ctx)} tiles from {cache.name}")
    else:
        print(f"classifying with the classic pre-filter ({args.workers} workers)...", flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_classify, [str(img_dir / t) for t in tiles], chunksize=16), 1):
                rows.append(r)
                if i % 500 == 0:
                    print(f"  {i}/{len(tiles)}", flush=True)
        ctx = pd.DataFrame(rows, columns=keep)

    # ---- 2. ground truth and detector output
    split = pd.read_csv(args.data / "split.csv").set_index("ImageId")
    preds = pd.read_csv(args.preds)
    matched = pd.read_csv(args.matched)
    flags = flag_predictions(preds, lab_dir, tiles)
    flags.to_csv(args.out / "wp6_pred_flags.csv", index=False)
    fp = count_false_alarms(flags, tiles)

    tdf = (ctx.merge(fp, on="image")
              .assign(n_ships=lambda d: d.image.map(split.n_ships).fillna(0).astype(int)))
    # the scheduler needs three contexts; "candidates"/"empty_sea" are both open sea
    tdf["ctx3"] = np.where(tdf.context == CLOUD, "cloud",
                  np.where(tdf.context == LAND, "coast",
                  np.where(tdf.n_ships > 0, "ships", "empty")))
    tdf.to_csv(args.out / "wp6_tiles.csv", index=False)

    sdf = matched[matched.image.isin(set(tiles))].copy()
    sdf["conf"] = sdf.conf.fillna(0.0)
    sdf = sdf.merge(tdf[["image", "ctx3", "cloud_frac"]], on="image", how="left")
    sdf.to_csv(args.out / "wp6_ships.csv", index=False)

    # ---- 3. summary
    summary = {
        "split": args.split,
        "tiles": len(tdf),
        "ships": int(sdf.shape[0]),
        "context_counts": tdf.ctx3.value_counts().to_dict(),
        "prefilter_context_counts": tdf.context.value_counts().to_dict(),
        "prefilter_ms_median": float(tdf.prefilter_ms.median()),
        "ships_per_ship_tile": float(tdf.loc[tdf.ctx3 == "ships", "n_ships"].mean()),
        "ships_per_coast_tile": float(tdf.loc[tdf.ctx3 == "coast", "n_ships"].mean()),
        "detector_recall_any_conf": float(sdf.found.mean()),
        "conf_percentiles_found": {str(q): float(np.percentile(sdf.loc[sdf.found, "conf"], q))
                                   for q in (10, 25, 50, 75, 90)},
        "false_alarms": {},
    }
    for t in FP_THRESHOLDS:
        c = f"n_fp_{t:g}"
        empty = tdf[tdf.n_ships == 0]
        summary["false_alarms"][f"conf>={t:g}"] = {
            "fp_total": int(tdf[c].sum()),
            "fp_per_tile": float(tdf[c].mean()),
            "empty_tiles_with_fp": float((empty[c] > 0).mean()),
            "recall_at_thr": float((sdf.found & (sdf.conf >= t)).mean()),
        }
    (args.out / "wp6_catalogue.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved wp6_tiles.csv ({len(tdf)} rows), wp6_ships.csv ({len(sdf)} rows), "
          f"wp6_catalogue.json in {args.out}")


if __name__ == "__main__":
    main()
