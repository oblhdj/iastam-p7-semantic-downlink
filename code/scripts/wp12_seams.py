"""WP12 -- tile seams and SAHI, measured instead of derived.

ARCHITECTURE.md section E item 2: "`sahi_cost.png` is a geometric formula, not an experiment."
The Phase-2 guide argues SAHI costs 1.51x compute, that large ships have a 53-84% chance of
being cut by a tile border, and that an "edge-triggered re-inference" alternative would cost
about 1.01x. All three are arithmetic on box sizes. None was ever run through the detector.

Why it was parked, and why that was reasonable: the Airbus tiles are **already 768 px and
already cut**, so there is no wide frame to slice and SAHI changes nothing for the current
numbers. But the seam question is still testable -- treat each real 768 px tile as the "frame"
and cut it ourselves. Then the ships that straddle the cut are real ships, the detector is the
real detector, and the three policies can be compared on recall and compute for real:

  whole        one inference on the 768 px tile            -- what we do today (reference)
  grid         2x2 non-overlapping 384 px slices           -- naive tiling, seams unprotected
  sahi         overlapping slices, 20% overlap             -- the standard fix
  edge         grid + one extra slice centred on the seam, only where the cheap classic
               pre-filter sees a blob touching a slice border  -- our proposed alternative

Every slice is inferred at the model's native 768 px (what SAHI does in practice), so compute
is proportional to the number of inferences -- the same accounting the Phase-2 guide used.

    .venv312/Scripts/python.exe scripts/wp12_seams.py --tiles 300

Outputs: results/wp12_seams.json, wp12_seams.csv, wp12_seams.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch

from sat7.prefilter import PrefilterConfig, run_prefilter
from wp11_integration_demo import gt_boxes, iou

TILE = 768


def slices_grid(half: int = TILE // 2):
    """2x2 non-overlapping slices: (x0, y0, w, h)."""
    return [(x, y, half, half) for y in (0, half) for x in (0, half)]


def slices_sahi(overlap: float = 0.2, half: int = TILE // 2):
    """Overlapping slices at the given overlap ratio, clipped to the tile."""
    stride = int(round(half * (1 - overlap)))
    xs = sorted({min(x, TILE - half) for x in range(0, TILE, stride)})
    ys = sorted({min(y, TILE - half) for y in range(0, TILE, stride)})
    return [(x, y, half, half) for y in ys for x in xs]


def straddles(g, half: int = TILE // 2) -> bool:
    """Does this ground-truth box cross the 2x2 grid's seam lines?"""
    cx, cy, w, h = g
    return (cx - w / 2 < half < cx + w / 2) or (cy - h / 2 < half < cy + h / 2)


def detect_slices(model, img, slc, imgsz: int, conf: float, dev):
    """Run the detector on each slice, return boxes in FULL-tile coordinates."""
    crops = [img[y:y + h, x:x + w] for (x, y, w, h) in slc]
    res = model.predict(crops, imgsz=imgsz, conf=conf, device=dev, verbose=False)
    out = []
    for (x, y, w, h), r in zip(slc, res):
        for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append((float(b[0]) + x, float(b[1]) + y, float(b[2]), float(b[3]), float(c)))
    return out


def nms(boxes, thr: float = 0.5):
    """Merge duplicate detections from overlapping slices, highest confidence first."""
    keep = []
    for b in sorted(boxes, key=lambda z: -z[4]):
        if all(iou(b[:4], k[:4]) < thr for k in keep):
            keep.append(b)
    return keep


def recall_of(preds, gts, thr: float = 0.5):
    """Greedy IoU matching, same rule as WP1. Returns a found-flag per ground-truth ship."""
    used, found = set(), []
    for g in gts:
        best, bi = 0.0, -1
        for k, p in enumerate(preds):
            if k in used:
                continue
            v = iou(g, p[:4])
            if v > best:
                best, bi = v, k
        if best >= thr:
            used.add(bi)
        found.append(best >= thr)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=300, help="tiles WITH ships to test")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    dev = 0 if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    img_dir = args.data / "images" / "test"
    lbl_dir = args.data / "labels" / "test"
    files = sorted(img_dir.glob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    # only tiles that actually contain ships can say anything about seams
    with_ships = [f for f in files if gt_boxes(lbl_dir / (f.stem + ".txt"))]
    pick = [with_ships[i] for i in sorted(rng.choice(len(with_ships),
                                                     min(args.tiles, len(with_ships)),
                                                     replace=False))]
    print(f"=== WP12 seams: {len(pick)} real tiles with ships, device {dev} ===\n")

    grid, sahi = slices_grid(), slices_sahi(args.overlap)
    print(f"slices per frame: grid {len(grid)}, sahi@{args.overlap:.0%} {len(sahi)}, "
          f"whole 1  ->  SAHI costs {len(sahi)/len(grid):.2f}x the grid\n")

    rows = []
    t0 = time.perf_counter()
    for n, f in enumerate(pick, 1):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        gts = gt_boxes(lbl_dir / (f.stem + ".txt"))
        pre = run_prefilter(img, PrefilterConfig())

        # --- edge-triggered: extra slice on a seam only if a classic blob touches a grid border
        half, margin = TILE // 2, 24
        need_v = need_h = False
        for b in pre.boxes:
            bx, by, bw, bh = (b[0], b[1], b[2], b[3]) if len(b) == 4 else (b[0], b[1], 0, 0)
            need_v |= abs(bx - half) < margin or abs(bx + bw - half) < margin
            need_h |= abs(by - half) < margin or abs(by + bh - half) < margin
        extra = []
        if need_v:
            extra.append((half // 2, 0, half, TILE - 1) if False else (half - half // 2, 0, half, half))
            extra.append((half - half // 2, half, half, half))
        if need_h:
            extra.append((0, half - half // 2, half, half))
            extra.append((half, half - half // 2, half, half))

        policies = {
            "whole": [(0, 0, TILE, TILE)],
            "grid": grid,
            "sahi": sahi,
            "edge": grid + extra,
        }
        per = {}
        for name, slc in policies.items():
            preds = detect_slices(model, img, slc, args.imgsz, args.conf, dev)
            if len(slc) > 1:
                preds = nms(preds)
            per[name] = recall_of(preds, gts)

        for i, g in enumerate(gts):
            rows.append({"image": f.name, "size_px": max(g[2], g[3]), "straddles": straddles(g),
                         **{f"found_{k}": v[i] for k, v in per.items()}})
        for name, slc in policies.items():
            rows[-1][f"slices_{name}"] = len(slc)
        if n % 50 == 0:
            print(f"  {n}/{len(pick)} tiles", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "wp12_seams.csv", index=False)
    pol = ["whole", "grid", "sahi", "edge"]

    # compute: mean slices per frame, normalised to the naive grid
    slices = {p: float(df[f"slices_{p}"].dropna().mean()) for p in pol}
    print(f"\n{len(df)} ground-truth ships, {int(df.straddles.sum())} of them "
          f"({df.straddles.mean():.1%}) straddle a 2x2 seam\n")

    print(f"{'policy':8s} {'recall':>8s} {'on seam':>9s} {'off seam':>9s} "
          f"{'slices/frame':>13s} {'compute':>8s}")
    report = {"tiles": len(pick), "ships": int(len(df)),
              "straddling_ships": int(df.straddles.sum()),
              "straddling_share": float(df.straddles.mean()), "policies": {}}
    for p in pol:
        col = df[f"found_{p}"]
        r_all = float(col.mean())
        r_on = float(col[df.straddles].mean()) if df.straddles.any() else float("nan")
        r_off = float(col[~df.straddles].mean())
        cost = slices[p] / slices["grid"]
        report["policies"][p] = {"recall": r_all, "recall_on_seam": r_on,
                                 "recall_off_seam": r_off,
                                 "slices_per_frame": slices[p], "compute_vs_grid": cost}
        print(f"{p:8s} {r_all:8.4f} {r_on:9.4f} {r_off:9.4f} {slices[p]:13.2f} {cost:7.2f}x")

    # ---- the claim being tested
    w, g = report["policies"]["whole"], report["policies"]["grid"]
    s, e = report["policies"]["sahi"], report["policies"]["edge"]
    print("\nWhat this says:")
    print(f"  cutting a tile costs {100*(w['recall']-g['recall']):+.1f} pts of recall overall, "
          f"{100*(w['recall_on_seam']-g['recall_on_seam']):+.1f} pts on ships that straddle the cut")
    print(f"  SAHI recovers {100*(s['recall_on_seam']-g['recall_on_seam']):+.1f} pts on seam ships "
          f"for {s['compute_vs_grid']:.2f}x compute")
    print(f"  edge-triggered recovers {100*(e['recall_on_seam']-g['recall_on_seam']):+.1f} pts "
          f"for {e['compute_vs_grid']:.2f}x compute")
    report["elapsed_s"] = time.perf_counter() - t0
    (args.out / "wp12_seams.json").write_text(json.dumps(report, indent=2))

    # ---- figure
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(pol))
    a0.bar(x - 0.2, [report["policies"][p]["recall_on_seam"] for p in pol], 0.4,
           label="ships on a seam", color="#C0392B")
    a0.bar(x + 0.2, [report["policies"][p]["recall_off_seam"] for p in pol], 0.4,
           label="ships away from a seam", color="#1B6CA8")
    a0.set_xticks(x, pol)
    a0.set_ylabel("recall")
    a0.set_title("Seams hurt the ships that cross them", fontsize=10)
    a0.legend(fontsize=8, frameon=False)
    a0.grid(alpha=0.25, axis="y")

    for p, c in zip(pol, ["#9CA3AF", "#C0392B", "#7D3C98", "#2E8B57"]):
        a1.scatter(report["policies"][p]["compute_vs_grid"],
                   report["policies"][p]["recall_on_seam"], s=90, color=c, label=p, zorder=3)
    a1.set_xlabel("compute, relative to naive 2x2 grid")
    a1.set_ylabel("recall on ships crossing a seam")
    a1.set_title("What each policy buys for what it costs", fontsize=10)
    a1.legend(fontsize=8, frameon=False)
    a1.grid(alpha=0.25)
    for ax in (a0, a1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "wp12_seams.png", dpi=160)
    print(f"\nSaved wp12_seams.json|.csv|.png in {args.out}")


if __name__ == "__main__":
    main()
