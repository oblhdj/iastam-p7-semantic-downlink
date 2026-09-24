"""WP12b -- if a tile seam cuts a ship in half, does the detector still see it?

The SAHI decision has been argued from geometry alone. Geometry gives P(cut): on a real
10,000 px swath tiled at 768 px with no overlap, a 200 px ship is cut 45% of the time and a
380 px ship 74.5%. But **P(cut) is not P(miss)** -- a ship with 60% of its hull visible may
well still be detected, and then the seam costs nothing. Nobody had measured that, so the
"should we pay 1.47x compute for SAHI" question had no empirical input.

This measures the missing curve: **recall as a function of the visible fraction of a ship.**

Method. Take real test ships. For each one, crop the tile along a vertical line through the
ship so that only a fraction ``f`` of its width is left inside, and run the detector on that
crop -- which is exactly what a tile boundary does: the rest of the hull is in the next tile,
so this tile simply ends. Matching is against the **clipped** ground-truth box (the visible
part), because the question is "does the satellite know a ship is here", and finding the
visible half is a success, not a failure.

Combined with the geometry, this gives the expected ship loss per frame for each policy, which
is what the SAHI decision actually needs.

    .venv312/Scripts/python.exe scripts/wp12b_cut_recall.py --ships 400

Outputs: results/wp12b_cut_recall.json, wp12b_cut_recall.csv, wp12b_cut_recall.png
"""
from __future__ import annotations

import argparse
import json
import sys
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

from wp11_integration_demo import gt_boxes, iou

TILE = 768
FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SIZE_BINS = [(0, 32, "small (<32 px)"), (32, 96, "medium (32-96 px)"), (96, 1e9, "large (>96 px)")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ships", type=int, default=400)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--frame", type=int, default=10_000, help="swath size for the loss estimate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    dev = 0 if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    img_dir, lbl_dir = args.data / "images" / "test", args.data / "labels" / "test"
    rng = np.random.default_rng(args.seed)
    files = sorted(img_dir.glob("*.jpg"))

    # collect real ships, one per tile so each crop is independent
    targets = []
    for i in rng.permutation(len(files)):
        f = files[i]
        gts = gt_boxes(lbl_dir / (f.stem + ".txt"))
        gts = [g for g in gts if g[0] - g[2] / 2 > 8 and g[2] >= 8]    # need room to cut
        if gts:
            targets.append((f, max(gts, key=lambda g: g[2])))
        if len(targets) >= args.ships:
            break
    print(f"=== WP12b: {len(targets)} real ships, cut at {len(FRACTIONS)} visible fractions "
          f"({len(targets)*len(FRACTIONS)} detector runs), device {dev} ===\n")

    rows = []
    for n, (f, g) in enumerate(targets, 1):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        cx, cy, w, h = g
        left = cx - w / 2
        for fr in FRACTIONS:
            # cut so that `fr` of the ship's width remains inside the tile
            cut = int(round(left + fr * w))
            cut = max(16, min(TILE, cut))
            crop = img[:, :cut]
            if crop.shape[1] < 16:
                continue
            vis_w = min(cut, cx + w / 2) - left           # visible width in pixels
            if vis_w <= 0:
                continue
            clipped = (left + vis_w / 2, cy, vis_w, h)     # GT box clipped to what is visible
            res = model.predict(crop, imgsz=args.imgsz, conf=args.conf, device=dev, verbose=False)
            r = res[0]
            best = 0.0
            for b in r.boxes.xywh.cpu().numpy():
                best = max(best, iou(clipped, (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
            rows.append({"image": f.name, "size_px": max(w, h), "visible_fraction": fr,
                         "visible_px": vis_w, "found": best >= 0.5, "best_iou": best})
        if n % 50 == 0:
            print(f"  {n}/{len(targets)} ships", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "wp12b_cut_recall.csv", index=False)

    print("\nRecall vs how much of the ship stayed inside the tile:")
    print(f"{'visible':>8s} {'recall':>8s} {'small':>8s} {'medium':>8s} {'large':>8s}")
    curve = {}
    for fr in FRACTIONS:
        s = df[df.visible_fraction == fr]
        if not len(s):
            continue
        row = [f"{fr:8.0%}", f"{s.found.mean():8.3f}"]
        for lo, hi, _ in SIZE_BINS:
            b = s[(s.size_px >= lo) & (s.size_px < hi)]
            row.append(f"{b.found.mean():8.3f}" if len(b) else f"{'-':>8s}")
        print(" ".join(row))
        curve[fr] = float(s.found.mean())

    full = curve.get(1.0, 1.0)
    print(f"\nIntact-ship recall (visible 100%): {full:.3f}  <- the reference")

    # ---------------------------------------------------------------- what it costs on a swath
    # A ship cut by a seam keeps a uniformly distributed fraction of itself in each tile; the
    # larger piece is what the detector gets, so the visible fraction is U(0.5, 1).
    keep = [f for f in FRACTIONS if f >= 0.5]
    p_found_if_cut = float(np.mean([curve[f] for f in keep])) if keep else float("nan")
    print(f"P(still detected | cut, larger piece visible) = {p_found_if_cut:.3f}")

    n_tiles = (args.frame // TILE + 1) ** 2
    stride = int(TILE * 0.8)
    n_sahi = ((args.frame - TILE) // stride + 2) ** 2
    print(f"\nOn a {args.frame}x{args.frame} swath tiled at {TILE} px "
          f"({n_tiles} tiles, SAHI@20% {n_sahi} = {n_sahi/n_tiles:.2f}x compute):")
    print(f"{'ship size':>20s} {'P(cut)':>8s} {'loss no-overlap':>16s} {'loss SAHI':>10s}")
    loss = {}
    for L, label in ((43, "median 43 px"), (100, "medium 100 px"),
                     (200, "large 200 px"), (380, "largest 380 px")):
        p_cut = 1 - (1 - min(L / TILE, 1.0)) ** 2
        l_no = p_cut * (full - p_found_if_cut)
        loss[label] = {"size_px": L, "p_cut": p_cut, "loss_no_overlap": l_no}
        print(f"{label:>20s} {p_cut:8.1%} {l_no:16.1%} {0.0:10.1%}")

    report = {"ships": len(targets), "runs": int(len(df)),
              "recall_vs_visible_fraction": curve,
              "intact_recall": full,
              "p_detected_given_cut": p_found_if_cut,
              "frame_px": args.frame, "tiles_no_overlap": n_tiles, "tiles_sahi20": n_sahi,
              "sahi_compute_x": n_sahi / n_tiles, "expected_loss": loss}
    (args.out / "wp12b_cut_recall.json").write_text(json.dumps(report, indent=2))

    # ---------------------------------------------------------------- figure
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    xs = sorted(curve)
    a0.plot([100 * x for x in xs], [curve[x] for x in xs], "o-", color="#1B6CA8", lw=2)
    for lo, hi, name in SIZE_BINS:
        ys = []
        for fr in xs:
            b = df[(df.visible_fraction == fr) & (df.size_px >= lo) & (df.size_px < hi)]
            ys.append(b.found.mean() if len(b) else np.nan)
        a0.plot([100 * x for x in xs], ys, "--", lw=1, alpha=0.8, label=name)
    a0.axvline(50, color="#C0392B", ls=":", lw=1.5)
    a0.text(51, 0.05, "worst case for a\nseam-cut ship", color="#C0392B", fontsize=8)
    a0.set(xlabel="% of the ship left inside the tile", ylabel="recall", ylim=(0, 1.02))
    a0.set_title("Does a cut ship still get detected?", fontsize=10)
    a0.legend(fontsize=7.5, frameon=False, loc="lower right")
    a0.grid(alpha=0.25)

    labels = list(loss)
    a1.bar(range(len(labels)), [100 * loss[k]["p_cut"] for k in labels],
           color="#C0392B", alpha=0.45, label="P(cut by a seam)")
    a1.bar(range(len(labels)), [100 * loss[k]["loss_no_overlap"] for k in labels],
           color="#C0392B", label="ships actually lost")
    a1.set_xticks(range(len(labels)), [k.split()[0] for k in labels])
    a1.set_ylabel("% of ships")
    a1.set_title(f"Cut is not lost: {args.frame//1000}k swath, no overlap", fontsize=10)
    a1.legend(fontsize=8, frameon=False)
    a1.grid(alpha=0.25, axis="y")
    for ax in (a0, a1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "wp12b_cut_recall.png", dpi=160)
    print(f"\nSaved wp12b_cut_recall.json|.csv|.png in {args.out}")


if __name__ == "__main__":
    main()
