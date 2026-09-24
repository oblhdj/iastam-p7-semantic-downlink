"""WP7 step 1: measure the two products that actually cost us bytes.

On the real workload (WP6) the downlink budget breaks down as:
    coastal tiles 61.7% | thumbnails 23.7% | ship information (L0+L1+L2) 13.1% | false alarms 1.5%
So 86% of a "semantic downlink" is not semantic. Before changing the design we measure the
cheaper options on real tiles, rather than assuming what a smaller thumbnail costs.

Two ladders, both on real Airbus tiles:
  * thumbnail: 48 / 64 / 96 / 128 px at q30 / q60 / q80
  * coastal tile: q30..q80, plus the ship-crop-mosaic alternative (send only the ROIs
    around every detected ship instead of the whole tile)

For the thumbnail we also measure whether a ship is still *visible* in it, because a
thumbnail whose ships have vanished is not a safety net, however cheap it is. Visibility
here = the ship's peak brightness stands above the local sea background by >= 3 sigma
after downscaling, evaluated at the ship's own position from the ground-truth box.

Output: results/wp7_cheap_products.csv|.json|.png

    python scripts/wp7_measure_cheap_products.py --n 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

THUMB_PX = (48, 64, 96, 128)
THUMB_Q = (30, 60, 80)
TILE_Q = (30, 40, 50, 60, 80)


def encode_size(img: np.ndarray, quality: int) -> tuple[int, np.ndarray | None]:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return 0, None
    return len(buf), cv2.imdecode(buf, cv2.IMREAD_COLOR)


def ship_visible(thumb_gray: np.ndarray, box, tile_hw, k_sigma: float = 3.0) -> bool:
    """Does this ship still stand out from the sea in the downscaled thumbnail?"""
    th, tw = thumb_gray.shape
    H, W = tile_hw
    cx, cy, bw, bh = box
    x = int(round(cx / W * tw))
    y = int(round(cy / H * th))
    r = max(1, int(round(max(bw / W * tw, bh / H * th) / 2)))
    x0, x1 = max(0, x - r), min(tw, x + r + 1)
    y0, y1 = max(0, y - r), min(th, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    patch = thumb_gray[y0:y1, x0:x1].astype(float)
    bg = thumb_gray.astype(float).copy()
    bg[y0:y1, x0:x1] = np.nan                      # sea = everything but this ship
    mu, sd = np.nanmean(bg), np.nanstd(bg)
    return bool(patch.max() >= mu + k_sigma * max(sd, 1e-6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-dir", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=400, help="tiles with ships to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    img_dir = args.yolo_dir / "images" / args.split
    lbl_dir = args.yolo_dir / "labels" / args.split
    files = sorted(img_dir.glob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)

    rows, vis_rows, used = [], [], 0
    for f in files:
        if used >= args.n:
            break
        img = cv2.imread(str(f))
        lbl = lbl_dir / (f.stem + ".txt")
        if img is None or not lbl.exists():
            continue
        H, W = img.shape[:2]
        boxes = []
        for line in lbl.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                boxes.append(tuple(float(v) * s for v, s in zip(p[1:], (W, H, W, H))))
        if not boxes:
            continue
        used += 1

        # ---- thumbnail ladder: bytes and whether the ships survive the downscale
        for px in THUMB_PX:
            small = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
            for q in THUMB_Q:
                n_bytes, dec = encode_size(small, q)
                rows.append({"product": f"thumbnail {px}px", "quality": q, "bytes": n_bytes})
                if dec is None:
                    continue
                gray = cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY)
                for b in boxes:
                    vis_rows.append({"px": px, "quality": q, "ship_px": max(b[2], b[3]),
                                     "visible": ship_visible(gray, b, (H, W))})

        # ---- coastal tile ladder (bytes AND quality, measured on the same tiles)
        for q in TILE_Q:
            n_bytes, dec = encode_size(img, q)
            mse = np.mean((img.astype(float) - dec.astype(float)) ** 2) if dec is not None else np.nan
            rows.append({"product": "whole tile", "quality": q, "bytes": n_bytes,
                         "psnr": 99.0 if mse == 0 else 20 * np.log10(255 / np.sqrt(mse))})

        # ---- alternative: a mosaic of ROIs around every ship, instead of the whole tile
        for q, pad in ((60, 1.5), (80, 1.5)):
            total = 0
            for cx, cy, bw, bh in boxes:
                s = max(bw, bh) * pad
                x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
                x1, y1 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
                if x1 > x0 and y1 > y0:
                    total += encode_size(img[y0:y1, x0:x1], q)[0]
            rows.append({"product": f"ROI mosaic x{pad:g}", "quality": q, "bytes": total,
                         "n_ships": len(boxes)})

    d = pd.DataFrame(rows)
    v = pd.DataFrame(vis_rows)
    d.to_csv(args.out / "wp7_cheap_products.csv", index=False)

    med = d.groupby(["product", "quality"]).bytes.median().rename("median_bytes").reset_index()
    vis = v.groupby(["px", "quality"]).visible.mean().rename("ship_visible").reset_index()
    print(f"{used} real tiles\n")
    print("Thumbnail ladder (median bytes | fraction of ships still visible):")
    t = (med[med["product"].str.startswith("thumbnail")]
         .assign(px=lambda x: x["product"].str.extract(r"(\d+)").astype(int))
         .merge(vis, on=["px", "quality"]))
    print(t[["product", "quality", "median_bytes", "ship_visible"]].to_string(index=False))
    print("\nCoastal tile ladder (median bytes and PSNR, measured on the same tiles):")
    q_tbl = (d[~d["product"].str.startswith("thumbnail")]
             .groupby(["product", "quality"])
             .agg(median_bytes=("bytes", "median"), psnr=("psnr", "median")).reset_index())
    print(q_tbl.to_string(index=False))

    summary = {
        "tiles": used,
        "thumbnail": {f"{r.px}px_q{r.quality}": {"bytes": float(r.median_bytes),
                                                 "ship_visible": float(r.ship_visible)}
                      for r in t.itertuples()},
        "whole_tile": {f"q{r.quality}": {"bytes": float(r.median_bytes), "psnr": float(r.psnr)}
                       for r in q_tbl[q_tbl["product"] == "whole tile"].itertuples()},
        "roi_mosaic": {f"{r.product} q{r.quality}": float(r.median_bytes)
                       for r in med[med["product"].str.startswith("ROI")].itertuples()},
    }
    (args.out / "wp7_cheap_products.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for px, g in t.groupby("px"):
        axes[0].plot(g.median_bytes, g.ship_visible, "o-", label=f"{px} px")
    axes[0].set(xlabel="bytes per thumbnail (median, REAL)", ylabel="ships still visible",
                title="Thumbnail: what a cheaper safety net costs")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.25)
    wt = med[med["product"] == "whole tile"]
    axes[1].plot(wt.quality, wt.median_bytes / 1e3, "o-", color="#C0392B", label="whole tile")
    for name, g in med[med["product"].str.startswith("ROI")].groupby("product"):
        axes[1].plot(g.quality, g.median_bytes / 1e3, "s--", label=name)
    axes[1].set(xlabel="JPEG quality", ylabel="kB per coastal tile",
                title="Coastal tile vs a mosaic of ship ROIs")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.25)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"WP7 -- cheaper products, MEASURED on {used} real Airbus tiles", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "wp7_cheap_products.png", dpi=160)
    print(f"\nSaved wp7_cheap_products.csv|.json|.png in {args.out}")


if __name__ == "__main__":
    main()
