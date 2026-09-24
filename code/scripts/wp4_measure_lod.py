"""WP4 -- measure the real cost of each level of detail on Airbus ships.

Until now the scheduler used *assumed* sizes (L1 = 2 KB, L2 = 30 KB...). This script
measures them on real ship crops from the leakage-free dataset:

  * crop each ground-truth ship with a margin,
  * encode it as JPEG at several qualities (and JPEG2000 if OpenCV supports it),
  * record the real number of bytes and the resulting image quality (PSNR),
  * also measure the thumbnail of a whole tile and a compressed coastal tile.

Later (once the detector exists) the same crops are re-run through the detector to get
g(level) = probability that the ground can still confirm the ship.

    python scripts/wp4_measure_lod.py --yolo-dir data/yolo_ships --n 400
Outputs: results/wp4_lod_sizes.csv, results/wp4_lod_sizes.png, printed summary table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUALITIES = (20, 40, 60, 80, 95)
MARGINS = {"tight": 0.15, "context": 0.6, "wake": 1.5}   # crop margin as a fraction of ship size


def encode_size(img: np.ndarray, ext: str, param: list[int]) -> tuple[int, np.ndarray | None]:
    """(bytes, decoded image) or (0, None) if the codec refuses this image."""
    try:
        ok, buf = cv2.imencode(ext, img, param)
    except cv2.error:                       # e.g. JPEG2000 refuses very small crops
        return 0, None
    if not ok:
        return 0, None
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return (len(buf), dec) if dec is not None else (0, None)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-6 else float(10 * np.log10(255.0 ** 2 / mse))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-dir", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=400, help="tiles to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    img_dir = args.yolo_dir / "images" / args.split
    lbl_dir = args.yolo_dir / "labels" / args.split
    files = sorted(img_dir.glob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)

    has_jp2 = encode_size(np.zeros((32, 32, 3), np.uint8), ".jp2", [])[0] > 0
    rows, tiles_used = [], 0
    for f in files:
        if tiles_used >= args.n:
            break
        img = cv2.imread(str(f))
        lbl = lbl_dir / (f.stem + ".txt")
        if img is None or not lbl.exists():
            continue
        h, w = img.shape[:2]
        boxes = []
        for line in lbl.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                cx, cy, bw, bh = (float(v) * np.array([w, h, w, h])[i] for i, v in enumerate(p[1:]))
                boxes.append((cx, cy, bw, bh))
        if not boxes:
            continue
        tiles_used += 1

        # whole-tile products: thumbnail (safety net) and compressed tile (coast / port)
        thumb = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        rows.append({"product": "thumbnail 96px", "quality": 60, "bytes": encode_size(thumb, ".jpg",
                     [cv2.IMWRITE_JPEG_QUALITY, 60])[0], "ship_px": np.nan, "psnr": np.nan})
        for q in (40, 60):
            n_bytes, dec = encode_size(img, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, q])
            if dec is not None:
                rows.append({"product": f"whole tile q{q}", "quality": q, "bytes": n_bytes, "ship_px": np.nan,
                             "psnr": psnr(img, dec)})

        for cx, cy, bw, bh in boxes:
            size = max(bw, bh)
            for name, margin in MARGINS.items():
                half = size * (1 + margin) / 2
                x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
                x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                for q in QUALITIES:
                    n_bytes, dec = encode_size(crop, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, q])
                    if dec is not None:
                        rows.append({"product": f"crop {name} q{q}", "quality": q, "bytes": n_bytes,
                                     "ship_px": size, "psnr": psnr(crop, dec)})
                if has_jp2:
                    n_bytes, dec = encode_size(crop, ".jp2", [])
                    if dec is not None:
                        rows.append({"product": f"crop {name} jp2", "quality": -1, "bytes": n_bytes,
                                     "ship_px": size, "psnr": psnr(crop, dec)})

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "wp4_lod_sizes.csv", index=False)
    summary = df.groupby("product").agg(n=("bytes", "size"), median_bytes=("bytes", "median"),
                                        p90_bytes=("bytes", lambda s: s.quantile(0.9)),
                                        median_psnr=("psnr", "median")).sort_values("median_bytes")
    print(f"WP4: {tiles_used} tiles, {int((df.ship_px.notna()).sum() / (len(QUALITIES) + int(has_jp2)))} ship crops "
          f"(JPEG2000 available: {has_jp2})\n")
    print(summary.round(1).to_string())
    print("\nAssumed so far -> measured (median):")
    for lbl, assumed, product in [("L1 small chip", 2000, "crop tight q40"),
                                  ("L2 ROI", 30000, "crop context q80"),
                                  ("coastal tile", 80000, "whole tile q60"),
                                  ("thumbnail", 1000, "thumbnail 96px")]:
        if product in summary.index:
            got = summary.loc[product, "median_bytes"]
            print(f"  {lbl:14s} {assumed:>7,} B  ->  {int(got):>7,} B   ({product})")

    crops = df[df.ship_px.notna() & df["product"].str.startswith("crop")]
    if len(crops):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for name in MARGINS:
            sub = crops[crops["product"].str.contains(f"crop {name} q")]
            g = sub.groupby("quality").bytes.median()
            axes[0].plot(g.index, g.values / 1024, "o-", label=f"{name} margin")
        axes[0].set_xlabel("JPEG quality")
        axes[0].set_ylabel("median size (KB)")
        axes[0].set_title("Cost of one ship crop", fontsize=10)
        axes[0].legend(frameon=False, fontsize=8)
        sub = crops[crops["product"] == "crop context q60"]
        axes[1].scatter(sub.ship_px, sub.bytes / 1024, s=6, alpha=0.35, color="#1B6CA8")
        axes[1].set_xlabel("ship size (px)")
        axes[1].set_ylabel("size (KB)")
        axes[1].set_title("Bigger ships cost more (context crop, q60)", fontsize=10)
        for a in axes:
            a.spines[["top", "right"]].set_visible(False)
            a.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.out / "wp4_lod_sizes.png", dpi=160)
        print(f"\nSaved {args.out / 'wp4_lod_sizes.csv'} and wp4_lod_sizes.png")


if __name__ == "__main__":
    main()
