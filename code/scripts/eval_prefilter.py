"""Evaluate the classic CV pre-filter: how many ships does it keep, how much does it drop?

    python scripts/eval_prefilter.py --demo                      # synthetic tiles, no data needed
    python scripts/eval_prefilter.py --airbus-dir D:/data/airbus --n 2000
    python scripts/eval_prefilter.py --airbus-dir D:/data/airbus --split-csv data/yolo_ships/split.csv

Key numbers for the paper:
  * ship recall          -- ships covered by a candidate (the SAFETY metric: must stay high)
  * ship images kept     -- tiles with ships NOT dropped as empty/cloud
  * empty images dropped -- tiles the expensive detector never has to see (the SAVING)
  * runtime per tile     -- measured on this machine; energy = runtime x --power-w (ESTIMATE)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from sat7.prefilter import CLOUD, EMPTY, PrefilterConfig, match_ships, run_prefilter
from sat7.rle import Box, rles_to_boxes
from sat7.synthetic import make_tile


def demo_samples(n: int, seed: int):
    rng = np.random.default_rng(seed)
    for i in range(n):
        k = int(rng.choice([0, 0, 0, 1, 2, 4]))
        cloud = float(rng.choice([0.0, 0.0, 0.0, 0.3, 0.7]))
        img, boxes = make_tile(n_ships=k, cloud_cover=cloud, seed=seed * 100_000 + i)
        yield f"synthetic_{i:05d}", img, boxes


def yolo_samples(yolo_dir: Path, split: str, n: int, seed: int):
    """Tiles + ground-truth boxes from the leakage-free dataset built by WP0."""
    img_dir, lbl_dir = yolo_dir / "images" / split, yolo_dir / "labels" / split
    files = sorted(img_dir.glob("*.jpg"))
    np.random.default_rng(seed).shuffle(files)
    for f in files[:n]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        lbl = lbl_dir / (f.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().split("\n"):
                p = line.split()
                if len(p) == 5:
                    cx, cy, bw, bh = (float(v) for v in p[1:])
                    boxes.append(Box(int((cx - bw / 2) * w), int((cy - bh / 2) * h), max(1, int(bw * w)),
                                     max(1, int(bh * h))))
        yield f.name, img, boxes


def airbus_samples(airbus_dir: Path, n: int, split_csv: Path | None, seed: int):
    df = pd.read_csv(airbus_dir / "train_ship_segmentations_v2.csv")
    groups = df.groupby("ImageId")["EncodedPixels"].apply(lambda s: [r for r in s if isinstance(r, str)])
    if split_csv:
        ids = pd.read_csv(split_csv).query("split == 'test'").ImageId.tolist()
    else:
        ids = list(groups.index)
        np.random.default_rng(seed).shuffle(ids)
    for image_id in ids[:n]:
        img = cv2.imread(str(airbus_dir / "train_v2" / image_id))
        if img is not None:
            yield image_id, img, rles_to_boxes(groups[image_id])


def draw(img, gt, pred):
    out = img.copy()
    for b in gt:
        cv2.rectangle(out, (b.x - 2, b.y - 2), (b.x + b.w + 2, b.y + b.h + 2), (0, 200, 0), 2)
    for b in pred:
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 140, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true")
    src.add_argument("--airbus-dir", type=Path)
    src.add_argument("--yolo-dir", type=Path, help="dataset built by wp0_build_dataset.py")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tophat", type=int, help="override the top-hat structuring element size (px)")
    ap.add_argument("--split-csv", type=Path, help="use the test split written by airbus_to_yolo.py")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--power-w", type=float, default=0.0,
                    help="assumed device power (W) to turn runtime into an energy ESTIMATE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    if args.demo:
        samples = demo_samples(args.n, args.seed)
    elif args.yolo_dir:
        samples = yolo_samples(args.yolo_dir, args.split, args.n, args.seed)
    else:
        samples = airbus_samples(args.airbus_dir, args.n, args.split_csv, args.seed)
    cfg = PrefilterConfig(**({"tophat_ksize": args.tophat} if args.tophat else {}))
    rows, gallery = [], []
    for image_id, img, gt in samples:
        res = run_prefilter(img, cfg)
        found = match_ships(gt, res.boxes)
        rows.append({"image": image_id, "n_ships": len(gt), "n_found": sum(found),
                     "n_candidates": len(res.boxes), "context": res.context,
                     "dropped": res.context in (EMPTY, CLOUD), "runtime_ms": res.runtime_ms})
        if len(gallery) < 8 and gt:
            gallery.append(cv2.resize(draw(img, gt, res.boxes), (384, 384)))

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "demo" if args.demo else ("real" if args.yolo_dir else "airbus")
    df.to_csv(args.out / f"prefilter_{tag}.csv", index=False)

    ship_imgs, empty_imgs = df[df.n_ships > 0], df[df.n_ships == 0]
    total_ships = int(df.n_ships.sum())
    # ships inside dropped tiles are lost even if a candidate existed
    kept_found = int(df.loc[~df.dropped, "n_found"].sum())
    print(f"Pre-filter evaluation ({tag}): {len(df)} tiles, {total_ships} ships")
    print(f"  ship recall after context decision : {kept_found / max(1, total_ships):.3f}   <- safety metric")
    print(f"  ship tiles kept                    : {(~ship_imgs.dropped).mean():.3f}")
    print(f"  empty tiles dropped                : {empty_imgs.dropped.mean() if len(empty_imgs) else float('nan'):.3f}   <- compute saved")
    print(f"  candidates per kept tile (mean)    : {df.loc[~df.dropped, 'n_candidates'].mean():.2f}")
    print(f"  contexts                           : {df.context.value_counts().to_dict()}")
    rt = df.runtime_ms
    print(f"  runtime per tile (this machine)    : median {rt.median():.1f} ms, p90 {rt.quantile(0.9):.1f} ms")
    if args.power_w:
        print(f"  energy per tile (ESTIMATE, {args.power_w} W): {rt.median() / 1e3 * args.power_w * 1e3:.1f} mJ")
    if gallery:
        while len(gallery) % 4:
            gallery.append(np.zeros_like(gallery[0]))
        rows_img = [np.hstack(gallery[i:i + 4]) for i in range(0, len(gallery), 4)]
        cv2.imwrite(str(args.out / f"prefilter_{tag}_gallery.jpg"), np.vstack(rows_img))
        print(f"  gallery (green = truth, orange = candidates): {args.out / f'prefilter_{tag}_gallery.jpg'}")


if __name__ == "__main__":
    main()
