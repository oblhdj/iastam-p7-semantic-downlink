"""Convert the Airbus Ship Detection dataset into a YOLO dataset (one class: ship).

Expected input (Kaggle "airbus-ship-detection"):
    <airbus-dir>/train_ship_segmentations_v2.csv
    <airbus-dir>/train_v2/*.jpg

    python scripts/airbus_to_yolo.py --airbus-dir D:/data/airbus --out data/yolo_ships \
        --max-ship-images 8000 --empty-ratio 0.25

Output (Ultralytics layout):
    <out>/images/{train,val,test}/*.jpg   (hard links when possible, else copies)
    <out>/labels/{train,val,test}/*.txt   "0 cx cy w h" per ship, normalised
    <out>/data.yaml                       for `yolo detect train data=<out>/data.yaml`
    <out>/split.csv                       ImageId, split, n_ships (reused by eval_prefilter.py)

WARNING (data leakage): Airbus tiles are 768x768 crops of larger scenes, and
neighbouring crops overlap. A random split by ImageId can put the same ship in
train and test and inflate scores. Run a near-duplicate check (FAISS) before
reporting final test numbers.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sat7.rle import AIRBUS_SHAPE, rles_to_boxes


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)          # no extra disk space on the same drive
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--airbus-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--max-ship-images", type=int, default=8000, help="0 = all")
    ap.add_argument("--empty-ratio", type=float, default=0.25,
                    help="empty images kept per ship image (negatives teach the model the sea)")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv = args.airbus_dir / "train_ship_segmentations_v2.csv"
    img_dir = args.airbus_dir / "train_v2"
    if not csv.exists() or not img_dir.exists():
        sys.exit(f"Expected {csv} and {img_dir}")

    df = pd.read_csv(csv)
    groups = df.groupby("ImageId")["EncodedPixels"].apply(lambda s: [r for r in s if isinstance(r, str)])
    ship_ids = [i for i, r in groups.items() if r]
    empty_ids = [i for i, r in groups.items() if not r]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ship_ids)
    rng.shuffle(empty_ids)
    if args.max_ship_images:
        ship_ids = ship_ids[:args.max_ship_images]
    empty_ids = empty_ids[:int(len(ship_ids) * args.empty_ratio)]
    ids = ship_ids + empty_ids
    rng.shuffle(ids)

    n = len(ids)
    n_test, n_val = int(n * args.test), int(n * args.val)
    split = {i: "test" for i in ids[:n_test]}
    split.update({i: "val" for i in ids[n_test:n_test + n_val]})
    split.update({i: "train" for i in ids[n_test + n_val:]})

    h, w = AIRBUS_SHAPE
    rows, n_boxes, missing = [], 0, 0
    for k, image_id in enumerate(ids, 1):
        src = img_dir / image_id
        if not src.exists():
            missing += 1
            continue
        s = split[image_id]
        boxes = rles_to_boxes(groups[image_id])
        place(src, args.out / "images" / s / image_id)
        lbl = args.out / "labels" / s / (Path(image_id).stem + ".txt")
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("".join("0 %.6f %.6f %.6f %.6f\n" % b.to_yolo(w, h) for b in boxes))
        rows.append({"ImageId": image_id, "split": s, "n_ships": len(boxes)})
        n_boxes += len(boxes)
        if k % 1000 == 0:
            print(f"  {k}/{n} images")

    pd.DataFrame(rows).to_csv(args.out / "split.csv", index=False)
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: ship\n")
    counts = pd.DataFrame(rows).groupby("split").size().to_dict()
    print(f"Done: {len(rows)} images ({len(ship_ids)} with ships, {len(empty_ids)} empty), "
          f"{n_boxes} ship boxes, splits {counts}, missing files {missing}")
    print(f"Train with:  yolo detect train data={(args.out / 'data.yaml').as_posix()} model=yolov8n.pt imgsz=768")


if __name__ == "__main__":
    main()
