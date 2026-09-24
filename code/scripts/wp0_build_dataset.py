"""WP0 -- build a leakage-free YOLO dataset from the Airbus zip.

Reads the images *directly from the zip* (no 30 GB extraction), finds near-duplicate
tiles, groups them, splits whole groups into train/val/test, extracts only the selected
images, writes YOLO labels and a report.

    python scripts/wp0_build_dataset.py --zip C:/Users/.../airbus-ship-detection.zip
    python scripts/wp0_build_dataset.py --zip ... --all-ships --workers 12   # full run

Outputs
    <out>/images/{train,val,test}/*.jpg     copied byte-for-byte from the zip
    <out>/labels/{train,val,test}/*.txt     YOLO boxes: "0 cx cy w h" (normalised)
    <out>/data.yaml                         for `yolo detect train data=...`
    <out>/split.csv                         ImageId, group, split, n_ships
    <out>/duplicates.csv                    every duplicate pair with its evidence
    results/wp0_report.md                   numbers for the paper
    results/wp0_groups.png                  group-size histogram + split composition
    results/wp0_duplicates.jpg              example duplicate pairs (for the slides)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sat7.dedup import (WINDOWS, OrbVerifier, build_groups, check_no_leakage, group_aware_split,
                        signatures, thumb_gray, thumb_vector)
from sat7.rle import AIRBUS_SHAPE, rles_to_boxes

CSV_NAME = "train_ship_segmentations_v2.csv"
IMG_DIR = "train_v2"


class ZipReader:
    """Thread-safe reader: one ZipFile handle per thread."""

    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()

    @property
    def zf(self) -> zipfile.ZipFile:
        if not hasattr(self._local, "zf"):
            self._local.zf = zipfile.ZipFile(self.path)
        return self._local.zf

    def raw(self, image_id: str) -> bytes:
        return self.zf.read(f"{IMG_DIR}/{image_id}")

    def gray(self, image_id: str) -> np.ndarray | None:
        buf = np.frombuffer(self.raw(image_id), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def montage(reader: ZipReader, pairs: list[dict], out: Path, k: int = 4, size: int = 300) -> None:
    rows = []
    for p in pairs[:k]:
        imgs = []
        for key in ("a", "b"):
            buf = np.frombuffer(reader.raw(p[key]), np.uint8)
            img = cv2.resize(cv2.imdecode(buf, cv2.IMREAD_COLOR), (size, size))
            cv2.putText(img, p[key][:9], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            imgs.append(img)
        pair = np.hstack(imgs)
        tag = f"coarse={p.get('corr')} detail={p.get('detail')} {p['method']}" + \
              (f" inliers={p['inliers']}" if p.get("inliers") else "")
        cv2.putText(pair, tag, (6, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
        rows.append(pair)
    if rows:
        cv2.imwrite(str(out), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--max-ship-images", type=int, default=8000)
    ap.add_argument("--all-ships", action="store_true", help="use all 42 556 images with ships")
    ap.add_argument("--empty-ratio", type=float, default=0.25, help="empty tiles kept per ship tile")
    ap.add_argument("--max-distance", type=int, default=26,
                    help="only verify candidate pairs within this Hamming distance")
    ap.add_argument("--no-orb", action="store_true", help="skip geometric verification (faster)")
    ap.add_argument("--knn", type=int, default=12, help="thumbnail neighbours examined per tile")
    ap.add_argument("--lsh-bucket", type=int, default=32,
                    help="max tiles per hash band bucket (0 = no banding; 200 blew up memory at 53k tiles)")
    ap.add_argument("--max-orb-pairs", type=int, default=20000, help="budget for geometric checks")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-extract", action="store_true", help="analyse only, write no images")
    args = ap.parse_args()

    t_start = time.perf_counter()
    reader = ZipReader(args.zip)
    args.results.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. select subset
    with zipfile.ZipFile(args.zip) as z, z.open(CSV_NAME) as f:
        df = pd.read_csv(f)
    per_image = df.groupby("ImageId")["EncodedPixels"].apply(lambda s: [r for r in s if isinstance(r, str)])
    ship_ids = [i for i, r in per_image.items() if r]
    empty_ids = [i for i, r in per_image.items() if not r]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ship_ids)
    rng.shuffle(empty_ids)
    if not args.all_ships:
        ship_ids = ship_ids[:args.max_ship_images]
    empty_ids = empty_ids[:int(len(ship_ids) * args.empty_ratio)]
    ids = list(ship_ids) + list(empty_ids)
    rng.shuffle(ids)
    n_ships = [len(per_image[i]) for i in ids]
    print(f"[1/6] dataset: {len(per_image)} images in the zip, {len(ship_ids)} with ships selected "
          f"+ {len(empty_ids)} empty = {len(ids)} tiles, {sum(n_ships)} ships")

    # ---------------------------------------------------------------- 2. signatures
    t0 = time.perf_counter()
    sigs: list[list[np.uint64]] = [None] * len(ids)          # type: ignore[list-item]
    vecs = np.zeros((len(ids), 64), np.float32)
    thumbs: list[np.ndarray] = [None] * len(ids)             # type: ignore[list-item]
    done = 0
    lock = threading.Lock()

    def work(k: int) -> None:
        nonlocal done
        g = reader.gray(ids[k])
        sigs[k] = signatures(g) if g is not None else []
        if g is not None:
            vecs[k] = thumb_vector(g)
            thumbs[k] = thumb_gray(g)
        else:
            thumbs[k] = np.zeros((128, 128), np.uint8)
        with lock:
            done += 1
            if done % 2000 == 0:
                print(f"      hashed {done}/{len(ids)}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, range(len(ids))))
    failed = sum(1 for s in sigs if not s)
    print(f"[2/6] perceptual hashes: {len(ids)} tiles x {len(WINDOWS)} windows "
          f"in {time.perf_counter() - t0:.0f}s" + (f" ({failed} unreadable)" if failed else ""))

    # ---------------------------------------------------------------- 3. duplicate groups
    t0 = time.perf_counter()
    # thumbnails to disk (memory-mapped by the verification processes)
    cache = args.out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "thumbs128.npy", np.stack(thumbs).astype(np.uint8))
    verifier = None if args.no_orb else OrbVerifier()
    group_of, evidence = build_groups(ids, sigs, vecs, thumbs, max_distance=args.max_distance,
                                      verifier=verifier,
                                      gray_of=(None if args.no_orb else (lambda k: reader.gray(ids[k]))),
                                      knn=args.knn, max_orb_pairs=args.max_orb_pairs,
                                      use_lsh=args.lsh_bucket > 1, lsh_bucket=args.lsh_bucket,
                                      workers=args.workers,
                                      thumb_path=str(cache / "thumbs128.npy"),
                                      log=lambda m: print(m, flush=True))
    n_groups = len(set(group_of))
    sizes = pd.Series(group_of).value_counts()
    multi = sizes[sizes > 1]
    print(f"[3/6] duplicates: {len(evidence)} pairs -> {n_groups} groups for {len(ids)} tiles "
          f"({len(multi)} groups with >1 tile, largest {int(sizes.max())}) in {time.perf_counter() - t0:.0f}s")

    # ---------------------------------------------------------------- 4. split by group
    split = group_aware_split(group_of, n_ships, seed=args.seed)
    bad = check_no_leakage(group_of, split)
    if bad:
        sys.exit(f"FATAL: {len(bad)} groups span several splits")
    table = pd.DataFrame({"ImageId": ids, "group": group_of, "split": split, "n_ships": n_ships})
    naive = pd.Series(rng.choice(["train", "val", "test"], len(ids), p=[0.8, 0.1, 0.1]))
    leaked_naive = sum(1 for _, g in table.assign(naive=naive).groupby("group")
                       if g.naive.nunique() > 1)
    print(f"[4/6] split (no leakage: {len(bad)} groups span splits) -- a naive random split would have "
          f"leaked {leaked_naive} groups")
    for s in ("train", "val", "test"):
        part = table[table.split == s]
        print(f"      {s:5s}: {len(part):6d} tiles, {int(part.n_ships.sum()):6d} ships, "
              f"{int((part.n_ships > 0).sum()):5d} tiles with ships")

    # ---------------------------------------------------------------- 5. write dataset
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "split.csv", index=False)
    pd.DataFrame(evidence).to_csv(args.out / "duplicates.csv", index=False)
    if not args.skip_extract:
        t0 = time.perf_counter()
        h, w = AIRBUS_SHAPE
        for s in ("train", "val", "test"):
            (args.out / "images" / s).mkdir(parents=True, exist_ok=True)
            (args.out / "labels" / s).mkdir(parents=True, exist_ok=True)

        def write(k: int) -> None:
            image_id, s = ids[k], split[k]
            (args.out / "images" / s / image_id).write_bytes(reader.raw(image_id))
            boxes = rles_to_boxes(per_image[image_id])
            (args.out / "labels" / s / f"{Path(image_id).stem}.txt").write_text(
                "".join("0 %.6f %.6f %.6f %.6f\n" % b.to_yolo(w, h) for b in boxes))

        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(write, range(len(ids))))
        (args.out / "data.yaml").write_text(
            f"path: {args.out.resolve().as_posix()}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: ship\n")
        mb = sum(f.stat().st_size for f in (args.out / "images").rglob("*.jpg")) / 1e6
        print(f"[5/6] extracted {len(ids)} images + labels ({mb:.0f} MB) in {time.perf_counter() - t0:.0f}s")
    else:
        print("[5/6] extraction skipped (--skip-extract)")

    # ---------------------------------------------------------------- 6. report
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    counts = sizes.value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values, color="#1B6CA8")
    axes[0].set_xlabel("tiles per group")
    axes[0].set_ylabel("number of groups")
    axes[0].set_yscale("log")
    axes[0].set_title("Duplicate groups", fontsize=10)
    comp = table.groupby("split").agg(tiles=("ImageId", "size"), ships=("n_ships", "sum")).reindex(
        ["train", "val", "test"])
    x = np.arange(3)
    axes[1].bar(x - 0.2, comp.tiles, 0.4, label="tiles", color="#1B6CA8")
    axes[1].bar(x + 0.2, comp.ships, 0.4, label="ships", color="#E07A1F")
    axes[1].set_xticks(x, comp.index)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Split composition (whole groups)", fontsize=10)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.results / "wp0_groups.png", dpi=160)
    if evidence:
        montage(reader, sorted(evidence, key=lambda e: -e["corr"]), args.results / "wp0_duplicates.jpg")

    stats = {
        "zip_images": int(len(per_image)),
        "selected_tiles": len(ids),
        "ship_tiles": int(len(ship_ids)),
        "empty_tiles": int(len(empty_ids)),
        "ships": int(sum(n_ships)),
        "duplicate_pairs": len(evidence),
        "pairs_by_method": {str(k): int(v) for k, v in
                            pd.Series([e["method"] for e in evidence]).value_counts().items()} if evidence else {},
        "groups": int(n_groups),
        "groups_with_duplicates": int(len(multi)),
        "largest_group": int(sizes.max()),
        "tiles_in_duplicate_groups": int(multi.sum()) if len(multi) else 0,
        "groups_spanning_splits": len(bad),
        "groups_a_naive_split_would_leak": int(leaked_naive),
        "split": {s: {"tiles": int(comp.loc[s, "tiles"]), "ships": int(comp.loc[s, "ships"])}
                  for s in comp.index},
        "runtime_s": round(time.perf_counter() - t_start, 1),
        "settings": {"max_distance": args.max_distance, "orb": not args.no_orb,
                     "empty_ratio": args.empty_ratio, "seed": args.seed},
    }
    (args.results / "wp0_stats.json").write_text(json.dumps(stats, indent=2))
    pct = 100 * stats["tiles_in_duplicate_groups"] / max(1, len(ids))
    (args.results / "wp0_report.md").write_text(f"""# WP0 -- Data integrity report

Source: `{args.zip.name}` ({stats['zip_images']} training tiles in the zip).

## Selection
{stats['selected_tiles']} tiles: {stats['ship_tiles']} with ships + {stats['empty_tiles']} empty
(ratio {args.empty_ratio}), {stats['ships']} ship instances.

## Near-duplicate detection
Perceptual hash (pHash, 64 bit) over {len(WINDOWS)} windows per tile, LSH banding for candidates,
Hamming distance <= {args.max_distance} accepted directly{'' if args.no_orb else ', ambiguous pairs confirmed by ORB + RANSAC'}.

* duplicate pairs found: **{stats['duplicate_pairs']}** {stats['pairs_by_method']}
* groups: **{stats['groups']}** ({stats['groups_with_duplicates']} contain duplicates, largest {stats['largest_group']} tiles)
* tiles inside a duplicate group: **{stats['tiles_in_duplicate_groups']} ({pct:.1f}%)**

## Leakage-free split
Whole groups are assigned to one split, balancing tiles and ships.

| split | tiles | ships |
|---|---|---|
""" + "".join(f"| {s} | {stats['split'][s]['tiles']} | {stats['split'][s]['ships']} |\n" for s in comp.index) + f"""
* groups spanning several splits: **{stats['groups_spanning_splits']}** (must be 0)
* a naive random split would have leaked **{stats['groups_a_naive_split_would_leak']}** groups

Runtime {stats['runtime_s']} s. Settings: {stats['settings']}.
""")
    print(f"[6/6] report: {args.results / 'wp0_report.md'} | stats: wp0_stats.json | "
          f"charts: wp0_groups.png, wp0_duplicates.jpg")
    print(f"Total {stats['runtime_s']}s. Train with: yolo detect train "
          f"data={(args.out / 'data.yaml').as_posix()} model=yolov8n.pt imgsz=768")


if __name__ == "__main__":
    try:
        main()
    except Exception:                      # make any crash visible in the log
        import traceback
        traceback.print_exc()
        sys.exit(1)
