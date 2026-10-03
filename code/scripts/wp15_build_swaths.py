"""WP15 -- stitch real test tiles into synthetic wide swaths (data prep only).

REAL. Every pixel and every ground-truth box here comes from the real Airbus test split
(`data/yolo_ships/images/test` + `labels/test`); nothing is synthesised or drawn from an
assumed distribution. This script does NOT run the detector and imports no torch -- it only
builds the input for a later measurement.

Why this exists (report 16, the swath decision): report 13 measured tile seams by cutting a
single real 768 px tile into quarters -- a 768-px-tile-cut-into-quarters experiment. That is a
*harsher* regime than a real sensor (the slices are small relative to the ships) and report 13
says so explicitly. To measure edge-triggered re-inference on something closer to the real
thing without adopting a new large-scene dataset -- which would throw away the trained detector
and every existing measurement -- we stitch the real 768 px test tiles we already have into
wide swaths and re-slice *those*. This file is the data prerequisite for tomorrow's
edge-triggered re-inference run on actual wide swaths; it is NOT the report-13 experiment.

Geometry (and why the straddle rate is a sanity check, not a coincidence):
  * A swath is an N x N grid of real 768 px tiles (default 4x4 = 3072 px). With the default
    overlap 0 the tiles abut and the swath side is exactly N*768 -- report 13's non-overlapping
    ("naive grid") convention.
  * Every tile's ground-truth boxes are shifted from tile-local to swath-global coordinates.
    Each tile contributes its own ships once, so no ship is dropped or duplicated by stitching.
  * The seams are the grid an onboard slicer would re-cut the swath on at the detector's native
    768 px. A real slicer cannot see where the original tile edges were, so its grid sits at an
    arbitrary phase. We put it at a half-tile (384 px) offset from the stitch grid, which
    bisects every source tile -- the exact midline cut report 13 measured on 2x2. At N=1 this
    reduces to wp12_seams.straddles identically; at 4x4 the straddling share should land close
    to report 13's 15.0% (it will not match exactly -- different sample, swath-edge effects).

Reuses, rather than reimplements, the pieces report 13 depends on:
  * the test-split loader `_gt_boxes` from wp6_build_catalogue (torch-free, same loader WP6
    uses) -- so the tiles and boxes are read exactly as every other WP reads them;
  * the seam-crossing predicate from wp12_seams.straddles, extended from one midline to the
    swath's grid of seam lines (`_crosses` below) -- kept identical so report 13's seam
    definition is not re-derived and cannot drift from its numbers.

    .venv/Scripts/python.exe scripts/wp15_build_swaths.py                 # 24 swaths, 4x4
    .venv/Scripts/python.exe scripts/wp15_build_swaths.py --swaths 8 --tiles-per-side 3
    .venv/Scripts/python.exe scripts/wp15_build_swaths.py --regenerate results/wp15_manifest.json

Outputs (results/):
  wp15_swaths/<swath_id>.jpg       the stitched swath images (skip with --no-images)
  wp15_examples/<swath_id>.png     2-3 annotated swaths (boxes + seams) to eyeball the transform
  wp15_ships.csv                   one row per ship: swath, source tile, global box, straddles_seam
  wp15_manifest.json               config + per-swath recipe (source tiles, offsets, seam grid)
  wp15_summary.json                swath/ship counts, % straddling, sanity note vs report 13
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

import cv2
import numpy as np
import pandas as pd

# The real test-split loader, reused verbatim so swath boxes are read the same way every WP
# reads them. wp6_build_catalogue imports only cv2/numpy/pandas/sat7.prefilter -- no torch --
# so this script stays torch-free and runs in .venv, not .venv312.
from wp6_build_catalogue import _gt_boxes

TILE = 768


def _crosses(lo: float, hi: float, lines) -> bool:
    """True if the interval (lo, hi) strictly contains any seam line.

    This is wp12_seams.straddles, extended from the single 2x2 midline to the swath's grid of
    seam lines and expressed on box edges (x0, x1) instead of (cx, w). wp12 tested
    `cx - w/2 < half < cx + w/2`; with lo = cx - w/2 and hi = cx + w/2 that is `lo < line < hi`
    for the one line `half`. Kept strict (`<`), as in wp12, so a box merely tangent to a seam
    does not count as straddling. The predicate is unchanged; only the number of lines grows.
    """
    return any(lo < ln < hi for ln in lines)


def seam_lines(swath_px: int, offset: int, stride: int = TILE) -> list[int]:
    """Seam positions along one axis: the detector's re-slice grid at a half-tile phase offset.

    offset = TILE // 2 bisects every source tile, reproducing report 13's midline cut.
    """
    return [x for x in range(offset, swath_px, stride) if 0 < x < swath_px]


def stitch(spec: list[tuple[int, int, str]], img_dir: Path, lbl_dir: Path,
           stride_px: int, swath_px: int):
    """Build one swath image + its ship rows from a list of (row, col, tile_name).

    Returns (swath_bgr, ship_rows, n_boxes_loaded). Later tiles overwrite earlier ones in any
    overlap strip; with the default overlap 0 there is no strip and the tiles simply abut.
    """
    canvas = np.zeros((swath_px, swath_px, 3), dtype=np.uint8)
    rows, n_loaded = [], 0
    xs = seam_lines(swath_px, TILE // 2)
    ys = xs  # square swath, same grid on both axes
    for (r, c, name) in spec:
        img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (TILE, TILE):
            raise ValueError(f"{name}: expected a {TILE}x{TILE} tile, got "
                             f"{None if img is None else img.shape}")
        ox, oy = c * stride_px, r * stride_px
        canvas[oy:oy + TILE, ox:ox + TILE] = img

        boxes = _gt_boxes(lbl_dir / name.replace(".jpg", ".txt"))  # (n, 4) xyxy, tile-local px
        # A few labels overhang the tile edge by a sub-pixel. Clip to the visible tile, the same
        # clipped-GT convention report 13 used ("Matching is against the clipped ground-truth
        # box"); this keeps boxes inside their tile without dropping any ship.
        boxes = np.clip(boxes, 0, TILE)
        n_loaded += len(boxes)
        for (x0, y0, x1, y1) in boxes:
            gx0, gy0, gx1, gy1 = x0 + ox, y0 + oy, x1 + ox, y1 + oy
            w, h = gx1 - gx0, gy1 - gy0
            rows.append({
                "source_tile": name, "grid_row": r, "grid_col": c,
                "x0": gx0, "y0": gy0, "x1": gx1, "y1": gy1,
                "cx": gx0 + w / 2, "cy": gy0 + h / 2, "w": w, "h": h,
                "size_px": float(max(w, h)),
                "straddles_seam": bool(_crosses(gx0, gx1, xs) or _crosses(gy0, gy1, ys)),
            })
    return canvas, rows, n_loaded


def annotate(swath_bgr, rows, swath_px: int, stride_px: int, n_side: int, max_side: int = 1600):
    """Draw seams (yellow), stitch boundaries (blue), and ship boxes (red=straddles, green=not)."""
    out = swath_bgr.copy()
    for x in seam_lines(swath_px, TILE // 2):                       # seam grid
        cv2.line(out, (x, 0), (x, swath_px - 1), (0, 220, 220), 2)
        cv2.line(out, (0, x), (swath_px - 1, x), (0, 220, 220), 2)
    for k in range(1, n_side):                                     # stitch boundaries
        p = k * stride_px
        cv2.line(out, (p, 0), (p, swath_px - 1), (200, 120, 0), 1)
        cv2.line(out, (0, p), (swath_px - 1, p), (200, 120, 0), 1)
    for r in rows:
        col = (40, 40, 220) if r["straddles_seam"] else (60, 200, 60)
        cv2.rectangle(out, (int(r["x0"]), int(r["y0"])), (int(r["x1"]), int(r["y1"])), col, 2)
    cv2.putText(out, "yellow=seam  blue=tile edge  red=straddles  green=clear",
                (12, swath_px - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if swath_px > max_side:                                        # downscale for quick viewing
        s = max_side / swath_px
        out = cv2.resize(out, (max_side, max_side), interpolation=cv2.INTER_AREA)
    return out


def check_invariants(rows, n_loaded_total: int, swath_px: int) -> None:
    """Hard checks: no ship dropped/duplicated, and no box outside its swath bounds."""
    assert len(rows) == n_loaded_total, (
        f"ship count changed in stitching: loaded {n_loaded_total} tile-local boxes but wrote "
        f"{len(rows)} global rows -- a ship was dropped or duplicated")
    eps = 1e-6
    for r in rows:
        assert 0 - eps <= r["x0"] < r["x1"] <= swath_px + eps and \
               0 - eps <= r["y0"] < r["y1"] <= swath_px + eps, (
            f"box {(r['x0'], r['y0'], r['x1'], r['y1'])} from {r['source_tile']} lies outside "
            f"the 0..{swath_px} swath bounds")


def build_specs(img_dir: Path, lbl_dir: Path, n_side: int, n_swaths: int,
                tile_pool: str, seed: int) -> list[list[tuple[int, int, str]]]:
    """Sample source tiles and lay them out into n_swaths grids of n_side x n_side."""
    files = sorted(p.name for p in img_dir.glob("*.jpg"))
    if tile_pool == "ships":  # keep swaths dense + the straddle rate comparable to report 13
        files = [f for f in files if len(_gt_boxes(lbl_dir / f.replace(".jpg", ".txt")))]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    per = n_side * n_side
    need = n_swaths * per
    if need > len(order):
        raise SystemExit(f"need {need} tiles for {n_swaths} swaths of {n_side}x{n_side} but the "
                         f"'{tile_pool}' pool has only {len(files)}; lower --swaths/--tiles-per-side")
    specs = []
    for s in range(n_swaths):
        chunk = [files[i] for i in order[s * per:(s + 1) * per]]
        specs.append([(k // n_side, k % n_side, name) for k, name in enumerate(chunk)])
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--split", default="test")
    ap.add_argument("--swaths", type=int, default=24, help="number of synthetic swaths to build")
    ap.add_argument("--tiles-per-side", type=int, default=4, help="N; swath is N*N tiles")
    ap.add_argument("--overlap", type=float, default=0.0,
                    help="fraction of tile overlap between neighbours (0 = report 13's abutting grid)")
    ap.add_argument("--tile-pool", choices=("ships", "all"), default="ships",
                    help="'ships' = only tiles that contain ships (comparable to report 13); "
                         "'all' = realistic mix including empty ocean")
    ap.add_argument("--examples", type=int, default=3, help="annotated swaths to draw")
    ap.add_argument("--no-images", action="store_true", help="write manifest/CSV only, skip swath jpgs")
    ap.add_argument("--regenerate", type=Path, default=None,
                    help="rebuild images+CSV deterministically from a previously written manifest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    img_dir = args.data / "images" / args.split
    lbl_dir = args.data / "labels" / args.split
    t0 = time.perf_counter()

    if args.regenerate:
        man = json.loads(args.regenerate.read_text())
        n_side, overlap = man["tiles_per_side"], man["overlap"]
        specs = [[tuple(t) for t in sw["source_tiles"]] for sw in man["swaths"]]
        print(f"regenerating {len(specs)} swaths from {args.regenerate.name}")
    else:
        n_side, overlap = args.tiles_per_side, args.overlap
        specs = build_specs(img_dir, lbl_dir, n_side, args.swaths, args.tile_pool, args.seed)
        print(f"built {len(specs)} swath recipes of {n_side}x{n_side} from the '{args.tile_pool}' "
              f"pool of the {args.split} split")

    stride_px = int(round(TILE * (1 - overlap)))
    swath_px = (n_side - 1) * stride_px + TILE
    xs = seam_lines(swath_px, TILE // 2)
    print(f"swath {swath_px}px, tile stride {stride_px}px (overlap {overlap:.0%}), "
          f"{len(xs)} seam lines/axis at x={xs}\n")

    sw_dir = args.out / "wp15_swaths"
    ex_dir = args.out / "wp15_examples"
    if not args.no_images:
        sw_dir.mkdir(parents=True, exist_ok=True)
    ex_dir.mkdir(parents=True, exist_ok=True)

    all_rows, manifest_swaths, n_loaded_total = [], [], 0
    for s, spec in enumerate(specs):
        swath_id = f"swath_{s:03d}"
        canvas, rows, n_loaded = stitch(spec, img_dir, lbl_dir, stride_px, swath_px)
        n_loaded_total += n_loaded
        for r in rows:
            r["swath_id"] = swath_id
        all_rows.extend(rows)
        if not args.no_images:
            cv2.imwrite(str(sw_dir / f"{swath_id}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if s < args.examples:
            cv2.imwrite(str(ex_dir / f"{swath_id}.png"),
                        annotate(canvas, rows, swath_px, stride_px, n_side))
        manifest_swaths.append({
            "swath_id": swath_id, "n_ships": len(rows),
            "n_straddling": int(sum(r["straddles_seam"] for r in rows)),
            "source_tiles": [[r_, c_, name] for (r_, c_, name) in spec],
        })
        if (s + 1) % 10 == 0:
            print(f"  {s + 1}/{len(specs)} swaths", flush=True)

    # ---- the invariants the whole exercise rests on
    check_invariants(all_rows, n_loaded_total, swath_px)
    print(f"\n[ok] ship count preserved: {n_loaded_total} loaded == {len(all_rows)} written")
    print(f"[ok] every box inside 0..{swath_px}px\n")

    df = pd.DataFrame(all_rows)[["swath_id", "source_tile", "grid_row", "grid_col",
                                 "x0", "y0", "x1", "y1", "cx", "cy", "w", "h",
                                 "size_px", "straddles_seam"]]
    df.to_csv(args.out / "wp15_ships.csv", index=False)

    manifest = {
        "label": "REAL -- stitched from real Airbus test tiles; no detector, no torch",
        "purpose": "data prerequisite for edge-triggered re-inference on wide swaths (report 16); "
                   "NOT the 768px-tile-cut-into-quarters experiment of report 13",
        "split": args.split, "tiles_per_side": n_side, "overlap": overlap,
        "tile_px": TILE, "stride_px": stride_px, "swath_px": swath_px,
        "seam_offset_px": TILE // 2, "seam_stride_px": TILE, "seam_lines_per_axis": xs,
        "tile_pool": args.tile_pool, "seed": args.seed, "swaths": manifest_swaths,
    }
    (args.out / "wp15_manifest.json").write_text(json.dumps(manifest, indent=2))

    n_ships = len(df)
    n_strad = int(df.straddles_seam.sum())
    summary = {
        "label": "REAL",
        "n_swaths": len(specs), "swath_px": swath_px, "tiles_per_side": n_side,
        "overlap": overlap, "tile_pool": args.tile_pool,
        "n_source_tiles": len(specs) * n_side * n_side,
        "n_ships": n_ships,
        "n_straddling_seam": n_strad,
        "pct_straddling_seam": round(100 * n_strad / n_ships, 2) if n_ships else 0.0,
        "ships_per_swath_mean": round(n_ships / len(specs), 2),
        "report13_2x2_straddle_pct": 15.0,
        "sanity_note": ("straddle % should sit near report 13's 15.0% because the seam grid "
                        "bisects every source tile exactly as report 13's 2x2 midline cut did; "
                        "it will not match exactly (different sample, swath-edge effects, N!=2)"),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (args.out / "wp15_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    imgs = "skipped (--no-images)" if args.no_images else f"{len(specs)} in {sw_dir}"
    print(f"\nSaved wp15_ships.csv ({n_ships} rows), wp15_manifest.json, wp15_summary.json; "
          f"swath jpgs: {imgs}; {min(args.examples, len(specs))} annotated in {ex_dir}")


if __name__ == "__main__":
    main()
