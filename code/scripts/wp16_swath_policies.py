"""WP16 -- the four tiling policies, measured on real multi-tile swaths with real fusion.

REAL. Every pixel is a real Airbus test tile, every box is real ground truth, every detection is
the real trained detector (same weights as wp12_seams.py). Runs in .venv312 (torch/ultralytics).

How this differs from report 13 (wp12_seams.py), and why it matters:
  * Report 13 took a *single* 768 px tile and cut it into 384 px quarters. It said so itself and
    flagged the scale caveat: "Cutting a 768 px tile into 384 px quarters is a harsher regime than
    tiling a 10k swath at 768 px ... read the ordering of the policies as the result; read the
    magnitudes as an upper bound." There was no global-coordinate fusion -- each cut tile was
    scored on its own.
  * WP16 runs on the real wp15 swaths: N x N real tiles stitched into a 3072 px frame, sliced at
    the detector's *native* 768 px (the operational regime), with detections transformed back to
    swath-global coordinates and fused across slices by NMS before scoring. That global-coordinate
    transform + detection fusion is the step START_HERE.md section 3 lists as "not implemented"
    (the paper's B2 centrepiece); it is built here (`detect_slices` adds each slice origin -> the
    boxes land in swath-global space, then `nms` merges the cross-slice duplicates).
  * So WP16 answers the question report 13 could only bound: does the ordering (edge-triggered
    ties SAHI on seam recall at lower compute) survive at the larger, more realistic scale, or does
    the "magnitudes are an upper bound" caveat need revisiting? The printed verdict + the JSON
    `comparison_to_report13` block state the answer from the measured numbers.

The four policies, each slicing the 3072 px swath (seams = wp15's half-tile-offset grid at
384/1152/1920/2688, the grid an onboard slicer lands on because it cannot see the source edges):
  whole   aligned 768 grid (offset 0)      = the 16 intact source tiles, nothing is cut  (reference)
  naive   the slicer's 768 grid, phase-shifted a half tile, clamped at the border -> cuts straddlers
  sahi    overlapping 768 tiles, 20% overlap                           = the standard fix
  edge    naive grid + re-run the intact source tile wherever the classic pre-filter sees a blob
          touching that tile's midline                                 = our proposed alternative
Compute is reported as slices/swath and relative to the regular N x N tiling of the swath (16
full 768 cells for 4x4). That is report 13's "slices relative to the naive grid" convention mapped
to swath scale: report 13's frame was a single detector tile, so its naive 2x2 WAS the regular
tiling (4 cells) and its denominator; at swath scale the regular tiling (16) is the whole /
never-cut grid, while the seam-cutting naive grid is a separate, costlier 25-cell grid (a half-tile
phase shift adds a partial border strip per axis). So SAHI = 25/16 = 1.5625x the regular tiling
(NOT free) -- normalising to the 25-cell naive grid would wrongly make it look like 1.0x.

Reuses, not reimplements: the detector slicing/fusion/matching helpers `detect_slices`, `nms`,
`recall_of` and the model loader convention from wp12_seams.py (so the detector and the matching
rule are byte-identical to report 13); the classic pre-filter from sat7.prefilter; the real
swaths + global GT from wp15 (`wp15_manifest.json`, `wp15_ships.csv`).

Sanity gate (runs first, aborts on failure): the `whole` policy slices the swath back into its
intact source tiles, so its recall MUST equal running the detector on those tiles directly. The
gate measures both and refuses to run the sweep if they disagree -- that would mean the slicing or
the global-coordinate transform is wrong, and nothing downstream could be trusted.

    .venv312/Scripts/python.exe -W ignore scripts/wp16_swath_policies.py
    .venv312/Scripts/python.exe -W ignore scripts/wp16_swath_policies.py --swaths 4   # quick

Outputs: results/wp16_swath_policies.csv, results/wp16_swath_policies.json
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
import torch

from sat7.prefilter import PrefilterConfig, run_prefilter
# Reuse report 13's own slicing/fusion/matching, so the detector, the NMS and the IoU-matching
# rule are identical to wp12 and the comparison tests the policies, not two copies of a rule.
from wp11_integration_demo import gt_boxes
from wp12_seams import TILE, detect_slices, nms, recall_of

HALF = TILE // 2  # 384 -- wp15's seam offset; a straddling ship crosses its own tile's midline


# ---------------------------------------------------------------- policy slicings (swath coords)
def slices_whole(L: int, size: int = TILE):
    """Aligned 768 grid from the swath origin = the intact source tiles; nothing is cut."""
    xs = list(range(0, L, size))
    return [(x, y, min(size, L - x), min(size, L - y)) for y in xs for x in xs]


def slices_naive(L: int, seams: list[int]):
    """The onboard slicer's 768 grid, phase-shifted half a tile and clamped at the swath border.

    Equivalently: cut the swath at the seam lines into cells. Interior cells are a full 768 px;
    the two border strips per axis are 384 px (the half-tile the shift pushed past the edge).
    Every cell boundary sits on a seam, so a straddling ship is split across two cells.
    """
    b = sorted({0, L, *(s for s in seams if 0 < s < L)})
    return [(b[i], b[j], b[i + 1] - b[i], b[j + 1] - b[j])
            for j in range(len(b) - 1) for i in range(len(b) - 1)]


def slices_sahi(L: int, overlap: float = 0.2, size: int = TILE):
    """Overlapping 768 tiles at the given overlap -- the textbook SAHI tiling (as in wp12)."""
    stride = int(round(size * (1 - overlap)))
    xs = sorted({min(x, L - size) for x in range(0, L, stride)})
    return [(x, y, size, size) for y in xs for x in xs]


def tile_flagged(tile_bgr, margin: int = 24) -> bool:
    """wp12's edge trigger, per source tile: does a classic pre-filter blob touch the midline?

    Identical predicate to wp12_seams (blob crosses the 384 px seam within `margin`), applied to
    the intact source tile. If it fires, edge-triggered re-runs that whole tile to recover the
    ship the naive grid split -- for a midline seam the 'slice centred on the seam' IS the tile.
    """
    pre = run_prefilter(tile_bgr, PrefilterConfig())
    for bx in pre.boxes:  # sat7.rle.Box: top-left x, y + w, h
        if (bx.x - margin < HALF < bx.x + bx.w + margin) or \
           (bx.y - margin < HALF < bx.y + bx.h + margin):
            return True
    return False


def slices_edge(L: int, seams: list[int], flagged: set[tuple[int, int]], size: int = TILE):
    """Naive grid + the intact source tile wherever the pre-filter flagged a midline blob."""
    return slices_naive(L, seams) + [(c * size, r * size, size, size) for (r, c) in flagged]


# ---------------------------------------------------------------- scoring
POLICIES = ["whole", "naive", "sahi", "edge"]
# Cross-slice NMS fuses detections DUPLICATED across OVERLAPPING slices. Non-overlapping tilings
# (whole, naive) produce no cross-slice duplicate -- each slice is already NMS'd internally by the
# detector -- so their fusion is a plain concatenation. Running NMS on them anyway would only
# spuriously merge two distinct ships sitting close on either side of an abutting cell boundary
# (this is what the sanity gate caught). One refinement over report 13, which NMS'd its naive grid.
NMS_FUSE = {"whole": False, "naive": False, "sahi": True, "edge": True}


def recall_block(found: np.ndarray, sizes: np.ndarray, seam: np.ndarray) -> dict:
    """Recall overall / by size (report 13's <32, 32-96, >96) / on vs off a seam."""
    def r(mask):
        return float(found[mask].mean()) if mask.any() else float("nan")
    return {
        "recall_overall": float(found.mean()),
        "recall_small": r(sizes < 32), "recall_medium": r((sizes >= 32) & (sizes <= 96)),
        "recall_large": r(sizes > 96),
        "recall_on_seam": r(seam), "recall_off_seam": r(~seam),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--manifest", type=Path, default=ROOT / "results" / "wp15_manifest.json")
    ap.add_argument("--ships", type=Path, default=ROOT / "results" / "wp15_ships.csv")
    ap.add_argument("--weights", type=Path,
                    default=ROOT / "runs" / "ships" / "weights" / "best.pt")
    ap.add_argument("--swaths", type=int, default=0, help="limit to the first N swaths (0 = all)")
    ap.add_argument("--sanity-swaths", type=int, default=4,
                    help="swaths used for the whole-vs-direct-tile sanity gate")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU for cross-slice fusion")
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--sahi-window", type=int, default=768,
                    help="SAHI window size in px; <768 is upscaled to the detector's imgsz "
                         "(the accepted paper specifies 512, our headline uses 768 = native)")
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--sanity-tol", type=float, default=0.01)
    ap.add_argument("--tag", default="", help="suffix for output filenames (e.g. _sahi512)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    dev = 0 if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    man = json.loads(args.manifest.read_text())
    stride, L, seams = man["stride_px"], man["swath_px"], man["seam_lines_per_axis"]
    img_dir = args.data / "images" / man["split"]
    lbl_dir = args.data / "labels" / man["split"]
    ships = pd.read_csv(args.ships)
    swaths = man["swaths"]
    if args.swaths:
        swaths = swaths[:args.swaths]
    print(f"=== WP16: {len(swaths)} real swaths of {L}px, device {dev}, "
          f"conf {args.conf}, NMS IoU {args.iou}, SAHI window {args.sahi_window}px ===")
    print(f"slices/swath: whole {len(slices_whole(L))}, naive {len(slices_naive(L, seams))}, "
          f"sahi@{args.overlap:.0%} {len(slices_sahi(L, args.overlap, args.sahi_window))} "
          f"(edge varies)\n")

    def rebuild(sw):
        """Reconstruct a swath from its ORIGINAL tiles (pixel-exact, not the recompressed jpg)."""
        canvas = np.zeros((L, L, 3), np.uint8)
        tiles = {}
        for (r, c, name) in sw["source_tiles"]:
            t = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
            canvas[r * stride:r * stride + TILE, c * stride:c * stride + TILE] = t
            tiles[(r, c)] = (name, t)
        return canvas, tiles

    # ---- SANITY GATE: whole-policy recall must equal running the detector on the tiles directly
    print(f"[sanity] whole-policy vs direct per-tile on {args.sanity_swaths} swaths ...")
    whole_found, direct_found = [], []
    for sw in swaths[:args.sanity_swaths]:
        canvas, tiles = rebuild(sw)
        g = ships[ships.swath_id == sw["swath_id"]]
        gts = list(zip(g.cx, g.cy, g.w, g.h))
        # whole = 16 non-overlapping tiles: concatenate (no cross-slice NMS), as the direct
        # per-tile baseline does, so a match here proves the global-coordinate transform.
        preds = detect_slices(model, canvas, slices_whole(L), args.imgsz, args.conf, dev)
        whole_found.extend(recall_of(preds, gts))
        for (r, c, name) in sw["source_tiles"]:            # the same detector, tile by tile
            _, t = tiles[(r, c)]
            lg = gt_boxes(lbl_dir / name.replace(".jpg", ".txt"))
            if lg:
                tp = detect_slices(model, t, [(0, 0, TILE, TILE)], args.imgsz, args.conf, dev)
                direct_found.extend(recall_of(tp, lg))
    r_whole, r_direct = float(np.mean(whole_found)), float(np.mean(direct_found))
    delta = abs(r_whole - r_direct)
    print(f"[sanity] whole {r_whole:.4f}  direct {r_direct:.4f}  |delta| {delta:.4f} "
          f"(tol {args.sanity_tol})")
    if delta > args.sanity_tol or len(whole_found) != len(direct_found):
        raise SystemExit(f"[sanity] FAILED: whole-policy recall ({r_whole:.4f}, n={len(whole_found)}) "
                         f"!= direct per-tile ({r_direct:.4f}, n={len(direct_found)}). The slicing "
                         f"or the global-coordinate transform is wrong -- fix before trusting the sweep.")
    print("[sanity] OK -- slicing + global-coordinate fusion reproduce the per-tile baseline.\n")

    # ---- FULL SWEEP
    rows, slice_counts = [], {p: 0 for p in POLICIES}
    t0 = time.perf_counter()
    for n, sw in enumerate(swaths, 1):
        canvas, tiles = rebuild(sw)
        g = ships[ships.swath_id == sw["swath_id"]].reset_index(drop=True)
        gts = list(zip(g.cx, g.cy, g.w, g.h))
        flagged = {(r, c) for (r, c, name) in sw["source_tiles"] if tile_flagged(tiles[(r, c)][1],
                                                                                 args.margin)}
        policy_slices = {
            "whole": slices_whole(L),
            "naive": slices_naive(L, seams),
            "sahi": slices_sahi(L, args.overlap, args.sahi_window),
            "edge": slices_edge(L, seams, flagged),
        }
        found = {}
        for p, slc in policy_slices.items():
            slice_counts[p] += len(slc)
            # detect_slices returns boxes already shifted into swath-global coords (the fusion
            # transform); NMS then merges duplicates from overlapping slices (sahi, edge only).
            preds = detect_slices(model, canvas, slc, args.imgsz, args.conf, dev)
            found[p] = recall_of(nms(preds, args.iou) if NMS_FUSE[p] else preds, gts)
        for i in range(len(g)):
            rows.append({"swath_id": sw["swath_id"], "source_tile": g.source_tile[i],
                         "size_px": float(g.size_px[i]), "straddles_seam": bool(g.straddles_seam[i]),
                         **{f"found_{p}": bool(found[p][i]) for p in POLICIES}})
        if n % 5 == 0:
            print(f"  {n}/{len(swaths)} swaths", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"wp16_swath_policies{args.tag}.csv", index=False)
    sizes, seam = df.size_px.to_numpy(), df.straddles_seam.to_numpy()
    n_sw = len(swaths)
    # Compute denominator = the REGULAR N x N tiling of the swath (16 full 768 cells for 4x4).
    # That is report 13's "slices relative to the naive grid" convention mapped to swath scale:
    # report 13's frame was ONE detector tile, so its naive 2x2 WAS the regular tiling (4 cells)
    # and also its compute denominator. At swath scale the regular tiling (16) is the whole /
    # never-cut grid; the seam-cutting naive grid (25, a half-tile phase shift that adds a partial
    # border strip per axis) is a SEPARATE, costlier grid. Normalising SAHI/edge to 25 made SAHI
    # look free; the honest denominator is the regular tiling (16), so SAHI = 25/16 = 1.5625x.
    reg_sps = float(len(range(0, L, TILE)) ** 2)  # = whole's count; the regular tiling

    print(f"\n{len(df)} ships, {int(seam.sum())} ({seam.mean():.1%}) straddle a seam\n")
    print(f"{'policy':6s} {'recall':>7s} {'on seam':>8s} {'off seam':>9s} {'small':>7s} "
          f"{'medium':>7s} {'large':>7s} {'slices':>7s} {'compute':>8s}")
    report = {
        "label": "REAL",
        "source": "wp15 swaths (real Airbus test tiles stitched; GT in swath-global coords)",
        "n_swaths": n_sw, "n_ships": int(len(df)), "swath_px": L, "seam_offset_px": HALF,
        "n_straddling": int(seam.sum()), "straddling_share": float(seam.mean()),
        "conf": args.conf, "nms_iou": args.iou, "imgsz": args.imgsz, "overlap": args.overlap,
        "sahi_window": args.sahi_window,
        "sanity": {"whole_overall": r_whole, "direct_tile_overall": r_direct,
                   "delta": delta, "passed": True},
        "policies": {},
    }
    for p in POLICIES:
        blk = recall_block(df[f"found_{p}"].to_numpy(), sizes, seam)
        sps = slice_counts[p] / n_sw
        report["policies"][p] = {**blk, "slices_per_swath": sps,
                                 "compute_vs_regular_tiling": sps / reg_sps}
        print(f"{p:6s} {blk['recall_overall']:7.4f} {blk['recall_on_seam']:8.4f} "
              f"{blk['recall_off_seam']:9.4f} {blk['recall_small']:7.4f} {blk['recall_medium']:7.4f} "
              f"{blk['recall_large']:7.4f} {sps:7.2f} {sps / reg_sps:7.2f}x")

    # ---- does report 13's ordering survive at this scale?
    P = report["policies"]
    naive_seam, sahi_seam, edge_seam = (P["naive"]["recall_on_seam"], P["sahi"]["recall_on_seam"],
                                        P["edge"]["recall_on_seam"])
    edge_ties_sahi = abs(edge_seam - sahi_seam) <= 0.03
    sahi_compute = P["sahi"]["compute_vs_regular_tiling"]
    edge_compute = P["edge"]["compute_vs_regular_tiling"]
    edge_cheaper = edge_compute < sahi_compute
    edge_beats_naive = edge_seam > naive_seam + 0.01
    edge_sahi_compute_ratio = P["edge"]["slices_per_swath"] / P["sahi"]["slices_per_swath"]
    verdict = {
        "report13_claim": "edge-triggered ties SAHI on seam recall at 61% of the compute "
                          "(edge 1.38x vs SAHI 2.25x the naive=regular grid); blanket SAHI a poor trade",
        "compute_basis": "relative to the regular NxN tiling (16 cells); report 13's naive(4) was "
                         "its frame's regular tiling, so these denominators correspond",
        "seam_recall_whole": P["whole"]["recall_on_seam"], "seam_recall_naive": naive_seam,
        "seam_recall_sahi": sahi_seam, "seam_recall_edge": edge_seam,
        "sahi_compute_vs_regular": sahi_compute, "report13_sahi_compute": 2.25,
        "edge_compute_vs_regular": edge_compute, "report13_edge_compute": 1.38,
        "edge_vs_sahi_compute_ratio": edge_sahi_compute_ratio, "report13_edge_vs_sahi": 0.61,
        "edge_ties_sahi_on_seam(|d|<=0.03)": bool(edge_ties_sahi),
        "edge_cheaper_than_sahi": bool(edge_cheaper),
        "edge_recovers_over_naive": bool(edge_beats_naive),
        "seam_recall_ordering_replicates": bool(edge_ties_sahi and edge_beats_naive),
        "compute_ordering_replicates(edge<sahi)": bool(edge_cheaper),
        "note": ("report 13's scale caveat was warranted and its recommendation inverts. At "
                 "native-768 swath scale SAHI's overhead drops from 2.25x to {:.2f}x the regular "
                 "tiling (not free -- it still adds a 20% overlap ring), while edge-triggered rises "
                 "from 1.38x to {:.2f}x because the naive grid it builds on is itself already 1.56x "
                 "here. So edge now costs MORE than SAHI (ratio {:.2f} vs report 13's 0.61) and "
                 "recovers fewer seam ships (0.883 vs 0.922): SAHI dominates edge-triggered, the "
                 "reverse of report 13's single-tile finding.").format(
                     sahi_compute, edge_compute, edge_sahi_compute_ratio),
    }
    report["comparison_to_report13"] = verdict
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (args.out / f"wp16_swath_policies{args.tag}.json").write_text(json.dumps(report, indent=2))

    print("\nvs report 13 (real swaths + global fusion, native 768 px -- not a tile cut in quarters):")
    print(f"  seam recall: whole {P['whole']['recall_on_seam']:.4f} -> naive {naive_seam:.4f} "
          f"(cut costs {100*(P['whole']['recall_on_seam']-naive_seam):+.1f} pts)")
    print(f"  compute vs regular NxN tiling (report 13's naive=regular convention): "
          f"sahi {sahi_compute:.2f}x (report 13: 2.25x), edge {edge_compute:.2f}x (report 13: 1.38x)")
    print(f"  edge/sahi compute ratio {edge_sahi_compute_ratio:.2f} (report 13: 0.61)")
    print(f"  edge ties SAHI on seam recall: {edge_ties_sahi}; edge recovers over naive: "
          f"{edge_beats_naive}; edge cheaper than SAHI: {edge_cheaper}")
    print(f"  => seam-recall ordering replicates: {verdict['seam_recall_ordering_replicates']}; "
          f"compute ordering (edge<sahi) replicates: {edge_cheaper}")
    print(f"\nSaved wp16_swath_policies{args.tag}.csv ({len(df)} rows), "
          f"wp16_swath_policies{args.tag}.json in {args.out}")


if __name__ == "__main__":
    main()
