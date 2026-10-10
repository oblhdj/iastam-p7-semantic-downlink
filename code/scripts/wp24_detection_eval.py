"""WP24 -- detection evaluation at full scale: precision / recall / F1 / AP for B1, B2 and the SAHI
ablations (audit G1, B8, B9).

REAL. Why this exists: until 10 Oct 2026 the B0->B4 table quoted B1 = 0.782 and B2 = 0.745 from
wp18's small samples -- 262 and 47 ships, recall only (now wp18_campaign.json legacy_small_samples;
the table's B1 / B2 come from the same-input run wp26). This file is the full-scale reference for
"the detector on native tiles", which is not the paper's B1. The two SAHI ablations the paper lists ("without tile overlap", "without detection
fusion") existed as PerceptionConfig flags but were never measured. This scores all of them on the
full data with the standard detection metrics (sat7.evaluation), through the library path the
pipeline actually uses (sat7.perception: load_detector -> detect_image -> b2_sahi_fusion):

  B1  whole tile, all 5,320 test tiles -- from the committed wp1_predictions.csv (the canonical
      detector output) and, with --live-b1, re-run live with the torch-free ONNX detector and
      cross-checked box by box against it
  swaths (wp15: 24 real 3072 px swaths stitched from 16 test tiles each, 716 ships):
      whole_grid       PerceptionConfig(mode="sahi", overlap=0.0, fuse=False) on the swath: the
                       aligned 768 grid = the 16 source tiles, i.e. B1 on these ships (wp16 "whole")
      sahi_B2          mode="sahi", repo defaults: window 768, overlap 0.20, fusion NMS IoU 0.50
                       (wp16 "sahi", the B2 configuration)
      sahi_no_fusion   the same windows, fuse=False: global coordinates, duplicates kept
      no_overlap       overlap=0.0 with the slicer's grid phase-shifted half a tile (the swath is
                       framed by 384 px of black nodata so the abutting windows straddle the stitch
                       seams, as an onboard slicer would -- otherwise overlap=0 reproduces the source
                       tiles exactly); fusion on (repo default). Comparable to wp16 "naive"

Gates (wp16 pattern, refuse to report on failure):
  (a) global coordinates: whole_grid boxes must EQUAL running the detector on each source tile and
      adding its origin (x_global = x0 + x_local, y_global = y0 + y_local) -- exact, every box
  (b) reproduction: WP1-rule recall at 0.25 for whole_grid / sahi_B2 within 0.005 of wp16's committed
      PyTorch-GPU values (ONNX FP32 on CPU drifts in the last digits)

Detections are kept down to conf 0.05 so AP has a curve; P / R / F1 are at the 0.25 onboard cut,
IoU 0.5. A swath ablation row adds seam / off-seam recall (wp15 flags the ships that straddle a seam).

    python scripts/wp24_detection_eval.py                # B1 committed + live swaths (CPU, ~1.5 min)
    python scripts/wp24_detection_eval.py --live-b1      # + live B1 on 5,320 tiles (CPU, ~3 min)

Needs the built split (data/yolo_ships) for labels and pixels, and detector weights; no torch with
the default ONNX model. Output: results/wp24_detection_eval.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from sat7.b2_sahi_fusion import _iou
from sat7.datasets import parse_yolo_lines, recall_of
from sat7.evaluation import evaluate_detections, match_image
from sat7.imagery import load_image
from sat7.perception import PerceptionConfig, detect_image, load_detector, slice_count

TILE, HALF = 768, 384
CONF_HIGH = 0.670                   # LoDConfig.conf_high (WP2): the P1 / P2 boundary


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def gt_of(lbl_dir: Path, name: str, w: int = TILE, h: int = TILE):
    p = lbl_dir / (Path(name).stem + ".txt")
    return parse_yolo_lines(p.read_text().splitlines(), w, h, classes={0}) if p.exists() else []


def wp1_recall(images, conf_thr: float) -> float:
    """WP1's own rule (GT-ordered greedy, IoU >= 0.5) at the cut -- the one B1/B2 were quoted with."""
    found = []
    for dets, gts in images:
        found += recall_of([d for d in dets if d[4] >= conf_thr], gts, iou_thr=0.5)
    return round(float(np.mean(found)), 4) if found else float("nan")


def cross_check(live, ref):
    """Pair live boxes with committed ones (greedy, IoU >= 0.5); count threshold flips."""
    used, pairs = set(), []
    for d in sorted(live, key=lambda z: -z[4]):
        best, bj = 0.0, -1
        for j, c in enumerate(ref):
            if j not in used and (v := _iou(d[:4], c[:4])) > best:
                best, bj = v, j
        if best >= 0.5:
            used.add(bj)
            pairs.append((d, ref[bj]))
    paired_live = {id(a) for a, _ in pairs}
    flips = {}
    for name, thr in (("det_thr_0.25", 0.25), ("conf_high_0.670", CONF_HIGH)):
        f = sum((a[4] >= thr) != (b[4] >= thr) for a, b in pairs)
        f += sum(d[4] >= thr for d in live if id(d) not in paired_live)
        f += sum(r[4] >= thr for j, r in enumerate(ref) if j not in used)
        flips[name] = int(f)
    return len(pairs), max((abs(a[4] - b[4]) for a, b in pairs), default=0.0), flips


def rebuild(sw, img_dir: Path, L: int, stride: int) -> np.ndarray:
    """A swath from its original tiles, pixel-exact (the wp16 / wp18 construction)."""
    canvas = np.zeros((L, L, 3), np.uint8)
    for (r, c, name) in sw["source_tiles"]:
        tile, _ = load_image(img_dir / name)
        canvas[r * stride:r * stride + TILE, c * stride:c * stride + TILE] = tile
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--model", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.onnx")
    ap.add_argument("--conf", type=float, default=0.25, help="onboard operating cut")
    ap.add_argument("--swaths", type=int, default=0, help="limit swaths (0 = all 24)")
    ap.add_argument("--live-b1", action="store_true", help="re-run B1 live on every test tile")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    res_dir = ROOT / "results"
    img_dir, lbl_dir = args.data / "images" / "test", args.data / "labels" / "test"
    if not lbl_dir.is_dir():
        sys.exit(f"labels not found at {lbl_dir}: build the split with scripts/wp0_build_dataset.py")
    t_all = time.perf_counter()
    det = load_detector(args.model)
    backend = f"onnxruntime {det.ort_version} CPU" if hasattr(det, "ort_version") else "ultralytics"
    report = {"label": "REAL -- trained detector on real Airbus test tiles / wp15 swaths, real labels",
              "model": str(args.model), "backend": backend, "conf_thr": args.conf, "iou_thr": 0.5,
              "raw_conf": 0.05, "metric_note": "P/R/F1 at conf_thr; AP over all boxes >= 0.05 "
              "(ultralytics 101-point AP; the 0.05 dump truncates the tail, so AP is a lower bound "
              "on what conf 0.001 would give)"}
    print(f"=== WP24 detection evaluation, {backend}, P/R/F1 @ conf {args.conf}, IoU 0.5 ===\n")

    # ------------------------------------------------------------------ B1, committed predictions
    names = [r["image"] for r in read_csv(res_dir / "wp6_tiles.csv")]          # the 5,320 test tiles
    preds = {n: [] for n in names}
    for r in read_csv(res_dir / "wp1_predictions.csv"):
        preds[r["image"]].append(tuple(float(r[k]) for k in ("cx", "cy", "w", "h", "conf")))
    gts = {n: gt_of(lbl_dir, n) for n in names}
    b1_images = [(preds[n], gts[n]) for n in names]
    b1 = evaluate_detections(b1_images, conf_thr=args.conf)
    b1["recall_WP1_rule"] = wp1_recall(b1_images, args.conf)
    report["B1_committed_all_test_tiles"] = b1
    print(f"[B1] whole tile, committed wp1_predictions.csv, {b1['images']} tiles / {b1['n_gt']} ships")
    print(f"     P {b1['precision']}  R {b1['recall']}  F1 {b1['F1']}  AP50 {b1['AP50']}  "
          f"AP50-95 {b1['AP50-95']}   (WP1-rule recall {b1['recall_WP1_rule']})")
    print(f"     recall by size: { {k: v['recall'] for k, v in b1['recall_by_size'].items()} }")

    # ------------------------------------------------------------------ B1, live re-run (optional)
    if args.live_b1:
        t0 = time.perf_counter()
        live_images, paired, n_live, n_ref, max_d = [], 0, 0, 0, 0.0
        flips = {"det_thr_0.25": 0, "conf_high_0.670": 0}
        for i, n in enumerate(names, 1):
            img, _ = load_image(img_dir / n)
            dets = detect_image(img, det, PerceptionConfig(mode="whole"))
            live_images.append((dets, gts[n]))
            p, d, f = cross_check(dets, preds[n])
            paired, n_live, n_ref, max_d = paired + p, n_live + len(dets), n_ref + len(preds[n]), max(max_d, d)
            for k in flips:
                flips[k] += f[k]
            if i % 1000 == 0:
                print(f"     live B1 {i}/{len(names)} tiles", flush=True)
        b1l = evaluate_detections(live_images, conf_thr=args.conf)
        b1l["recall_WP1_rule"] = wp1_recall(live_images, args.conf)
        b1l["vs_committed"] = {"live_boxes": n_live, "committed_boxes": n_ref, "paired": paired,
                               "max_abs_dconf": round(max_d, 6), "decision_flips": flips}
        b1l["seconds"] = round(time.perf_counter() - t0, 1)
        report["B1_live"] = b1l
        print(f"[B1 live] {b1l['seconds']} s: P {b1l['precision']}  R {b1l['recall']}  F1 {b1l['F1']}  "
              f"AP50 {b1l['AP50']}; vs committed: {paired}/{n_ref} boxes paired (live {n_live}), "
              f"max |dconf| {max_d:.1e}, flips {flips}")

    # ------------------------------------------------------------------ swaths: B2 + ablations
    man = json.loads((res_dir / "wp15_manifest.json").read_text())
    L, stride = man["swath_px"], man["stride_px"]
    ships = read_csv(res_dir / "wp15_ships.csv")
    swaths = man["swaths"][:args.swaths] if args.swaths else man["swaths"]
    rows = {
        "whole_grid": PerceptionConfig(mode="sahi", window=TILE, overlap=0.0, fuse=False),
        "sahi_B2": PerceptionConfig(mode="sahi"),                         # repo defaults
        "sahi_no_fusion": PerceptionConfig(mode="sahi", fuse=False),
        "no_overlap": PerceptionConfig(mode="sahi", window=TILE, overlap=0.0),   # on the shifted frame
    }
    per_row = {k: [] for k in rows}
    seam_flags, calls = [], {k: 0 for k in rows}
    gate_a_boxes, gate_a_equal = 0, True
    t0 = time.perf_counter()
    for n_sw, sw in enumerate(swaths, 1):
        canvas = rebuild(sw, img_dir, L, stride)
        g = [s for s in ships if s["swath_id"] == sw["swath_id"]]
        sgts = [(float(s["cx"]), float(s["cy"]), float(s["w"]), float(s["h"])) for s in g]
        seam_flags.append([s["straddles_seam"] == "True" for s in g])
        for name, cfg in rows.items():
            if name == "no_overlap":
                framed = cv2.copyMakeBorder(canvas, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT, value=0)
                d = detect_image(framed, det, cfg)
                d = [(x - HALF, y - HALF, w, h, c) for (x, y, w, h, c) in d]
                calls[name] += slice_count(framed.shape[1], framed.shape[0], cfg)
            else:
                d = detect_image(canvas, det, cfg)
                calls[name] += slice_count(L, L, cfg)
            per_row[name].append((d, sgts))
        # gate (a): the global transform -- detect each source tile alone, add its origin
        direct = []
        for (r, c, tname) in sw["source_tiles"]:
            tile = canvas[r * stride:r * stride + TILE, c * stride:c * stride + TILE]
            x0, y0 = c * stride, r * stride
            direct += [(x0 + x, y0 + y, w, h, cf) for (x, y, w, h, cf) in det([tile])[0]]
        grid = per_row["whole_grid"][-1][0]
        gate_a_boxes += len(direct)
        gate_a_equal &= sorted(direct) == sorted(grid)
        if n_sw % 6 == 0:
            print(f"     swaths {n_sw}/{len(swaths)}", flush=True)
    sw_secs = round(time.perf_counter() - t0, 1)
    if not gate_a_equal:
        raise SystemExit("[gate a] FAILED: whole-grid boxes != per-tile boxes + origin; the "
                         "global-coordinate transform is wrong -- fix before trusting any row")
    print(f"\n[gate a] global coordinates OK: {gate_a_boxes} whole-grid boxes == per-tile detection "
          f"+ (x0, y0), exactly")

    wp16 = json.loads((res_dir / "wp16_swath_policies.json").read_text())["policies"]
    swath_out = {}
    for name, images in per_row.items():
        m = evaluate_detections(images, conf_thr=args.conf)
        m["recall_WP1_rule"] = wp1_recall(images, args.conf)
        hits, seams = [], []
        for (dets, sgts), flags in zip(images, seam_flags):
            _, gc = match_image(dets, sgts, 0.5)
            hits += [c >= args.conf and c > 0 for c in gc]
            seams += flags
        hits, seams = np.asarray(hits, bool), np.asarray(seams, bool)
        m["recall_on_seam"] = round(float(hits[seams].mean()), 4)
        m["recall_off_seam"] = round(float(hits[~seams].mean()), 4)
        m["n_on_seam"] = int(seams.sum())
        m["detector_calls_per_swath"] = calls[name] / len(swaths)
        m["compute_vs_regular_tiling_x"] = round(calls[name] / len(swaths) / 16, 4)
        m["config"] = {"mode": rows[name].mode, "window": rows[name].window, "overlap": rows[name].overlap,
                       "fuse": rows[name].fuse, "nms_iou": rows[name].nms_iou,
                       "framing": "384 px black nodata border (slicer phase = half tile)"
                       if name == "no_overlap" else "none"}
        swath_out[name] = m
    gate_b = {}
    for name, key in (("whole_grid", "whole"), ("sahi_B2", "sahi"), ("no_overlap", "naive")):
        ref = round(wp16[key]["recall_overall"], 4)
        got = swath_out[name]["recall_WP1_rule"]
        gate_b[name] = {"wp16_policy": key, "wp16_recall": ref, "here": got, "abs_diff": round(abs(got - ref), 4)}
    # wp16's figures are over all its swaths; a subset (--swaths N) is a different ship sample, so
    # the reproduction gate only applies to the full set (a 2-swath run reproduces wp18's 47-ship B2)
    full_set = len(swaths) == int(json.loads((res_dir / "wp16_swath_policies.json").read_text())["n_swaths"])
    passed_b = (all(gate_b[k]["abs_diff"] <= 0.005 for k in ("whole_grid", "sahi_B2"))
                if full_set else None)
    report["swaths"] = {"n_swaths": len(swaths), "n_ships": swath_out["sahi_B2"]["n_gt"],
                        "seconds": sw_secs, "rows": swath_out,
                        "gate_a_global_coordinates": {"boxes": gate_a_boxes, "exact_match": True},
                        "gate_b_reproduces_wp16": {"rows": gate_b, "tolerance": 0.005, "passed": passed_b,
                                                   "note": "no_overlap is framed with black nodata, "
                                                   "wp16 naive with 384 px cells letterboxed grey: "
                                                   "comparable, not identical -- reported, not gated"}}
    if passed_b is False:
        raise SystemExit(f"[gate b] FAILED: {gate_b}")
    print(f"[gate b] {'reproduces' if passed_b else 'NOT APPLIED (swath subset) --'} wp16 "
          f"(WP1-rule recall @0.25): " + ", ".join(
        f"{k} {v['here']} vs {v['wp16_recall']}" for k, v in gate_b.items()))

    print(f"\n[swaths] {len(swaths)} swaths / {swath_out['sahi_B2']['n_gt']} ships, {sw_secs} s")
    print(f"     {'row':<16}{'calls':>6}{'P':>8}{'R':>8}{'F1':>8}{'AP50':>8}{'AP50-95':>9}"
          f"{'R seam':>8}{'R off':>8}{'FP':>6}")
    for name, m in swath_out.items():
        print(f"     {name:<16}{m['detector_calls_per_swath']:6.0f}{m['precision']:8.4f}{m['recall']:8.4f}"
              f"{m['F1']:8.4f}{m['AP50']:8.4f}{m['AP50-95']:9.4f}{m['recall_on_seam']:8.4f}"
              f"{m['recall_off_seam']:8.4f}{m['FP']:6d}")

    old = json.loads((res_dir / "wp18_campaign.json").read_text())["legacy_small_samples"]
    camp = {"B1": old["native_tiles"], "B2": old["swaths"]}
    report["superseded_small_sample"] = {
        "B1": {"recall": camp["B1"]["recall"]["overall"], "gt_ships": camp["B1"]["gt_ships"]},
        "B2": {"recall": camp["B2"]["recall"]["overall"], "gt_ships": camp["B2"]["gt_ships"]},
        "source": "wp18_campaign.json legacy_small_samples (a detector call per native tile; SAHI on 2 "
                  "swaths) -- what the B0-B4 table called B1 / B2 until 10 Oct 2026"}
    print(f"\n[vs the earlier small-sample figures] native tiles {camp['B1']['recall']['overall']} "
          f"(n={camp['B1']['gt_ships']}) -> {b1['recall']} (n={b1['n_gt']});  swaths "
          f"{camp['B2']['recall']['overall']} (n={camp['B2']['gt_ships']}) -> "
          f"{swath_out['sahi_B2']['recall']} (n={swath_out['sahi_B2']['n_gt']})")
    report["elapsed_s"] = round(time.perf_counter() - t_all, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wp24_detection_eval.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {args.out / 'wp24_detection_eval.json'} ({report['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
