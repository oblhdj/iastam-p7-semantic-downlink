"""WP11 -- the end-to-end integration demo: real image bytes through the real chain.

ARCHITECTURE.md section E item 1: "nothing runs tile -> prefilter -> detector -> LoD ->
scheduler -> ground". Every stage was measured in isolation, and WP6-WP10 replay a stored
catalogue (`wp6_tiles.csv`, `wp6_ships.csv`) rather than running the models. That replay had
never been checked against the thing it replays.

This script runs the actual chain on actual JPEGs:

    tile  ->  classic pre-filter  ->  learned gate  ->  detector  ->  LoD encoder
          ->  value-greedy scheduler over real orbit passes  ->  ground

and then does the part that makes it a *test* rather than a demo: it rebuilds the catalogue
schema from live inference and compares it, tile by tile, with the stored catalogue. If the
two agree, WP6-WP10 are replaying something faithful. If they do not, every downstream number
is suspect and this is where it shows up.

    .venv312/Scripts/python.exe scripts/wp11_integration_demo.py --tiles 400

Outputs: results/wp11_integration.json, wp11_funnel.png, wp11_per_tile.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
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

from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite
from sat7.prefilter import CLOUD, LAND, PrefilterConfig, run_prefilter
from sat7.real_workload import RealWorkloadConfig, load_size_model, workload_from_catalogue
from sat7.scheduler import LoDConfig, ValueGreedy, encode_lod, metrics, simulate
from wp3_train_gate import load_gate
# Reuse the catalogue's OWN post-processing rather than reimplementing it. The first version
# of this script counted a false alarm as "greedy, exclusive, IoU >= 0.5" while the catalogue
# uses "overlaps any GT box at IoU >= 0.3"; the cross-check then reported a 30% disagreement
# that was entirely my own divergence. Importing the real functions makes the comparison test
# the models and the pipeline, not two copies of a matching rule.
from wp6_build_catalogue import FP_THRESHOLDS, count_false_alarms, flag_predictions
from wp6_simulate_real import onboard_ceiling


def gt_boxes(label_path: Path, w: int = 768, h: int = 768):
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            out.append((float(p[1]) * w, float(p[2]) * h, float(p[3]) * w, float(p[4]) * h))
    return out


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=400, help="real tiles to push through the chain")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.pt")
    ap.add_argument("--gate", type=Path, default=ROOT / "models" / "gate.pt")
    ap.add_argument("--gate-thr", type=float, default=0.126,
                    help="gate threshold chosen on val (WP3), not on test")
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--day-tiles", type=int, default=40_000, help="tiles/day for the orbit sim")
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir, lbl_dir = args.data / "images" / "test", args.data / "labels" / "test"
    files = sorted(img_dir.glob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    picks = sorted(rng.choice(len(files), min(args.tiles, len(files)), replace=False))
    sample = [files[i] for i in picks]
    print("=== WP11 end-to-end integration: %d real tiles, device %s ===\n" % (len(sample), dev))

    report: dict = {"tiles": len(sample), "device": dev, "gate_threshold": args.gate_thr,
                    "det_thr": args.det_thr}
    stage_ms: dict = {}

    # ------------------------------------------------------------ stage 1: classic pre-filter
    print("[1/6] classic pre-filter (the cheap stage that runs on every tile) ...")
    rows = []
    # Keep only the gate-sized copy, never the full tile: at 768x768x3 the full images are
    # 1.77 MB each, so holding all 5,320 would need 9.4 GB and the run thrashes. The 128 px
    # copies are 261 MB. (Found the hard way on the first full-split run.)
    small = []
    gate_px = int(torch.load(args.gate, map_location="cpu")["size"])
    for f in sample:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        r = run_prefilter(img, PrefilterConfig())
        small.append(cv2.resize(img, (gate_px, gate_px), interpolation=cv2.INTER_AREA))
        del img
        rows.append({"image": f.name, "context": r.context, "cloud_frac": r.cloud_frac,
                     "edge_density": r.edge_density, "big_blob_frac": r.big_blob_frac,
                     "n_candidates": len(r.boxes), "prefilter_ms": r.runtime_ms})
    pre = pd.DataFrame(rows)
    stage_ms["prefilter"] = float(pre.prefilter_ms.median())
    print("      %d tiles, %.1f ms/tile (median, its own timer)" % (len(pre), stage_ms["prefilter"]))
    print("      contexts: %s" % pre.context.value_counts().to_dict())

    # ------------------------------------------------------------ stage 2: learned gate
    print("\n[2/6] learned gate (WP3, 47k parameters) ...")
    gate, gpx = load_gate(args.gate, dev)
    assert gpx == gate_px
    scores = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(small), 64):
            chunk = np.stack(small[i:i + 64]).transpose(0, 3, 1, 2)
            x = torch.from_numpy(chunk).float().div_(255.0).to(dev)
            scores += torch.sigmoid(gate(x)).float().cpu().tolist()
    del small
    stage_ms["gate"] = 1000 * (time.perf_counter() - t1) / len(sample)
    pre["gate_score"] = scores
    pre["gate_keep"] = pre.gate_score >= args.gate_thr
    n_keep = int(pre.gate_keep.sum())
    print("      keeps %d/%d tiles (%.1f%%), %.3f ms/tile incl. transfer"
          % (n_keep, len(pre), 100 * n_keep / len(pre), stage_ms["gate"]))

    # ------------------------------------------------------------ stage 3: detector
    # Run on ALL tiles so the gate's cost can be measured rather than assumed: the satellite
    # would only run the survivors, but then we could not say what the gate threw away.
    print("\n[3/6] detector (YOLO, all tiles so the gate's cost is measurable) ...")
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    ship_rows, pred_rows = [], []
    t2 = time.perf_counter()
    for i in range(0, len(sample), 16):
        chunk = sample[i:i + 16]
        res = model.predict([str(c) for c in chunk], imgsz=768, conf=0.05,
                            device=0 if dev == "cuda" else "cpu", verbose=False)
        for f, r in zip(chunk, res):
            preds = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                     for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            for cx, cy, w, h, cf in preds:          # raw boxes, same schema as wp1_predictions
                pred_rows.append({"image": f.name, "cx": cx, "cy": cy, "w": w, "h": h, "conf": cf})
            used = set()                            # GT matching: same rule as WP1 (greedy, 0.5)
            for g in gt_boxes(lbl_dir / (f.stem + ".txt")):
                best, bi = 0.0, -1
                for k, p in enumerate(preds):
                    if k in used:
                        continue
                    v = iou(g, p[:4])
                    if v > best:
                        best, bi = v, k
                if best >= 0.5:
                    used.add(bi)
                ship_rows.append({"image": f.name, "size_px": max(g[2], g[3]),
                                  "found": best >= 0.5,
                                  "conf": preds[bi][4] if bi >= 0 and best >= 0.5 else 0.0})
    stage_ms["detector"] = 1000 * (time.perf_counter() - t2) / len(sample)
    ships = pd.DataFrame(ship_rows)
    preds_df = pd.DataFrame(pred_rows)
    print("      %d ground-truth ships, %d raw boxes @0.05, %.1f ms/tile"
          % (len(ships), len(preds_df), stage_ms["detector"]))
    print("      recall @%.2f: %.4f" % (args.det_thr, (ships.conf >= args.det_thr).mean()))

    # ------------------------------------------------------------ stage 4: catalogue schema
    names = [f.name for f in sample]
    flags = flag_predictions(preds_df, lbl_dir, names)     # catalogue's rule: IoU >= 0.3, any GT
    fp = count_false_alarms(flags, names)
    n_ships = ships.groupby("image").size()
    tiles = (pre.merge(fp, on="image")
                .assign(n_ships=lambda d: d.image.map(n_ships).fillna(0).astype(int)))
    tiles["ctx3"] = np.where(tiles.context == CLOUD, "cloud",
                             np.where(tiles.context == LAND, "coast",
                                      np.where(tiles.n_ships > 0, "ships", "empty")))
    ships = ships.merge(tiles[["image", "ctx3", "cloud_frac"]], on="image", how="left")

    # ------------------------------------------------------------ stage 5: THE TEST
    print("\n[4/6] cross-check against the stored catalogue that WP6-WP10 replay ...")
    cat_t = pd.read_csv(args.out / "wp6_tiles.csv")
    cat_s = pd.read_csv(args.out / "wp6_ships.csv")
    m = tiles.merge(cat_t, on="image", suffixes=("_live", "_cat"))
    check = {"tiles_compared": int(len(m))}
    for col in ("context", "ctx3", "n_candidates", "n_ships", "n_fp_0.25"):
        a, b = m[col + "_live"], m[col + "_cat"]
        agree = float((a == b).mean())
        n_diff = int((a != b).sum())
        check[col] = {"agree": agree, "n_differ": n_diff}
        print("      %-4s %-16s agreement %.4f  (%d differ)"
              % ("OK" if agree == 1.0 else "DIFF", col, agree, n_diff))
    # Pair ships positionally within each tile. Merging on (image, size_px) cross-produces
    # whenever one tile holds two ships of the same length, which inflated the count above
    # the number of ships and manufactured confidence "differences" out of mispaired rows.
    def _rank(d):
        d = d.sort_values(["image", "size_px", "conf"], kind="mergesort").copy()
        d["k"] = d.groupby("image").cumcount()
        return d
    ms = _rank(ships).merge(_rank(cat_s[cat_s.image.isin(set(names))]),
                            on=["image", "k"], suffixes=("_live", "_cat"))
    if len(ms):
        dsize = float(np.abs(ms.size_px_live - ms.size_px_cat).max())
        dconf = float(np.abs(ms.conf_live - ms.conf_cat).max())
        n_conf_differ = int((np.abs(ms.conf_live - ms.conf_cat) > 1e-6).sum())
        check["ships_compared"] = int(len(ms))
        check["ship_size_max_abs_diff"] = dsize
        check["ship_conf_max_abs_diff"] = dconf
        check["ship_conf_n_differ"] = n_conf_differ
        print("      %-4s ship pairing       %d ships, max |size diff| %.2e"
              % ("OK" if dsize < 1e-6 else "DIFF", len(ms), dsize))
        print("      ---- ship confidences   max |diff| %.2e, %d of %d differ bitwise"
              % (dconf, n_conf_differ, len(ms)))
        # Bitwise difference is expected: a different batch shape makes cuDNN pick a different
        # convolution algorithm, so FP32 results move in the last few digits. The question that
        # matters is whether any of that drift crosses a threshold the pipeline branches on.
        flips = {}
        for name, thr in (("det_thr", args.det_thr), ("conf_low", args.det_thr),
                          ("conf_high", 0.9)):
            flips[name] = int(((ms.conf_live >= thr) != (ms.conf_cat >= thr)).sum())
        check["decision_flips"] = flips
        worst = max(flips.values())
        print("      %-4s decisions changed  %s  <- the number that actually matters"
              % ("OK" if worst == 0 else "DIFF",
                 ", ".join("%s: %d" % (k, v) for k, v in flips.items())))
    report["catalogue_cross_check"] = check

    # ------------------------------------------------------------ stage 6: day + downlink
    print("\n[5/6] scale to a day, encode, schedule over real orbit passes ...")
    cfg = RealWorkloadConfig(tiles_per_day=args.day_tiles, det_thr=args.det_thr, seed=args.seed)
    wl, stats = workload_from_catalogue(tiles, ships, cfg)

    # How thin is the pool each context is resampled from? The orbit mix draws a *fixed share*
    # of the day from each context regardless of how many real tiles back it, so a rare context
    # can have one tile stamped out thousands of times -- and that tile's idiosyncrasies then
    # drive a headline number. This is the check that caught it.
    pool = tiles.ctx3.value_counts().to_dict()
    thin = {}
    print("      source pool backing each context of the simulated day:")
    for ctx, drawn in sorted(stats.context_counts.items(), key=lambda kv: -kv[1]):
        have = int(pool.get(ctx, 0))
        reuse = drawn / have if have else float("inf")
        mark = "  <-- THIN" if have < 30 else ""
        if have < 30:
            thin[ctx] = {"real_tiles": have, "drawn": int(drawn), "reuse_x": round(reuse, 1)}
        print("        %-6s %6d drawn from %4d real tiles  (x%.0f reuse)%s"
              % (ctx, drawn, have, reuse, mark))
    report["thin_context_pools"] = thin
    if thin:
        print("      WARNING: %s backed by <30 real tiles -- absolute recall from this run is"
              % ", ".join(thin))
        print("               NOT comparable with WP6, which draws from all 5,320.")
    lod = LoDConfig(conf_low=args.det_thr,
                    size_model=load_size_model(args.out / "wp6_size_model.json"))
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    passes = find_passes(sat, GroundStation(), start, 36, LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    items = encode_lod(wl, lod)
    res = simulate(items, passes, start, ValueGreedy(), args.storage_gb * 1e9, encoder="LoD")
    mt = metrics(res, wl, lod)
    mt.update(onboard_ceiling(wl, lod, "lod"))     # what the satellite still knew after stage 3
    mt["downlink_eff"] = (mt["ship_recall"] / mt["ceiling_recall"]
                          if mt["ceiling_recall"] else float("nan"))
    print("      day: %d tiles, %d ships, %d passes, %.0f MB capacity"
          % (stats.tiles, stats.ships, len(passes),
             sum(p.capacity_bytes for p in passes) / 1e6))
    print("      %d items encoded, %.1f MB offered" % (len(items), mt["MB_offered"]))

    # ------------------------------------------------------------ ground
    print("\n[6/6] ground segment ...")
    print("      ships recovered        %.4f  (ceiling %.4f = what the satellite still knew)"
          % (mt["ship_recall"], mt["ceiling_recall"]))
    print("      dark vessels recovered %.4f" % mt["dark_recall"])
    print("      bytes sent             %.1f MB of %.1f offered"
          % (mt["MB_sent"], mt["MB_offered"]))
    print("      median latency         %.2f h" % mt["latency_med_h"])
    report["day"] = {"tiles": stats.tiles, "ships": stats.ships, "passes": len(passes),
                     "items": len(items)}
    report["ground"] = {k: float(mt[k]) for k in
                        ("ship_recall", "ceiling_recall", "dark_recall", "downlink_eff",
                         "MB_sent", "MB_offered", "latency_med_h")}
    report["stage_ms_per_tile"] = stage_ms

    args.out.mkdir(parents=True, exist_ok=True)
    tiles.to_csv(args.out / "wp11_per_tile.csv", index=False)
    (args.out / "wp11_integration.json").write_text(json.dumps(report, indent=2))

    # ------------------------------------------------------------ figure
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    names = ["pre-filter\n(every tile)", "learned gate\n(every tile)", "detector\n(per tile)"]
    vals = [stage_ms["prefilter"], stage_ms["gate"], stage_ms["detector"]]
    bars = a0.bar(names, vals, color=["#C0392B", "#2E8B57", "#1B6CA8"])
    a0.set_yscale("log")
    a0.set_ylabel("ms per tile (log scale)")
    a0.set_title("Onboard cost of each stage, measured end to end", fontsize=10)
    for b, v in zip(bars, vals):
        a0.text(b.get_x() + b.get_width() / 2, v * 1.2, "%.2f" % v, ha="center", fontsize=9)
    a0.grid(alpha=0.25, axis="y")

    a1.bar(["offered\nby encoder", "sent\nover link"],
           [mt["MB_offered"], mt["MB_sent"]], color=["#9CA3AF", "#1B6CA8"])
    a1.set_ylabel("MB per day")
    a1.set_title("Downlink: %.1f%% of ships recovered\n(ceiling %.1f%% = limit of what the detector saw)"
                 % (100 * mt["ship_recall"], 100 * mt["ceiling_recall"]), fontsize=10)
    a1.grid(alpha=0.25, axis="y")
    for ax in (a0, a1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "wp11_funnel.png", dpi=160)
    print("\nSaved wp11_integration.json, wp11_per_tile.csv, wp11_funnel.png in %s" % args.out)


if __name__ == "__main__":
    main()
