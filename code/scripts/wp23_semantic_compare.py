"""WP23 -- LoD ladder vs the paper's P0-P3 scheme, and the joint program, on the real catalogue.

This is the configuration-switch deliverable: it runs the SAME real-detection workload, passes and
scheduler under both semantic encoders (`sat7.priority.encode_semantic(mode=...)`) and reports them
side by side, then solves the paper's joint program (`sat7.joint`) over the same objects at several
alpha/beta/gamma weightings. Nothing here needs torch -- it reuses the stored WP6 catalogue (report
12 proved live inference reproduces it to 1e-4), so it runs in the non-torch .venv:

    .venv/Scripts/python.exe scripts/wp23_semantic_compare.py
    .venv/Scripts/python.exe scripts/wp23_semantic_compare.py --day-tiles 40000 --link-share 0.25

Labels (project rule): recall/MB/latency are SIM-over-REAL (scheduler sim over real detections) and
the recall is at the workload's assumed cloud fraction -- never quote it bare (START_HERE section 7).
The joint program's D is SIM, T is SIM, E is TARGET (placeholder powers) unless calibrated powers are
injected; alpha/beta/gamma and the P0-P3 thresholds are ASSUMPTION. No new measured number is minted.

Outputs: results/wp23_semantic_compare.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from sat7.joint import (CostModel, JointConstraints, JointWeights, build_objects, pareto, solve)
from sat7.orbit import GroundStation, LINK_PRESETS, OrbitConfig, find_passes, make_satellite
from sat7.priority import PriorityConfig, encode_semantic, level_histogram
from sat7.real_workload import (RealWorkloadConfig, attach_coast_sizes, load_size_model,
                                workload_from_catalogue)
from sat7.scheduler import LoDConfig, ValueGreedy, metrics, simulate


def _sim(wl, items, lod, passes, start, storage_gb, encoder):
    res = simulate(items, passes, start, ValueGreedy(), storage_gb * 1e9, encoder=encoder)
    return metrics(res, wl, lod)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat-tiles", type=Path, default=ROOT / "results" / "wp6_tiles.csv")
    ap.add_argument("--cat-ships", type=Path, default=ROOT / "results" / "wp6_ships.csv")
    ap.add_argument("--size-model", type=Path, default=ROOT / "results" / "wp6_size_model.json")
    ap.add_argument("--day-tiles", type=int, default=40_000)
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    tiles = pd.read_csv(args.cat_tiles)
    attach_coast_sizes(tiles, Path(args.cat_tiles).parent)   # each coastal tile at its measured size (wp28)
    ships = pd.read_csv(args.cat_ships)
    cfg = RealWorkloadConfig(tiles_per_day=args.day_tiles, det_thr=args.det_thr, seed=args.seed)
    wl, stats = workload_from_catalogue(tiles, ships, cfg)
    lod = LoDConfig(conf_low=args.det_thr, size_model=load_size_model(args.size_model))
    pcfg = PriorityConfig(p1_conf=lod.conf_high)             # share the LoD operating point

    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    passes = find_passes(sat, GroundStation(), start, 36, LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]

    print(f"=== WP23 semantic-scheme comparison (SIM-over-REAL) ===")
    print(f"workload: {stats.tiles} tiles, {stats.ships} ships ({stats.detected} detected), "
          f"{len(passes)} passes, link_share {args.link_share}\n")

    # ---- the two semantic encoders, same scheduler + metrics -----------------------------------
    schemes = {}
    for mode, name in (("lod", "LoD ladder (repo)"), ("priority", "P0-P3 (paper Table I)")):
        items = encode_semantic(wl, mode=mode, lod=lod, cfg=pcfg)
        m = _sim(wl, items, lod, passes, start, args.storage_gb, name)
        schemes[mode] = {"name": name, "n_items": len(items),
                         "ship_recall": round(float(m["ship_recall"]), 4),
                         "dark_recall": round(float(m["dark_recall"]), 4),
                         "MB_sent": round(float(m["MB_sent"]), 1),
                         "MB_offered": round(float(m["MB_offered"]), 1),
                         "value_frac": round(float(m["value_frac"]), 4),
                         "latency_med_h": round(float(m["latency_med_h"]), 2),
                         "latency_p90_h": round(float(m["latency_p90_h"]), 2)}

    hist = level_histogram(wl, lod, pcfg)

    # ---- the joint program over the same objects ------------------------------------------------
    objs = build_objects(wl, lod, pcfg, CostModel())
    grid = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    con = JointConstraints(a_min=0.90, i_min=0.50)
    joint_rows = pareto(objs, con, grid=grid)
    headline = solve(objs, JointWeights(1, 1, 1), con)

    report = {
        "label_note": "recall/MB/latency SIM-over-REAL (scheduler sim over real detections); recall at "
                      "assumed cloud fraction, never bare (START_HERE 7). Joint: D/T SIM, E TARGET "
                      "(placeholder powers), alpha/beta/gamma + P0-P3 thresholds ASSUMPTION.",
        "workload": {"tiles": stats.tiles, "ships": stats.ships, "detected": stats.detected,
                     "dark": stats.dark, "passes": len(passes), "link_share": args.link_share,
                     "cloud_caveat": "recall band 0.40-0.82 over 0-50% cloud"},
        "semantic_schemes": schemes,
        "p0_p3_level_histogram": hist,
        "joint_program": {
            "objects": len(objs), "constraints": {"a_min": con.a_min, "i_min": con.i_min},
            "headline_1_1_1": {"J": round(headline.J, 4), "E_total_J": round(headline.E_total_J, 2),
                               "D_tx_MB": round(headline.D_tx_MB, 2), "T_total_h": round(headline.T_total_h, 2),
                               "accuracy": round(headline.accuracy, 4),
                               "info_preservation": round(headline.info_preservation, 4),
                               "feasible": headline.feasible, "gap_vs_lower_bound": round(headline.gap, 4)},
            "weighting_sweep": joint_rows},
    }
    (args.out / "wp23_semantic_compare.json").write_text(json.dumps(report, indent=2, default=float))

    # ---- print ----------------------------------------------------------------------------------
    print(f"{'scheme':24s} {'recall':>7s} {'dark':>6s} {'MB sent':>8s} {'val':>6s} {'lat_med_h':>10s}")
    for mode in ("lod", "priority"):
        s = schemes[mode]
        print(f"{s['name']:24s} {s['ship_recall']:7.4f} {s['dark_recall']:6.3f} "
              f"{s['MB_sent']:8.1f} {s['value_frac']:6.3f} {s['latency_med_h']:10.2f}")
    print(f"\nP0-P3 level histogram (detections): {hist}")
    print(f"\njoint program ({len(objs)} objects, A_min {con.a_min}, I_min {con.i_min}):")
    print(f"  {'(a,b,g)':12s} {'J':>10s} {'E_J':>10s} {'D_MB':>8s} {'T_h':>8s} {'acc':>6s} {'info':>6s} feas")
    for r in joint_rows:
        print(f"  ({r['alpha']},{r['beta']},{r['gamma']})      {r['J']:10.3f} {r['E_total_J']:10.2f} "
              f"{r['D_tx_MB']:8.2f} {r['T_total_h']:8.2f} {r['accuracy']:6.3f} "
              f"{r['info_preservation']:6.3f} {r['feasible']}")
    print(f"\nSaved wp23_semantic_compare.json in {args.out}")


if __name__ == "__main__":
    main()
