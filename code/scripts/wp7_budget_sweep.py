"""WP7 step 2: spend the byte budget where it buys ships.

WP6 measured that 86% of our downlink is coastal tiles (62%) and blanket thumbnails (24%),
while actual ship information is 13%. WP7 measured, on real tiles, what the cheaper
versions of those two products cost. This script puts the options through the real
workload and reports what each one buys, so the design choice is made on evidence.

Options compared (every byte figure MEASURED, none assumed):
  coastal handling  whole tile q60 (42.9 kB) | q40 (31.4 kB) | q30 (25.2 kB)
                    | ROI mosaic of detected ships (2.3 kB, no safety net for misses)
  thumbnails        every tile | gated: skipped where the classic pre-filter sees no
                    candidate AND the detector fired nothing (2% audit sample kept)

    python scripts/wp7_budget_sweep.py
    python scripts/wp7_budget_sweep.py --tiles 80000    # the congested regime

Output: results/wp7_budget_sweep.csv|.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite
from sat7.real_workload import (RealWorkloadConfig, load_catalogue, load_size_model,
                                workload_from_catalogue)
from sat7.scheduler import LoDConfig, ValueGreedy, encode_lod, metrics, simulate

# (label, coast_mode, coast bytes) -- bytes measured in wp7_cheap_products.json
# (label, coast_mode, whole-tile bytes, escalation threshold) -- bytes from wp7_cheap_products
COAST = [
    ("tile q60 (interim)", "tile", 41_839.0, 0),
    ("tile q40 (current)", "tile", 31_426.0, 0),
    ("tile q30", "tile", 25_202.0, 0),
    ("ROI mosaic only", "mosaic", 2_259.0, 0),
    ("adaptive q40, escalate>=25", "adaptive", 31_426.0, 25),
    ("adaptive q40, escalate>=11", "adaptive", 31_426.0, 11),
    ("adaptive q40, escalate>=6", "adaptive", 31_426.0, 6),
    ("adaptive q40, escalate>=3", "adaptive", 31_426.0, 3),
    ("adaptive q40, escalate>=1", "adaptive", 31_426.0, 1),
]


def bytes_by_kind(items) -> dict:
    d = {}
    for i in items:
        d[i.kind] = d.get(i.kind, 0.0) + i.size
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=40_000)
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--link", choices=sorted(LINK_PRESETS), default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    out = args.results

    tiles, ships = load_catalogue(out)
    sm = load_size_model(out / "wp6_size_model.json")
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    passes = find_passes(make_satellite(OrbitConfig(epoch=start)), GroundStation(), start, 36,
                         LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    storage = args.storage_gb * 1e9

    rows = []
    for rep in range(args.repeats):
        cfg = RealWorkloadConfig(tiles_per_day=args.tiles, det_thr=args.det_thr, seed=args.seed + rep)
        wl, st = workload_from_catalogue(tiles, ships, cfg)
        for coast_label, mode, cbytes, esc in COAST:
            for gate in (False, True):
                lod = LoDConfig(conf_low=args.det_thr, size_model=sm, coast_mode=mode,
                                coast_tile_bytes=cbytes, coast_escalate=esc,
                                gate_thumbnails=gate)
                items = encode_lod(wl, lod)
                m = metrics(simulate(items, passes, start, ValueGreedy(), storage), wl, lod)
                kb = bytes_by_kind(items)
                ship_bytes = sum(kb.get(k, 0.0) for k in ("L0", "L1", "L2"))
                rows.append(m | {
                    "coast": coast_label, "thumbnails": "gated" if gate else "every tile",
                    "rep": rep,
                    "MB_coast": (kb.get("tile", 0) + kb.get("mosaic", 0)) / 1e6,
                    "n_whole_tiles": sum(1 for i in items if i.kind == "tile"),
                    "MB_thumb": kb.get("thumb", 0) / 1e6,
                    "MB_ships": ship_bytes / 1e6,
                    "semantic_share": ship_bytes / max(1e-9, sum(kb.values())),
                    "n_thumb": sum(1 for i in items if i.kind == "thumb"),
                })
    d = pd.DataFrame(rows)
    agg = (d.groupby(["coast", "thumbnails"], sort=False)
             .agg(ship_recall=("ship_recall", "mean"), dark_recall=("dark_recall", "mean"),
                  latency_med_h=("latency_med_h", "mean"), MB_offered=("MB_offered", "mean"),
                  MB_sent=("MB_sent", "mean"), MB_coast=("MB_coast", "mean"),
                  MB_thumb=("MB_thumb", "mean"), MB_ships=("MB_ships", "mean"), n_whole_tiles=("n_whole_tiles", "mean"),
                  semantic_share=("semantic_share", "mean"))
             .reset_index())
    base = agg.iloc[0]
    agg["d_recall"] = agg.ship_recall - base.ship_recall
    agg["bytes_vs_base"] = agg.MB_offered / base.MB_offered
    agg.to_csv(out / "wp7_budget_sweep.csv", index=False)

    print(f"REAL workload, {args.tiles:,} tiles/day, {args.repeats} resampled days, "
          f"value-greedy, {sum(p.capacity_bytes for p in passes)/1e6:.0f} MB capacity\n")
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(agg[["coast", "thumbnails", "ship_recall", "d_recall", "dark_recall",
                   "MB_offered", "bytes_vs_base", "MB_coast", "n_whole_tiles", "MB_ships",
                   "semantic_share", "latency_med_h"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    a = axes[0]
    for th, g in agg.groupby("thumbnails"):
        a.plot(g.MB_offered, g.ship_recall, "o-" if th == "every tile" else "s--",
               label=f"thumbnails: {th}")
        for r in g.itertuples():
            a.annotate(r.coast, (r.MB_offered, r.ship_recall), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
    a.set(xlabel="MB generated per day (offered)", ylabel="Fraction of real ships delivered",
          title="What each byte actually buys")
    a.legend(fontsize=8, frameon=False)
    a.grid(alpha=0.25)

    b = axes[1]
    w = agg[agg.thumbnails == "every tile"]
    x = np.arange(len(w))
    b.bar(x, w.MB_coast, 0.6, label="coastal", color="#C0392B")
    b.bar(x, w.MB_thumb, 0.6, bottom=w.MB_coast, label="thumbnails", color="#9CA3AF")
    b.bar(x, w.MB_ships, 0.6, bottom=w.MB_coast + w.MB_thumb, label="ship information",
          color="#E07A1F")
    b.set_xticks(x, w.coast, fontsize=7, rotation=12)
    b.set(ylabel="MB per day", title="Where the budget goes")
    b.legend(fontsize=8, frameon=False)
    b.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"WP7 -- byte budget on the REAL workload ({args.tiles:,} tiles/day)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "wp7_budget_sweep.png", dpi=160)
    print(f"\nSaved wp7_budget_sweep.csv|.png in {out}")


if __name__ == "__main__":
    main()
