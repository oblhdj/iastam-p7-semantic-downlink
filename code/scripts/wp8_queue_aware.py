"""WP8: let the buffer decide how much detail a coastal tile deserves.

WP7 found that adaptive coastal detail (escalate to a whole tile only where the classic
stage saw objects the network did not confirm) wins under congestion and loses at nominal
load -- because a fixed threshold cannot know which regime it is in. The satellite can:
the buffer is right there when the compressor runs.

``coast_mode="queue"`` reads the pressure -- bytes queued against the link capacity coming
up in the next 12 h, both known onboard -- and slides the escalation threshold between
``esc_min`` (plenty of room: keep the safety net) and ``esc_max`` (hopeless backlog: whole
tiles would only crowd out ship reports).

This closes the loop between our two contributions: the encoder stops being a fixed policy
and becomes part of the scheduler.

    python scripts/wp8_queue_aware.py
    python scripts/wp8_queue_aware.py --loads 40000 160000

Output: results/wp8_queue_aware.csv|.png, wp8_control_trace.png
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
from sat7.scheduler import LoDConfig, ValueGreedy, metrics, simulate_online

# (label, coast_mode, fixed escalation threshold)
POLICIES = [
    ("whole tile always", "tile", 0),
    ("adaptive, fixed >=3", "adaptive", 3),
    ("adaptive, fixed >=11", "adaptive", 11),
    ("adaptive, fixed >=25", "adaptive", 25),
    ("mosaic always", "mosaic", 0),
    ("queue-aware (ours)", "queue", 0),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loads", type=int, nargs="+",
                    default=[20_000, 40_000, 80_000, 160_000, 320_000])
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--link", choices=sorted(LINK_PRESETS), default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2)
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

    rows, traces = [], {}
    for load in args.loads:
        for rep in range(args.repeats):
            cfg = RealWorkloadConfig(tiles_per_day=load, det_thr=args.det_thr, seed=args.seed + rep)
            wl, st = workload_from_catalogue(tiles, ships, cfg)
            for label, mode, esc in POLICIES:
                lod = LoDConfig(conf_low=args.det_thr, size_model=sm, coast_mode=mode,
                                coast_escalate=esc or LoDConfig.coast_escalate)
                res, trace = simulate_online(wl, lod, passes, start, ValueGreedy(), storage,
                                             encoder=label)
                m = metrics(res, wl, lod)
                rows.append(m | {"policy": label, "tiles": load, "rep": rep})
                if mode == "queue" and rep == 0:
                    traces[load] = trace

    d = pd.DataFrame(rows)
    agg = (d.groupby(["tiles", "policy"], sort=False)
             .agg(ship_recall=("ship_recall", "mean"), dark_recall=("dark_recall", "mean"),
                  MB_offered=("MB_offered", "mean"), MB_sent=("MB_sent", "mean"),
                  latency_med_h=("latency_med_h", "mean"))
             .reset_index())
    agg.to_csv(out / "wp8_queue_aware.csv", index=False)

    print(f"REAL workload, online encode+schedule, value-greedy, {args.repeats} days each,"
          f" link share {args.link_share}\n")
    for load, g in agg.groupby("tiles"):
        best = g.loc[g.ship_recall.idxmax()]
        print(f"--- {load:,} tiles/day "
              f"(best: {best.policy}, recall {best.ship_recall:.3f}) ---")
        with pd.option_context("display.width", 200, "display.precision", 3):
            print(g[["policy", "ship_recall", "dark_recall", "MB_offered", "MB_sent",
                     "latency_med_h"]].to_string(index=False))
        print()

    # ---- the metric that matters: how much do you lose by committing to one policy
    # before you know which regime the satellite will be in?
    agg["regret"] = agg.groupby("tiles").ship_recall.transform("max") - agg.ship_recall
    reg = (agg.groupby("policy", sort=False).regret.agg(worst_case="max", mean="mean")
              .sort_values("worst_case").reset_index())
    print("Regret against the best policy for each load "
          "(how much a fixed choice costs when the regime is unknown):")
    with pd.option_context("display.precision", 4):
        print(reg.to_string(index=False))
    print()
    agg.to_csv(out / "wp8_queue_aware.csv", index=False)

    # ---- chart 1: recall vs load, one line per policy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    a = axes[0]
    for pol, g in agg.groupby("policy", sort=False):
        style = "o-" if pol == "queue-aware (ours)" else "s--"
        lw = 2.4 if pol == "queue-aware (ours)" else 1.2
        a.plot(g.tiles, g.ship_recall, style, lw=lw, label=pol)
    a.set(xscale="log", xlabel="Tiles imaged per day", ylabel="Fraction of real ships delivered",
          title="A fixed coastal policy is only right in one regime")
    a.legend(fontsize=7, frameon=False)
    a.grid(alpha=0.25)
    a.spines[["top", "right"]].set_visible(False)

    b = axes[1]
    for pol, g in agg.groupby("policy", sort=False):
        style = "o-" if pol == "queue-aware (ours)" else "s--"
        lw = 2.4 if pol == "queue-aware (ours)" else 1.2
        b.plot(g.MB_offered, g.ship_recall, style, lw=lw, label=pol)
    b.set(xlabel="MB generated per day", ylabel="Fraction delivered",
          title="... and it is not on the efficient frontier either")
    b.grid(alpha=0.25)
    b.spines[["top", "right"]].set_visible(False)
    fig.suptitle("WP8 -- queue-aware coastal detail, REAL detections", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "wp8_queue_aware.png", dpi=160)

    # ---- chart 2: the control law doing its job
    if traces:
        fig, ax = plt.subplots(figsize=(9, 4))
        for load, tr in sorted(traces.items()):
            if tr["escalate"]:
                ax.plot(np.linspace(0, 24, len(tr["escalate"])), tr["escalate"],
                        lw=1.2, label=f"{load:,} tiles/day")
        ax.set(xlabel="hour of the day", ylabel="escalation threshold (unconfirmed candidates)",
               title="The control law: how much coastal detail the buffer allows\n"
                     "low = send whole tiles, high = send cheap mosaics")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "wp8_control_trace.png", dpi=160)

    print(f"Saved wp8_queue_aware.csv|.png and wp8_control_trace.png in {out}")


if __name__ == "__main__":
    main()
