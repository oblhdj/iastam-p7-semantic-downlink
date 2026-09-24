"""Contribution A experiment: what reaches the ground, and when, for each strategy.

Compares (same synthetic day, same passes):
  1. Bent pipe          raw tiles,                          FIFO
  2. Phi-sat-2 style    fixed patch per detected ship,      FIFO
  3. Ours, no buffer    confidence-aware LoD,               data older than the last pass is lost
  4. Ours, FIFO         confidence-aware LoD,               oldest first
  5. Ours (full)        confidence-aware LoD,               value-greedy + aging + truncation

Then sweeps the offered load to show where each strategy breaks.

    python scripts/simulate_day.py
    python scripts/simulate_day.py --tiles 80000 --link-share 0.1 --seed 2

Outputs in results/: sim_table.csv, sim_recall.png, sim_latency.png, sim_load_sweep.png
Everything here is a SIMULATION with a SYNTHETIC workload: say so on every slide.
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
from sat7.scheduler import (FIFO, LoDConfig, NoBuffer, ValueGreedy, WorkloadConfig, encode_fixed_patch,
                            encode_lod, encode_raw, generate_workload, metrics, ship_ci95, simulate)

COLORS = {"Bent pipe (raw, FIFO)": "#C0392B", "Phi-sat-2 style (patch, FIFO)": "#6B7280",
          "Ours: LoD + no buffer": "#8DB8E0", "Ours: LoD + FIFO": "#1B6CA8",
          "Ours: LoD + value-greedy": "#E07A1F"}


def strategies():
    return [
        ("Bent pipe (raw, FIFO)", encode_raw, FIFO()),
        ("Phi-sat-2 style (patch, FIFO)", encode_fixed_patch, FIFO()),
        ("Ours: LoD + no buffer", encode_lod, NoBuffer()),
        ("Ours: LoD + FIFO", encode_lod, FIFO()),
        ("Ours: LoD + value-greedy", encode_lod, ValueGreedy()),
    ]


def run_all(wl, passes, start, storage, lod):
    rows = []
    for label, enc, pol in strategies():
        m = metrics(simulate(enc(wl, lod), passes, start, pol, storage, encoder=label), wl, lod)
        m["strategy"] = label
        rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=40_000, help="tiles imaged per day")
    ap.add_argument("--link", choices=sorted(LINK_PRESETS), default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25,
                    help="fraction of each pass given to this payload (rest: telemetry, other data)")
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    # arrivals during 24 h, passes during 36 h so the last captures get a chance
    passes = find_passes(sat, GroundStation(), start, 36, LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    lod = LoDConfig()
    wl = generate_workload(WorkloadConfig(tiles_per_day=args.tiles, seed=args.seed))
    storage = args.storage_gb * 1e9

    df = run_all(wl, passes, start, storage, lod)
    n = len(wl.ships)
    df["recall_ci95"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in (ship_ci95(r, n) for r in df.ship_recall)]
    cols = ["strategy", "ship_recall", "recall_ci95", "dark_recall", "value_frac", "latency_med_h",
            "latency_p90_h", "dark_latency_med_h", "MB_sent", "MB_offered", "dropped_storage"]
    args.out.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(args.out / "sim_table.csv", index=False)

    print(f"SIMULATION -- synthetic workload: {args.tiles} tiles/day, {n} true ships "
          f"({sum(s.dark for s in wl.ships)} dark), seed {args.seed}")
    print(f"{len(passes)} passes in 36 h, link {args.link} x share {args.link_share} "
          f"-> {sum(p.capacity_bytes for p in passes) / 1e6:.0f} MB capacity\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(df[cols].to_string(index=False))

    # ---- chart 1: recall
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.ship_recall, 0.4, label="All ships", color=[COLORS[s] for s in df.strategy])
    ax.bar(x + 0.2, df.dark_recall, 0.4, label="Dark vessels", color=[COLORS[s] for s in df.strategy],
           hatch="//", edgecolor="white")
    ax.set_xticks(x, [s.replace(": ", ":\n").replace(" (", "\n(") for s in df.strategy], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction delivered to ground")
    ax.set_title("Ships reaching the ground in one day (plain = all, hatched = dark vessels)\n"
                 "SIMULATION, synthetic workload", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "sim_recall.png", dpi=160)

    # ---- chart 2: latency
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, df.latency_med_h, 0.4, label="median, all ships", color="#1B6CA8")
    ax.bar(x + 0.2, df.dark_latency_med_h, 0.4, label="median, dark vessels", color="#E07A1F")
    ax.set_xticks(x, [s.replace(": ", ":\n").replace(" (", "\n(") for s in df.strategy], fontsize=8)
    ax.set_ylabel("Capture -> ground (hours)")
    ax.set_title("Latency of delivered ships -- SIMULATION", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "sim_latency.png", dpi=160)

    # ---- chart 3: load sweep
    loads = [5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
    sweep = []
    for tiles in loads:
        w = generate_workload(WorkloadConfig(tiles_per_day=tiles, seed=args.seed))
        d = run_all(w, passes, start, storage, lod)
        d["tiles"] = tiles
        sweep.append(d)
    sw = pd.concat(sweep)
    sw.to_csv(args.out / "sim_load_sweep.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for strat, g in sw.groupby("strategy", sort=False):
        axes[0].plot(g.tiles, g.ship_recall, "o-", color=COLORS[strat], label=strat)
        axes[1].plot(g.tiles, g.dark_recall, "o-", color=COLORS[strat], label=strat)
    for a, t in zip(axes, ["All ships", "Dark vessels"]):
        a.set_xscale("log")
        a.set_xlabel("Tiles imaged per day (offered load)")
        a.set_title(t, fontsize=10)
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction delivered to ground")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle("When does each strategy break? -- SIMULATION, synthetic workload", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.out / "sim_load_sweep.png", dpi=160)
    print(f"\nSaved sim_table.csv, sim_recall.png, sim_latency.png, sim_load_sweep.png in {args.out}")


if __name__ == "__main__":
    main()
