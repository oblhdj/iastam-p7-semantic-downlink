"""WP5b: bound the loss over the WHOLE DAY, not just inside one contact window.

`wp5_optimality_gap.py` measured the per-window gap and said so plainly: it answers
"given the queue greedy got itself into, how much did it leave behind in that window?",
and it does **not** bound the end-to-end loss, because it never questions the choices that
produced the queue. This closes that hole, and with it the plan's unstarted nice-to-have
"offline LP upper bound".

Scheduling every window jointly is NP-hard (multi-dimensional knapsack with release times),
so instead of solving it we bracket it:

    online greedy   <=   clairvoyant   <=   joint optimum   <=   upper bound
    (what we do)         (foresight,          (unknown)         (relaxation)
                          feasible)

* **upper bound** -- integrality relaxed, every byte credited at the best rate its item
  could ever reach, capacity and release times respected (`joint_upper_bound`). Validated
  against brute-force enumeration over all pass assignments.
* **clairvoyant** -- a real, feasible schedule that knows the whole day in advance and
  places each item in the earliest window it can reach (`clairvoyant_value`).

Two numbers come out of it:
  total gap        (bound - online) / bound      -- everything still on the table
  price of online  (clairvoyant - online) / bound -- the part caused by not seeing the future

    .venv/Scripts/python.exe scripts/wp5_joint_bound.py

Output: results/wp5_joint_bound.csv|.png
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

from sat7.optimum import clairvoyant_value, item_value, joint_upper_bound
from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite
from sat7.real_workload import (RealWorkloadConfig, load_catalogue, load_size_model,
                                workload_from_catalogue)
from sat7.scheduler import FIFO, LoDConfig, ValueGreedy, encode_lod, simulate


def delivered_value(res) -> float:
    return sum(item_value(s.item, s.fraction) for s in res.sent)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loads", type=int, nargs="+", default=[20_000, 40_000, 80_000, 160_000, 320_000])
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
    lod = LoDConfig(conf_low=args.det_thr, size_model=sm)
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    passes = find_passes(make_satellite(OrbitConfig(epoch=start)), GroundStation(), start, 36,
                         LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    storage = args.storage_gb * 1e9

    rows = []
    for load in args.loads:
        for rep in range(args.repeats):
            cfg = RealWorkloadConfig(tiles_per_day=load, det_thr=args.det_thr, seed=args.seed + rep)
            wl, _ = workload_from_catalogue(tiles, ships, cfg)
            items = encode_lod(wl, lod)
            bound = joint_upper_bound(items, passes, start)
            clair = clairvoyant_value(items, passes, start)
            greedy = delivered_value(simulate(items, passes, start, ValueGreedy(), storage))
            fifo = delivered_value(simulate(items, passes, start, FIFO(), storage))
            rows.append({"tiles": load, "rep": rep, "bound": bound, "clairvoyant": clair,
                         "online_greedy": greedy, "online_fifo": fifo,
                         "offered_value": sum(i.value for i in items)})
    d = pd.DataFrame(rows)
    # the three quantities coincide exactly when nothing binds, so clip away the 1e-12
    # float noise rather than printing a negative gap
    clip = lambda x: x.clip(lower=0.0)
    d["total_gap"] = clip((d.bound - d.online_greedy) / d.bound)
    d["price_of_online"] = clip((d.clairvoyant - d.online_greedy) / d.bound)
    d["clair_gap"] = clip((d.bound - d.clairvoyant) / d.bound)
    d["fifo_gap"] = clip((d.bound - d.online_fifo) / d.bound)
    agg = d.groupby("tiles").mean(numeric_only=True).drop(columns=["rep"]).reset_index()
    agg.to_csv(out / "wp5_joint_bound.csv", index=False)

    bad = (d.online_greedy > d.bound + 1e-6).sum() + (d.clairvoyant > d.bound + 1e-6).sum()
    print(f"ordering check (online <= bound, clairvoyant <= bound): "
          f"{'all pass' if bad == 0 else f'{bad} VIOLATIONS'}\n")
    print("Whole-day value, REAL workload (value units, not ships):")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(agg[["tiles", "online_greedy", "clairvoyant", "bound",
                   "total_gap", "price_of_online", "clair_gap", "fifo_gap"]].to_string(index=False))

    sat = agg[agg.total_gap > 0.01]
    print(f"\nHEADLINE -- over the whole day the value-greedy schedule sits within "
          f"{100 * agg.total_gap.max():.1f}% of a bound on the best possible schedule "
          f"(worst load), and the part attributable to not knowing the future is at most "
          f"{100 * agg.price_of_online.max():.1f}%.")
    if len(sat):
        print(f"At the loads where anything binds ({', '.join(f'{t:,}' for t in sat.tiles)} "
              f"tiles/day) the total gap averages {100 * sat.total_gap.mean():.1f}%.")
    print(f"FIFO, for contrast, is {100 * agg.fifo_gap.max():.1f}% below the bound at worst.")
    print("\nThe bound is a relaxation, so the true optimum sits somewhere between "
          "'clairvoyant' and 'bound' -- the real loss is smaller than 'total gap' suggests.")

    # ---- chart
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a = axes[0]
    a.plot(agg.tiles, agg.bound, "k--", lw=1.4, label="upper bound (relaxation)")
    a.plot(agg.tiles, agg.clairvoyant, "s-", color="#2E8B57", label="clairvoyant (feasible)")
    a.plot(agg.tiles, agg.online_greedy, "o-", color="#E07A1F", lw=2.2, label="ours, online greedy")
    a.plot(agg.tiles, agg.online_fifo, "^-", color="#C0392B", label="online FIFO")
    a.fill_between(agg.tiles, agg.clairvoyant, agg.bound, color="#9CA3AF", alpha=0.25,
                   label="true optimum lies in here")
    a.set(xscale="log", xlabel="Tiles imaged per day", ylabel="value delivered over the day",
          title="Whole-day value against a bound on the best\npossible schedule")
    a.legend(fontsize=7, frameon=False)
    a.grid(alpha=0.25)

    b = axes[1]
    b.plot(agg.tiles, 100 * agg.total_gap, "o-", color="#E07A1F", lw=2.2,
           label="total gap (ours vs bound)")
    b.plot(agg.tiles, 100 * agg.price_of_online, "s--", color="#1B6CA8",
           label="price of not seeing the future")
    b.plot(agg.tiles, 100 * agg.fifo_gap, "^-", color="#C0392B", alpha=0.7, label="FIFO gap")
    b.set(xscale="log", xlabel="Tiles imaged per day", ylabel="% below the bound",
          title="How much is still on the table, end to end")
    b.legend(fontsize=7, frameon=False)
    b.grid(alpha=0.25)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "wp5_joint_bound.png", dpi=160)
    print(f"\nSaved wp5_joint_bound.csv|.png in {out}")


if __name__ == "__main__":
    main()
