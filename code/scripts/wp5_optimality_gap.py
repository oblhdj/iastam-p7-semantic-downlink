"""WP5 deliverable: how far from optimal is the greedy scheduler?

The technical plan lists "optimality gap on small passes" under *must have* and the
technical report states that greedy by value-per-byte is not optimal for indivisible items.
Neither had been measured. This script measures it on the real workload.

Method. Replay a real day. At every contact window, take the queue exactly as the greedy
policy left it, and compare:

    greedy      what ValueGreedy actually sends in that window
    optimum     the exact DP optimum for that same queue and capacity (sat7.optimum)
    bound       a fractional relaxation, which must dominate the optimum (a check, not a claim)

The gap is (optimum - greedy) / optimum, per pass. The trajectory follows the greedy policy,
so this answers "given the state greedy got itself into, how much value did it leave behind
in this window?" -- not the harder joint question of scheduling every pass at once, which is
stated as a limit rather than glossed over.

    python scripts/wp5_optimality_gap.py
    python scripts/wp5_optimality_gap.py --loads 40000 160000 --frac-steps 8

Output: results/wp5_optimality_gap.csv|.png
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

from sat7.optimum import measure_gap, plan_value
from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite
from sat7.real_workload import (RealWorkloadConfig, load_catalogue, load_size_model,
                                workload_from_catalogue)
from sat7.scheduler import FIFO, LoDConfig, ValueGreedy, encode_lod


def small_instance(queue: list, capacity: float, n: int, rng, max_buckets: int,
                   buckets_per_item: int = 2) -> tuple[list, float, float]:
    """A structurally faithful miniature of one contact window that the DP can solve exactly.

    The plan asks for the gap "on small passes", and there is a good reason to look there:
    the DP needs byte buckets fine enough to resolve a 40 B metadata record, which at a
    47 MB pass capacity would take a million states. Subsampling items and scaling the
    capacity by the same byte fraction keeps the size mix, the value mix and the saturation
    level, while shrinking the numbers until the resolution is affordable.

    Returns (items, capacity, bucket_bytes) with the bucket small enough that the *smallest*
    item spans at least ``buckets_per_item`` buckets -- so no item is invisible to the DP.
    Halves the sample until that holds, rather than assuming one sample size fits every pass.
    """
    while True:
        if len(queue) <= n:
            sub, cap = list(queue), capacity
        else:
            pick = rng.choice(len(queue), size=n, replace=False)
            sub = [queue[i] for i in pick]
            cap = capacity * sum(i.size for i in sub) / sum(i.size for i in queue)
        bucket = max(1.0, min(i.size for i in sub) / buckets_per_item)
        if cap / bucket <= max_buckets or n <= 32:
            return sub, cap, bucket
        n //= 2


def gaps_for_day(wl, lod, passes, start, policy, max_buckets: int, frac_steps: int,
                 small_n: int = 400, seed: int = 0, storage_bytes: float = 8e9) -> list[dict]:
    """Walk the day; at each pass compare the policy's plan with the exact optimum."""
    items = sorted(encode_lod(wl, lod), key=lambda it: it.created_s)
    rng = np.random.default_rng(seed)
    rows, queue, k = [], [], 0
    for p in passes:
        rise_s = (p.rise - start).total_seconds()
        while k < len(items) and items[k].created_s <= rise_s:
            queue.append(items[k])
            k += 1
        total = sum(i.size for i in queue)
        if total > storage_bytes:                      # same storage rule as simulate()
            for v in sorted(queue, key=lambda i: i.density):
                if total <= storage_bytes:
                    break
                queue.remove(v)
                total -= v.size
        if queue:
            for scale, n in (("operational", 0), ("small", small_n)):
                if n == 0:
                    q, cap, bucket = queue, p.capacity_bytes, None
                else:
                    q, cap, bucket = small_instance(queue, p.capacity_bytes, n, rng, max_buckets)
                r = measure_gap(q, cap, policy, rise_s, max_buckets, frac_steps,
                                bucket_bytes=bucket)
                smallest = min((i.size for i in q), default=1.0)
                rows.append({"policy": policy.name, "scale": scale, "rise_h": rise_s / 3600,
                             "capacity_MB": cap / 1e6, "queue_items": r.n_items,
                             "queue_MB": sum(i.size for i in q) / 1e6,
                             "greedy": r.greedy_value, "optimum": r.optimal_value,
                             "bound": r.bound, "gap": r.gap, "buckets": r.buckets,
                             "bucket_bytes": r.bucket_bytes,
                             # can the DP even see the smallest item? < 1 means it cannot
                             "resolution": smallest / r.bucket_bytes if r.bucket_bytes else 0.0,
                             "valid": r.valid})
        plan = policy.plan(queue, p.capacity_bytes, rise_s)
        sent = set(id(it) for it, _ in plan)
        queue = [i for i in queue if id(i) not in sent]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loads", type=int, nargs="+", default=[20_000, 40_000, 80_000, 160_000])
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--link", choices=sorted(LINK_PRESETS), default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-buckets", type=int, default=4000,
                    help="byte resolution of the DP (higher = finer, slower)")
    ap.add_argument("--frac-steps", type=int, default=4,
                    help="partial sends a progressive item may choose between")
    ap.add_argument("--small-n", type=int, default=400,
                    help="items in the miniature instance the DP can resolve exactly")
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

    rows = []
    for load in args.loads:
        for rep in range(args.repeats):
            cfg = RealWorkloadConfig(tiles_per_day=load, det_thr=args.det_thr, seed=args.seed + rep)
            wl, _ = workload_from_catalogue(tiles, ships, cfg)
            for policy in (ValueGreedy(), FIFO()):
                for r in gaps_for_day(wl, lod, passes, start, policy, args.max_buckets,
                                      args.frac_steps, args.small_n, args.seed + rep):
                    rows.append(r | {"tiles": load, "rep": rep})
    d = pd.DataFrame(rows)
    d.to_csv(out / "wp5_optimality_gap.csv", index=False)

    bad = (~d.valid).sum()
    print(f"{len(d)} pass-instances, DP resolution {args.max_buckets} buckets, "
          f"{args.frac_steps} truncation steps")
    print(f"sanity (greedy <= optimum <= fractional bound): "
          f"{'all pass' if bad == 0 else f'{bad} VIOLATIONS'}\n")

    d["saturated"] = d.queue_MB > d.capacity_MB
    for scale, gs in d.groupby("scale", sort=False):
        res = gs.resolution.min()
        note = ("the DP resolves every item exactly" if res >= 1 else
                f"smallest item spans {res:.3f} of a byte-bucket, so the DP is coarse here "
                f"and its gap is a LOWER bound")
        print(f"===== {scale} instances ({note}) =====")
        for pol, g in gs.groupby("policy", sort=False):
            t = (g.groupby("tiles")
                   .agg(passes=("gap", "size"), mean_gap=("gap", "mean"),
                        max_gap=("gap", "max"), median_items=("queue_items", "median"),
                        saturated=("saturated", "mean"))
                   .reset_index())
            print(f"--- {pol} ---")
            with pd.option_context("display.precision", 4):
                print(t.to_string(index=False))
        print()

    sm_ = d[(d.scale == "small") & d.saturated]
    vg = sm_[sm_.policy == ValueGreedy().name]
    ff = sm_[sm_.policy == FIFO().name]
    if len(vg):
        print(f"HEADLINE (saturated small passes, where the choice actually binds) -- "
              f"value-greedy loses {100 * vg.gap.mean():.2f}% of the optimal value on "
              f"average, worst case {100 * vg.gap.max():.2f}%, over {len(vg)} windows.")
    if len(ff):
        print(f"FIFO on the same queues: mean {100 * ff.gap.mean():.2f}%, "
              f"worst {100 * ff.gap.max():.2f}%.")
    op = d[(d.scale == "operational") & (d.policy == ValueGreedy().name)]
    print(f"At operational scale ({op.queue_items.median():.0f} items per window, median) the "
          f"gap is {100 * op.gap.mean():.3f}% -- with thousands of small items in one window "
          f"the instance is nearly fractional, and greedy is optimal for fractional knapsack.")

    # ---- chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    a = axes[0]
    sm_all = d[d.scale == "small"]
    for pol, g in sm_all.groupby("policy", sort=False):
        m = g.groupby("tiles").gap.mean()
        e = g.groupby("tiles").gap.max()
        a.plot(m.index, 100 * m.values, "o-", label=f"{pol}, mean")
        a.plot(e.index, 100 * e.values, "s--", alpha=0.6, label=f"{pol}, worst pass")
    a.set(xscale="log", xlabel="Tiles imaged per day",
          ylabel="value lost vs the exact optimum (%)",
          title="Optimality gap per contact window\n(REAL workload, exact DP)")
    a.legend(fontsize=7, frameon=False)
    a.grid(alpha=0.25)
    a.spines[["top", "right"]].set_visible(False)

    b = axes[1]
    b.hist(100 * sm_all[sm_all.policy == ValueGreedy().name].gap, bins=30,
           color="#1B6CA8", label="value-greedy (ours)")
    b.hist(100 * sm_all[sm_all.policy == FIFO().name].gap, bins=30,
           color="#C0392B", alpha=0.55, label="FIFO")
    b.set(xlabel="value lost vs optimum (%)", ylabel="contact windows",
          title="Distribution over every pass measured")
    b.legend(fontsize=8, frameon=False)
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "wp5_optimality_gap.png", dpi=160)
    print(f"\nSaved wp5_optimality_gap.csv|.png in {out}")


if __name__ == "__main__":
    main()
