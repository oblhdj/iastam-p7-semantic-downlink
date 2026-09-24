"""WP10b -- two-dimensional sensitivity: do the invented constants interact?

ARCHITECTURE.md section E item 5: "WP10 was one-at-a-time; `thumb_value` x cloud plainly
interact." One-at-a-time sweeps cannot see an interaction by construction, and WP10 found the
one conclusion that fails does so at `thumb_value = 0.1` -- where thumbnails crowd out ship
chips. Whether that boundary moves with the cloud fraction decides how safe the baseline is,
because cloud is the parameter WP10 ranked most influential (recall range 0.42).

Sweeps the full grid rather than a cross, and reports where "ours >= fair fixed-patch baseline"
actually breaks.

    .venv/Scripts/python.exe scripts/wp10b_interaction.py

Outputs: results/wp10b_interaction.csv, wp10b_interaction.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite
from sat7.real_workload import load_catalogue, load_size_model
from wp10_sensitivity import BASELINE, orbit_mix, run_point

CLOUD = [0.00, 0.15, 0.30, 0.50]
THUMB = [0.001, 0.01, 0.05, 0.10]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loads", type=int, nargs="+", default=[40_000, 160_000])
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    out = args.results

    tiles, ships = load_catalogue(out)
    sm = load_size_model(out / "wp6_size_model.json")
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    passes = find_passes(sat, GroundStation(), start, 36, LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    storage = args.storage_gb * 1e9

    rows = []
    for load in args.loads:
        for cloud in CLOUD:
            for thumb in THUMB:
                p = dict(BASELINE, cloud=cloud, thumb_value=thumb)
                r = run_point(tiles, ships, sm, passes, start, storage, load, p, args.seed)
                rows.append({"load": load, "cloud": cloud, "thumb_value": thumb, **r})
                print(f"  load {load:>7,} cloud {cloud:.2f} thumb {thumb:<6} -> "
                      f"recall {r['ship_recall']:.4f}  vs baseline {r['ours_minus_baseline']:+.4f}",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "wp10b_interaction.csv", index=False)

    # ---- where does "ours >= fair fixed-patch baseline" break, and does cloud move it?
    print("\n'ours - fair baseline', by cloud x thumb_value (negative = we lose):")
    for load in args.loads:
        g = df[df.load == load].pivot(index="cloud", columns="thumb_value",
                                      values="ours_minus_baseline")
        print(f"\n  load {load:,}")
        print(g.round(4).to_string())
        bad = df[(df.load == load) & (df.ours_minus_baseline < 0)]
        if len(bad):
            print(f"  -> LOSES at: " + ", ".join(
                f"cloud {r.cloud:.2f}/thumb {r.thumb_value}" for r in bad.itertuples()))
        else:
            print("  -> holds everywhere on this grid")

    # ---- is the effect additive? interaction = deviation from the one-at-a-time prediction
    print("\nInteraction check (does the thumb_value penalty depend on cloud?):")
    inter = []
    for load in args.loads:
        d = df[df.load == load]
        base = d[(d.cloud == 0.15) & (d.thumb_value == 0.01)].ship_recall.iloc[0]
        for cloud in CLOUD:
            a = d[(d.cloud == cloud) & (d.thumb_value == 0.01)].ship_recall.iloc[0]
            b = d[(d.cloud == cloud) & (d.thumb_value == 0.10)].ship_recall.iloc[0]
            inter.append({"load": load, "cloud": cloud, "thumb_penalty": b - a})
        t = pd.DataFrame([r for r in inter if r["load"] == load])
        spread = t.thumb_penalty.max() - t.thumb_penalty.min()
        print(f"  load {load:,}: thumb_value 0.01->0.10 costs "
              f"{t.thumb_penalty.min():+.4f} to {t.thumb_penalty.max():+.4f} recall "
              f"across cloud; spread {spread:.4f} "
              f"({'INTERACTS' if spread > 0.01 else 'additive, no real interaction'})")
    pd.DataFrame(inter).to_csv(out / "wp10b_thumb_penalty_by_cloud.csv", index=False)

    # ---- figure
    fig, axes = plt.subplots(1, len(args.loads), figsize=(5.6 * len(args.loads), 4.4),
                             squeeze=False)
    for ax, load in zip(axes[0], args.loads):
        g = df[df.load == load].pivot(index="cloud", columns="thumb_value",
                                      values="ours_minus_baseline")
        im = ax.imshow(g.values, cmap="RdYlGn", vmin=-0.02, vmax=0.08, aspect="auto")
        ax.set_xticks(range(len(g.columns)), [str(c) for c in g.columns])
        ax.set_yticks(range(len(g.index)), [f"{c:.0%}" for c in g.index])
        ax.set_xlabel("thumb_value")
        ax.set_ylabel("cloud fraction")
        ax.set_title(f"ours - fair baseline, {load:,} tiles/day", fontsize=10)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g.values[i, j]:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "wp10b_interaction.png", dpi=160)
    print(f"\nSaved wp10b_interaction.csv|.png in {out}")


if __name__ == "__main__":
    main()
