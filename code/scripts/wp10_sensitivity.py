"""WP10: which of our results depend on numbers we invented?

Several constants in the value model were never measured and never swept:

    dark_weight      5.0    a dark vessel is worth five ordinary ships
    dark_wake_value  1.0    the ROI+wake follow-up is worth one ordinary ship (added in WP9)
    thumb_value      0.01   a thumbnail is worth 1% of a ship
    p_dark           0.10   one ship in ten has no AIS match  (no AIS in the Airbus data)
    cloud fraction   0.15   of tiles are unusable            (the dataset has 0.4%)
    conf_high        0.9    the rung above which metadata alone is sent
    aging_per_hour   0.5    how fast a waiting item gains priority

Every headline in this project is computed on top of them. This sweeps each one alone,
around the baseline, and asks two different questions:

  1. how far does each outcome move?                      -> a tornado chart
  2. does any CONCLUSION flip?                            -> the comparative claims

The second matters more. A number that shifts is fine; a claim that reverses is not. The
claims checked are "value-greedy beats FIFO on our own payload" and "we beat a fair
fixed-patch baseline", at a load where the link actually binds.

    .venv/Scripts/python.exe scripts/wp10_sensitivity.py
    .venv/Scripts/python.exe scripts/wp10_sensitivity.py --loads 160000

Output: results/wp10_sensitivity.csv, wp10_tornado.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace as dcreplace
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
from sat7.scheduler import (FIFO, LoDConfig, ValueGreedy, encode_fixed_patch, encode_lod,
                            metrics, simulate)

BASELINE = {"dark_weight": 5.0, "dark_wake_value": 1.0, "thumb_value": 0.01,
            "p_dark": 0.10, "cloud": 0.15, "conf_high": 0.670, "aging": 0.5,
            "conf_low": 0.25}

SWEEPS = {
    "dark_weight":     [1.0, 2.0, 5.0, 10.0, 20.0],
    "dark_wake_value": [0.0, 0.5, 1.0, 2.0, 5.0],
    "thumb_value":     [0.001, 0.005, 0.01, 0.05, 0.1],
    "p_dark":          [0.02, 0.05, 0.10, 0.20, 0.40],
    "cloud":           [0.00, 0.05, 0.15, 0.30, 0.50],
    "conf_high":       [0.6, 0.670, 0.8, 0.9, 0.99],
    # conf_low is the bottom rung: below it the encoder sends nothing but the thumbnail.
    # Production ties it to the 0.25 detection threshold, NOT the 0.4 dataclass default, so
    # the calibrated "P=0.4" score of 0.329 would make it STRICTER, not more permissive.
    "conf_low":        [0.05, 0.10, 0.25, 0.329, 0.40],
    "aging":           [0.0, 0.25, 0.5, 1.0, 2.0],
}


def orbit_mix(cloud: float) -> tuple[float, float, float, float]:
    """Keep ships/coast/empty in the baseline proportion, give ``cloud`` the rest."""
    ships, coast, empty = 0.20, 0.05, 0.60
    scale = (1.0 - cloud) / (ships + coast + empty)
    return (cloud, ships * scale, coast * scale, empty * scale)


def run_point(tiles, ships, sm, passes, start, storage, load, p: dict, seed: int) -> dict:
    cfg = RealWorkloadConfig(tiles_per_day=load, det_thr=0.25, seed=seed,
                             p_dark=p["p_dark"], orbit_mix=orbit_mix(p["cloud"]))
    wl, st = workload_from_catalogue(tiles, ships, cfg)
    lod = LoDConfig(conf_low=p["conf_low"], size_model=sm, dark_weight=p["dark_weight"],
                    dark_wake_value=p["dark_wake_value"], thumb_value=p["thumb_value"],
                    conf_high=p["conf_high"])
    items = encode_lod(wl, lod)
    greedy = metrics(simulate(items, passes, start, ValueGreedy(p["aging"]), storage), wl, lod)
    fifo = metrics(simulate(items, passes, start, FIFO(), storage), wl, lod)
    base = metrics(simulate(encode_fixed_patch(wl, lod), passes, start, FIFO(), storage), wl, lod)
    return {"ship_recall": greedy["ship_recall"], "dark_recall": greedy["dark_recall"],
            "MB_offered": greedy["MB_offered"], "latency_med_h": greedy["latency_med_h"],
            "greedy_minus_fifo": greedy["ship_recall"] - fifo["ship_recall"],
            "ours_minus_baseline": greedy["ship_recall"] - base["ship_recall"],
            "n_dark": st.dark}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loads", type=int, nargs="+", default=[40_000, 160_000])
    ap.add_argument("--link", choices=sorted(LINK_PRESETS), default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    out = args.results

    tiles, ships = load_catalogue(out)
    sm = load_size_model(out / "wp6_size_model.json")
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    passes = find_passes(make_satellite(OrbitConfig(epoch=start)), GroundStation(), start, 36,
                         LINK_PRESETS[args.link])
    passes = [dcreplace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    storage = args.storage_gb * 1e9

    rows = []
    for load in args.loads:
        base_row = run_point(tiles, ships, sm, passes, start, storage, load, BASELINE, args.seed)
        rows.append({"load": load, "param": "(baseline)", "value": np.nan, **base_row})
        print(f"--- {load:,} tiles/day, baseline: recall {base_row['ship_recall']:.3f}, "
              f"dark {base_row['dark_recall']:.3f}, greedy-FIFO {base_row['greedy_minus_fifo']:+.3f}")
        for name, values in SWEEPS.items():
            for v in values:
                p = dict(BASELINE, **{name: v})
                r = run_point(tiles, ships, sm, passes, start, storage, load, p, args.seed)
                rows.append({"load": load, "param": name, "value": v, **r})
            print(f"    swept {name}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "wp10_sensitivity.csv", index=False)

    # ---------------------------------------------------------------- do conclusions hold?
    sw = d[d.param != "(baseline)"]
    print("\n=== Do the comparative claims survive every setting? ===")
    bad_g = sw[sw.greedy_minus_fifo < 0]
    bad_b = sw[sw.ours_minus_baseline < 0]
    print(f"'value-greedy >= FIFO'      violated in {len(bad_g)} of {len(sw)} settings")
    if len(bad_g):
        print(bad_g[["load", "param", "value", "greedy_minus_fifo"]].to_string(index=False))
    print(f"'ours >= fair fixed-patch'  violated in {len(bad_b)} of {len(sw)} settings")
    if len(bad_b):
        with pd.option_context("display.precision", 3):
            print(bad_b[["load", "param", "value", "ours_minus_baseline"]].to_string(index=False))

    # ---------------------------------------------------------------- tornado
    print("\n=== How far each assumption moves the outcome (range over its sweep) ===")
    for load in args.loads:
        g = sw[sw.load == load]
        rng = (g.groupby("param")
                .agg(recall_lo=("ship_recall", "min"), recall_hi=("ship_recall", "max"),
                     dark_lo=("dark_recall", "min"), dark_hi=("dark_recall", "max"),
                     MB_lo=("MB_offered", "min"), MB_hi=("MB_offered", "max")))
        rng["recall_range"] = rng.recall_hi - rng.recall_lo
        rng["dark_range"] = rng.dark_hi - rng.dark_lo
        rng = rng.sort_values("recall_range", ascending=False)
        print(f"\n{load:,} tiles/day:")
        with pd.option_context("display.precision", 4):
            print(rng[["recall_lo", "recall_hi", "recall_range", "dark_range",
                       "MB_lo", "MB_hi"]].to_string())

    fig, axes = plt.subplots(1, len(args.loads), figsize=(6.2 * len(args.loads), 4.6), squeeze=False)
    for ax, load in zip(axes[0], args.loads):
        g = sw[sw.load == load]
        base = d[(d.load == load) & (d.param == "(baseline)")].ship_recall.iloc[0]
        r = (g.groupby("param").ship_recall.agg(["min", "max"]))
        r["range"] = r["max"] - r["min"]
        r = r.sort_values("range")
        y = np.arange(len(r))
        ax.barh(y, r["max"] - r["min"], left=r["min"], color="#1B6CA8", alpha=0.85)
        ax.axvline(base, color="#C0392B", ls="--", lw=1.5, label=f"baseline {base:.3f}")
        ax.set_yticks(y, r.index, fontsize=8)
        ax.set(xlabel="ships delivered", title=f"{load:,} tiles/day")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("WP10 — how much each invented constant moves the result "
                 "(one at a time, REAL workload)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "wp10_tornado.png", dpi=160)
    print(f"\nSaved wp10_sensitivity.csv, wp10_tornado.png in {out}")


if __name__ == "__main__":
    main()
