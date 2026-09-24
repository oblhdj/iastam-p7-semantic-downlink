"""WP14 -- re-key the thumbnail gate on the learned gate instead of the classic pre-filter.

WP7 skips the safety-net thumbnail when the onboard software is confident a tile is empty.
That decision currently needs two opinions to agree: the **classic** pre-filter must report
``empty_sea`` *and* the detector must have fired nothing. The classic half of that test was
measured at **0.645** ship recall (WP3) and costs **56 ms/tile** (WP11); the learned gate reaches
**0.988** at 0.15 ms. So the gating decision is being made by the weakest, most expensive
component in the pipeline.

This swaps it and measures what changes. Thumbnails are ~24% of the byte budget (WP7), so a gate
that can call more tiles empty -- safely -- is worth real bytes.

The comparison is like-for-like: same catalogue, same day, same seed, same everything except the
signal driving ``gate_empty``.

    .venv/Scripts/python.exe scripts/wp14_gate_signal.py

Needs `results/wp3_gate_scores_test.csv` (from scripts/wp3_score_tiles.py, which needs torch).
Outputs: results/wp14_gate_signal.json, wp14_gate_signal.csv
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

import numpy as np
import pandas as pd

from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite
from sat7.real_workload import (GATE_EMPTY_THR, RealWorkloadConfig, load_catalogue,
                                load_size_model, workload_from_catalogue)
from sat7.scheduler import LoDConfig, ValueGreedy, encode_lod, metrics, simulate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loads", type=int, nargs="+", default=[40_000, 160_000])
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    out = args.results

    tiles, ships = load_catalogue(out)
    scores_path = out / "wp3_gate_scores_test.csv"
    if not scores_path.exists():
        raise SystemExit(f"{scores_path.name} missing -- run scripts/wp3_score_tiles.py first")
    gs = pd.read_csv(scores_path)
    tiles_gate = tiles.merge(gs, on="image", how="left")
    tiles_gate.attrs = tiles.attrs                       # keep the false-alarm flags
    missing = int(tiles_gate.gate_score.isna().sum())
    if missing:
        raise SystemExit(f"{missing} tiles have no gate score")

    # how differently do the two signals call "empty"?
    classic_empty = (tiles.context == "empty_sea").to_numpy()
    gate_empty = (tiles_gate.gate_score < GATE_EMPTY_THR).to_numpy()
    truly_empty = (tiles.n_ships == 0).to_numpy()
    print("Which tiles does each signal call empty? (before the detector's opinion is added)\n")
    print(f"{'signal':>10s} {'calls empty':>12s} {'of them truly empty':>21s} "
          f"{'ships at risk':>14s}")
    for name, m in (("classic", classic_empty), ("learned gate", gate_empty)):
        at_risk = int(tiles.loc[m & ~truly_empty, "n_ships"].sum())
        prec = (m & truly_empty).sum() / max(1, m.sum())
        print(f"{name:>10s} {m.sum():12d} {prec:20.1%} {at_risk:14d}")
    print(f"\n  threshold used for the gate: {GATE_EMPTY_THR} (chosen on val, WP3)\n")

    sm = load_size_model(out / "wp6_size_model.json")
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    passes = find_passes(sat, GroundStation(), start, 36, LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    storage = args.storage_gb * 1e9
    lod = LoDConfig(conf_low=args.det_thr, size_model=sm)

    rows = []
    for load in args.loads:
        for seed in args.seeds:
            for label, tl in (("classic", tiles), ("learned gate", tiles_gate)):
                cfg = RealWorkloadConfig(tiles_per_day=load, det_thr=args.det_thr, seed=seed)
                wl, _ = workload_from_catalogue(tl, ships, cfg)
                m = metrics(simulate(encode_lod(wl, lod), passes, start, ValueGreedy(), storage),
                            wl, lod)
                rows.append({"load": load, "seed": seed, "signal": label,
                             "ship_recall": m["ship_recall"], "dark_recall": m["dark_recall"],
                             "MB_offered": m["MB_offered"], "MB_sent": m["MB_sent"],
                             "latency_med_h": m["latency_med_h"],
                             "gated": int(sum(wl.gate_empty))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "wp14_gate_signal.csv", index=False)

    print("Averaged over seeds", args.seeds, ":\n")
    print(f"{'load':>8s} {'signal':>13s} {'thumbs skipped':>15s} {'MB offered':>11s} "
          f"{'recall':>8s} {'latency h':>10s}")
    g = df.groupby(["load", "signal"]).mean(numeric_only=True).reset_index()
    for load in args.loads:
        for label in ("classic", "learned gate"):
            r = g[(g.load == load) & (g.signal == label)].iloc[0]
            print(f"{load:8,d} {label:>13s} {r.gated:15,.0f} {r.MB_offered:11.2f} "
                  f"{r.ship_recall:8.4f} {r.latency_med_h:10.2f}")
        a = g[(g.load == load) & (g.signal == "classic")].iloc[0]
        b = g[(g.load == load) & (g.signal == "learned gate")].iloc[0]
        print(f"{'':8s} {'-> change':>13s} {b.gated-a.gated:+15,.0f} "
              f"{b.MB_offered-a.MB_offered:+11.2f} {b.ship_recall-a.ship_recall:+8.4f} "
              f"{b.latency_med_h-a.latency_med_h:+10.2f}\n")

    report = {"gate_threshold": GATE_EMPTY_THR,
              "tiles_called_empty": {"classic": int(classic_empty.sum()),
                                     "learned_gate": int(gate_empty.sum())},
              "ships_at_risk": {
                  "classic": int(tiles.loc[classic_empty & ~truly_empty, "n_ships"].sum()),
                  "learned_gate": int(tiles.loc[gate_empty & ~truly_empty, "n_ships"].sum())},
              "by_load": {}}
    for load in args.loads:
        a = g[(g.load == load) & (g.signal == "classic")].iloc[0]
        b = g[(g.load == load) & (g.signal == "learned gate")].iloc[0]
        report["by_load"][str(load)] = {
            "classic": {k: float(a[k]) for k in ("ship_recall", "MB_offered", "latency_med_h")},
            "learned_gate": {k: float(b[k]) for k in ("ship_recall", "MB_offered", "latency_med_h")},
            "delta_recall": float(b.ship_recall - a.ship_recall),
            "delta_MB": float(b.MB_offered - a.MB_offered),
            "bytes_saved_pct": float(100 * (1 - b.MB_offered / a.MB_offered))}
    (out / "wp14_gate_signal.json").write_text(json.dumps(report, indent=2))
    print(f"Saved wp14_gate_signal.json|.csv in {out}")


if __name__ == "__main__":
    main()
