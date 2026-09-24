"""WP6 step 3: the Contribution-A experiment, driven by REAL detections.

Same five strategies and the same orbit as scripts/simulate_day.py, but the day is
built from the 5,320 held-out Airbus tiles and the trained detector's real output
(sat7.real_workload): real ships per tile, real lengths, real confidences, real
misses, real false alarms, and a per-ship payload cost from the WP4 power law.

Still assumed, and printed on every run: tiles/day, the context mix, which ships are
dark (no AIS in the data), and capture times (the tiles carry no timestamps).

    python scripts/wp6_simulate_real.py
    python scripts/wp6_simulate_real.py --mix dataset --tiles 80000
    python scripts/wp6_simulate_real.py --no-size-model     # ablation: WP4 medians only

Outputs in results/: wp6_real_table.csv, wp6_real_recall.png, wp6_real_sweep.csv/.png,
wp6_thr_sweep.csv/.png, wp6_real_vs_sim.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
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
from sat7.scheduler import (FIFO, LoDConfig, NoBuffer, ValueGreedy, WorkloadConfig,
                            encode_fixed_patch, encode_lod, encode_raw, generate_workload,
                            metrics, ship_ci95, simulate)

COLORS = {"Bent pipe (raw, FIFO)": "#C0392B",
          "Phi-sat-2 style, ocean only": "#B0B5BD", "Phi-sat-2 style, fair (+coastal)": "#6B7280",
          "Ours: LoD + no buffer": "#8DB8E0", "Ours: LoD + FIFO": "#1B6CA8",
          "Ours: LoD + value-greedy": "#E07A1F"}

# (label, encoder, policy, coverage kind). "kind" says which ships the encoder can even
# describe, which is what the onboard ceiling below is computed from.
STRATEGIES = [
    ("Bent pipe (raw, FIFO)", encode_raw, FIFO, "raw"),
    ("Phi-sat-2 style, ocean only", partial(encode_fixed_patch, coastal=False), FIFO, "ocean"),
    ("Phi-sat-2 style, fair (+coastal)", encode_fixed_patch, FIFO, "detected"),
    ("Ours: LoD + no buffer", encode_lod, NoBuffer, "lod"),
    ("Ours: LoD + FIFO", encode_lod, FIFO, "lod"),
    ("Ours: LoD + value-greedy", encode_lod, ValueGreedy, "lod"),
]


def onboard_ceiling(wl, lod: LoDConfig, kind: str) -> dict:
    """The best an infinite downlink could do: what the onboard software still knows.

    A ship is *knowable* only if its tile survived the cloud gate and the encoder produces
    something that covers it:
      raw       every ship on every tile it downlinks;
      ocean     detected ships on open-ocean tiles only (the interim baseline, weakness #2);
      detected  detected ships anywhere, coast included (the fair fixed-patch baseline);
      lod       ours: detected ships anywhere, plus *undetected* ships on coastal tiles,
                because we send those tiles whole.
    Ships below this line are lost before any scheduling happens, so they are a detector
    problem, not a downlink problem. Keeping the two apart is the point: it says which
    work-package to spend the next week on.
    """
    known, dark_known, n, nd = 0, 0, 0, 0
    for t, ctx, ids in wl.tiles:
        for j in ids:
            s = wl.ships[j]
            n += 1
            nd += s.dark
            if ctx == "cloud":
                continue                       # cloud gate drops the tile for every strategy
            fired = s.confidence >= lod.conf_low
            ok = {"raw": True,
                  "ocean": ctx == "ships" and fired,
                  "detected": fired,
                  "lod": ctx == "coast" or fired}[kind]
            known += ok
            dark_known += ok and s.dark
    return {"ceiling_recall": known / n if n else float("nan"),
            "ceiling_dark_recall": dark_known / nd if nd else float("nan")}


def run_all(wl, passes, start, storage, lod) -> pd.DataFrame:
    rows = []
    for label, enc, pol, kind in STRATEGIES:
        m = metrics(simulate(enc(wl, lod), passes, start, pol(), storage, encoder=label), wl, lod)
        m["strategy"] = label
        m |= onboard_ceiling(wl, lod, kind)
        # of everything the onboard software could still have sent, how much got through?
        m["downlink_eff"] = (m["ship_recall"] / m["ceiling_recall"]
                             if m["ceiling_recall"] else float("nan"))
        rows.append(m)
    return pd.DataFrame(rows)


def build_day(tiles, ships, args, seed: int, tiles_per_day: int):
    cfg = RealWorkloadConfig(tiles_per_day=tiles_per_day, mix=args.mix, p_dark=args.p_dark,
                             det_thr=args.det_thr, seed=seed)
    return workload_from_catalogue(tiles, ships, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=40_000, help="ASSUMPTION: tiles imaged per day")
    ap.add_argument("--mix", choices=["orbit", "dataset"], default="orbit")
    ap.add_argument("--p-dark", type=float, default=0.10, help="ASSUMPTION: no AIS in the data")
    ap.add_argument("--det-thr", type=float, default=0.25, help="onboard detection threshold")
    ap.add_argument("--link", choices=sorted(LINK_PRESETS), default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3, help="days resampled, for a spread")
    ap.add_argument("--conf-low", type=float, default=None,
                    help="bottom LoD rung; default = --det-thr (the detection threshold IS the "
                         "bottom of the ladder). Set it to decouple the two.")
    ap.add_argument("--conf-high", type=float, default=None,
                    help="rung above which metadata alone is sent; default = LoDConfig (0.670, "
                         "the calibrated score meaning P=0.9 -- WP2)")
    ap.add_argument("--no-size-model", action="store_true",
                    help="ablation: charge every ship the WP4 median instead of its own length")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--tag", default="", help="suffix for the output files, so a variant run "
                    "(--mix dataset, --no-size-model, ...) does not overwrite the main results")
    args = ap.parse_args()
    out = args.results
    tag = f"_{args.tag}" if args.tag else ""

    tiles, ships = load_catalogue(out)
    sm = None if args.no_size_model else load_size_model(out / "wp6_size_model.json")
    # the onboard detection threshold IS the bottom of the level-of-detail ladder
    lod = LoDConfig(conf_low=args.det_thr if args.conf_low is None else args.conf_low,
                    size_model=sm,
                    **({} if args.conf_high is None else {"conf_high": args.conf_high}))

    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    passes = find_passes(sat, GroundStation(), start, 36, LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    storage = args.storage_gb * 1e9
    cap_mb = sum(p.capacity_bytes for p in passes) / 1e6

    # ---------------------------------------------------------------- main table
    wl, st = build_day(tiles, ships, args, args.seed, args.tiles)
    print(f"REAL workload -- {st.source_tiles} held-out tiles, mix={st.mix}")
    print(f"  day: {st.tiles} tiles, {st.ships} real ships ({st.detected} with conf>={args.det_thr}, "
          f"{st.dark} flagged dark [ASSUMPTION]), {st.false_alarms} real false alarms")
    print(f"  contexts: {st.context_counts}")
    print(f"  payload: {'per-ship WP4 power law' if sm else 'WP4 medians (ablation)'}")
    print(f"  {len(passes)} passes in 36 h, {args.link} x {args.link_share} -> {cap_mb:.0f} MB\n")

    frames = []
    for r in range(args.repeats):
        w, _ = build_day(tiles, ships, args, args.seed + r, args.tiles)
        d = run_all(w, passes, start, storage, lod)
        d["rep"] = r
        frames.append(d)
    reps = pd.concat(frames)
    df = (reps.groupby("strategy", sort=False)
              .agg({c: "mean" for c in reps.columns if c not in ("strategy", "encoder", "policy", "rep")})
              .reset_index())
    spread = reps.groupby("strategy", sort=False).ship_recall.agg(["min", "max"]).reset_index()
    df = df.merge(spread, on="strategy")
    n = st.ships
    df["recall_ci95"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in (ship_ci95(r, n) for r in df.ship_recall)]
    df["recall_spread"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(df["min"], df["max"])]
    # honest reference: the bytes a bent pipe would have to *offer*, not what the link
    # let it squeeze through (its MB_sent is just the capacity, which flatters it)
    raw_offered = float(df.loc[df.strategy == "Bent pipe (raw, FIFO)", "MB_offered"].iloc[0])
    df["data_reduction_x"] = raw_offered / df.MB_sent

    cols = ["strategy", "ship_recall", "recall_ci95", "recall_spread", "ceiling_recall",
            "downlink_eff", "dark_recall", "value_frac", "latency_med_h", "latency_p90_h",
            "dark_latency_med_h", "MB_sent", "MB_offered", "data_reduction_x", "dropped_storage"]
    df[cols].to_csv(out / f"wp6_real_table{tag}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 25, "display.precision", 3):
        print(df[cols].to_string(index=False))

    # ---------------------------------------------------------------- real vs synthetic
    wsyn = generate_workload(WorkloadConfig(tiles_per_day=args.tiles, seed=args.seed))
    dsyn = run_all(wsyn, passes, start, storage, LoDConfig(conf_low=args.det_thr))
    cmp = pd.concat([df.assign(workload="REAL detections (WP6)"),
                     dsyn.assign(workload="SYNTHETIC (WP5)")])
    keep = ["workload", "strategy", "ship_recall", "dark_recall", "latency_med_h", "MB_sent"]
    cmp[keep].to_csv(out / f"wp6_real_vs_sim{tag}.csv", index=False)
    print("\nREAL vs SYNTHETIC workload, same orbit and link:")
    piv = cmp.pivot_table(index="strategy", columns="workload",
                          values=["ship_recall", "MB_sent"], sort=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(piv.to_string())

    # ---------------------------------------------------------------- load sweep
    loads = [5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
    sweep = []
    for tpd in loads:
        w, _ = build_day(tiles, ships, args, args.seed, tpd)
        d = run_all(w, passes, start, storage, lod)
        d["tiles"] = tpd
        sweep.append(d)
    sw = pd.concat(sweep)
    sw.to_csv(out / f"wp6_real_sweep{tag}.csv", index=False)

    # ---------------------------------------------------------------- threshold sweep
    # Only possible with real confidences: where should the onboard cut-off sit?
    thr_rows = []
    for thr in [0.05, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70]:
        a2 = argparse.Namespace(**vars(args))
        a2.det_thr = thr
        w, s2 = build_day(tiles, ships, a2, args.seed, args.tiles)
        l2 = LoDConfig(conf_low=thr, size_model=sm)
        m = metrics(simulate(encode_lod(w, l2), passes, start, ValueGreedy(), storage,
                             encoder="Ours: LoD + value-greedy"), w, l2)
        m["det_thr"] = thr
        m["detected_frac"] = s2.detected / s2.ships
        m["false_alarms"] = s2.false_alarms
        thr_rows.append(m)
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(out / f"wp6_thr_sweep{tag}.csv", index=False)
    print("\nOnboard confidence threshold sweep (ours, value-greedy) -- REAL confidences:")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(thr[["det_thr", "detected_frac", "false_alarms", "ship_recall", "dark_recall",
                   "MB_sent", "latency_med_h"]].to_string(index=False))

    _charts(df, sw, thr, cmp, out, args, st, tag)
    print(f"\nSaved wp6_real_table{tag}.csv, wp6_real_vs_sim{tag}.csv, wp6_real_sweep{tag}.csv, "
          f"wp6_thr_sweep{tag}.csv and 3 charts in {out}")


def _charts(df, sw, thr, cmp, out: Path, args, st, tag: str = "") -> None:
    note = (f"REAL detections on {st.source_tiles} held-out tiles, resampled to "
            f"{args.tiles:,} tiles/day (mix={args.mix}, {args.p_dark:.0%} dark = ASSUMPTION)")

    # chart 1: recall, real workload
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.ship_recall, 0.4, color=[COLORS[s] for s in df.strategy])
    ax.bar(x + 0.2, df.dark_recall, 0.4, color=[COLORS[s] for s in df.strategy],
           hatch="//", edgecolor="white")
    ax.set_xticks(x, [s.replace(": ", ":\n").replace(" (", "\n(") for s in df.strategy], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of real ships delivered")
    ax.set_title("Ships reaching the ground in one day (plain = all, hatched = dark)\n" + note,
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"wp6_real_recall{tag}.png", dpi=160)

    # chart 2: load sweep
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for strat, g in sw.groupby("strategy", sort=False):
        axes[0].plot(g.tiles, g.ship_recall, "o-", color=COLORS[strat], label=strat)
        axes[1].plot(g.tiles, g.dark_recall, "o-", color=COLORS[strat], label=strat)
    for a, t in zip(axes, ["All real ships", "Dark vessels (assumed 10% of them)"]):
        a.set(xscale="log", xlabel="Tiles imaged per day (offered load)")
        a.set_title(t, fontsize=10)
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction delivered")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=8)
    fig.suptitle("When does each strategy break? -- " + note, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / f"wp6_real_sweep{tag}.png", dpi=160)

    # chart 3: threshold sweep + real vs synthetic
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    a = axes[0]
    a.plot(thr.det_thr, thr.ship_recall, "o-", color="#1B6CA8", label="ships delivered")
    a.plot(thr.det_thr, thr.dark_recall, "s-", color="#E07A1F", label="dark vessels delivered")
    a.set(xlabel="Onboard confidence threshold", ylabel="Fraction delivered", ylim=(0, 1.02))
    a2 = a.twinx()
    a2.plot(thr.det_thr, thr.MB_sent, "^--", color="#6B7280", label="MB downlinked")
    a2.set_ylabel("MB downlinked / day")
    a.set_title("Where to put the onboard cut-off\n(REAL confidences)", fontsize=9)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="center right")
    a.grid(alpha=0.25)

    b = axes[1]
    piv = cmp.pivot_table(index="strategy", columns="workload", values="ship_recall", sort=False)
    y = np.arange(len(piv))
    b.barh(y - 0.2, piv.iloc[:, 0], 0.4, color="#1B6CA8", label=piv.columns[0])
    b.barh(y + 0.2, piv.iloc[:, 1], 0.4, color="#9CA3AF", label=piv.columns[1])
    b.set_yticks(y, [s.replace(" (", "\n(") for s in piv.index], fontsize=7)
    b.set_xlabel("Fraction of ships delivered")
    b.set_title("Real detections vs the synthetic workload\nused in the interim report", fontsize=9)
    b.legend(fontsize=7, frameon=False, loc="lower right")
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"wp6_thr_and_compare{tag}.png", dpi=160)


if __name__ == "__main__":
    main()
