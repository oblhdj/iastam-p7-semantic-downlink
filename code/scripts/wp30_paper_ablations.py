"""WP30 -- the paper's six ablations (section VIII-G) and three trade-off curves (eqs 32-35), one command.

    .venv/Scripts/python.exe scripts/wp30_paper_ablations.py
    .venv/Scripts/python.exe scripts/wp30_paper_ablations.py --rerun-detection   # also re-measures wp26 / wp24

Six ablations, each against the full system on the same input:

    without SAHI              COLLATED  wp26_b0_b4.json            B1 (one call per scene) vs B2, 150 scenes
    without tile overlap      COLLATED  wp24_detection_eval.json   no_overlap vs sahi_B2, 24 swaths
    without detection fusion  COLLATED  wp24_detection_eval.json   sahi_no_fusion vs sahi_B2, 24 swaths
    without adaptive downlink RUN here  every detection at ONE fixed level (P1, P2 or P3)
    without ROI transmission  RUN here  the adaptive level capped at P1 (metadata only)
    without optional relay    RUN here  the full policy, direct path only (sat7.comms)

The three detection rows need the Airbus tiles and the detector, so they are read from the files
their own scripts wrote (REAL); --rerun-detection runs those scripts first. The three downlink rows
and every curve are computed here from the committed catalogue alone -- no dataset, no torch.

Pure P0-P3. The downlink rows use the paper's Table I scheme (sat7.priority) and nothing from the
LoD ladder: no thumbnails, no mosaic, no queue-aware coastal rule. A variant only changes which
level `classify` returns; the encoder, sizes, scheduler, passes and metrics are the full system's.
With every adaptive level capped at P1 and with every detection fixed at P1 the day is the same
(a detection is P1 or it is not sent), so those two rows are equal by construction and say so.

Curves (wp30_curves.csv, one row per point and curve, columns x / y plus every metric):
    D_tx_vs_recall, D_tx_vs_info_preservation   link share swept for each policy; P1 / P2 boundary
                                                swept for the adaptive one; D_tx_vs_recall also
                                                sweeps the detector's cut (det_thr)
    E_total_vs_T_total                          direct vs relay: each policy on both paths, and the
                                                relay's energy weight lam_E swept

Information preservation is sat7.joint's: sum_j w_j * credit(level delivered for j) / sum_j w_j
over the ships the detector fired on, credit P1 0.5 / P2 0.8 / P3 1.0 (ASSUMPTION). A progressive
item counts as delivered once the scheduler sent any admissible part of it, as recall does.

Labels (project rule): detection rows REAL. Downlink rows and curves SIM-over-REAL -- a scheduler
simulation over real detections with modeled crop sizes and measured coastal tiles, recall at the
workload's 15 % cloud fraction. Relay latency SIM, energy ASSUMPTION powers (sat7.comms).
E_total = E_proc + E_comm; E_proc is the same for every downlink variant (same detector, same
tiles), so differences in E_total are differences in E_comm.

Outputs in results/: wp30_ablations.json, wp30_curves.csv, wp30_curves.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "--root" in sys.argv:                                   # run a copy kept outside the repo
    ROOT = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sat7.priority as priority
from sat7.comms import CommsConfig, Links, build_links, simulate_comms
from sat7.energy import EnergyModel
from sat7.joint import INFO_CREDIT
from sat7.priority import P0, P1, P2, P3, PriorityConfig, encode_priority
from sat7.real_workload import (RealWorkloadConfig, attach_coast_sizes, load_size_model,
                                workload_from_catalogue)
from sat7.scheduler import LoDConfig, ValueGreedy, _w, metrics

SIM = "SIM-over-REAL"
KIND_LEVEL = {"P1": P1, "P2": P2, "P3": P3}
_adaptive = priority.classify                              # the paper's adaptive rule, untouched


# ----------------------------------------------------------------------------- the policies
def _fixed(level):
    def f(ship, ctx, cfg, lod):
        return P0 if ship.confidence < lod.conf_low else level
    return f


def _capped(level):
    def f(ship, ctx, cfg, lod):
        return min(level, _adaptive(ship, ctx, cfg, lod))
    return f


# name -> (classify rule, what it is). "full" is the reference every downlink row is compared to.
POLICIES = {
    "full": (_adaptive, "adaptive P0-P3 (paper Table I)"),
    "no_roi": (_capped(P1), "adaptive level capped at P1: metadata only, no ROI, no context"),
    "fixed_P1": (_fixed(P1), "no adaptation: every detection at P1"),
    "fixed_P2": (_fixed(P2), "no adaptation: every detection at P2 (metadata + ROI)"),
    "fixed_P3": (_fixed(P3), "no adaptation: every detection at P3 (metadata + ROI + context)"),
}


@contextmanager
def classify_as(rule, assigned: dict):
    """Run sat7.priority's encoder with another level rule, recording the level each ship got."""
    def wrapped(ship, ctx, cfg, lod):
        level = rule(ship, ctx, cfg, lod)
        assigned[ship.id] = level
        return level
    priority.classify = wrapped
    try:
        yield
    finally:
        priority.classify = _adaptive


def encode(wl, lod, pcfg, policy: str):
    assigned: dict[int, int] = {}
    with classify_as(POLICIES[policy][0], assigned):
        items = encode_priority(wl, lod, pcfg)
    return items, assigned


# ----------------------------------------------------------------------------- metrics
def info_preservation(sent, items, assigned, wl, lod) -> float:
    """sat7.joint's InformationPreservation, on what reached the ground."""
    offered, got, top = {}, {}, {}
    for it in items:
        for i in it.ships:
            offered[i] = offered.get(i, 0) + 1
    for s in sent:
        for i in s.item.ships:
            got[i] = got.get(i, 0) + 1
            top[i] = max(top.get(i, P0), KIND_LEVEL[s.item.kind])
    num = den = 0.0
    for i, level in assigned.items():                      # every ship on a tile the encoder saw
        if wl.ships[i].confidence < lod.conf_low:
            continue                                       # never fired: not a downlink decision
        w = _w(wl.ships[i], lod)
        den += w
        if i in got:                                       # all of its items arrived: the assigned
            num += w * INFO_CREDIT[level if got[i] == offered[i] else top[i]]   # level, else the best
    return num / den if den else float("nan")


def evaluate(items, assigned, wl, lod, links, start, ccfg, e_proc_J, storage) -> dict:
    cr = simulate_comms(items, links, start, ValueGreedy(), ccfg, storage)
    m, t = metrics(cr.result, wl, lod), cr.totals()
    lat = np.array([d.latency_s for d in cr.deliveries]) / 3600.0
    hist = {f"P{k}": sum(1 for v in assigned.values() if v == k) for k in (P0, P1, P2, P3)}
    return {
        "ship_recall": round(float(m["ship_recall"]), 4),
        "dark_recall": round(float(m["dark_recall"]), 4),
        "info_preservation": round(info_preservation(cr.result.sent, items, assigned, wl, lod), 4),
        "D_tx_MB": round(float(m["MB_sent"]), 2),
        "MB_offered": round(float(m["MB_offered"]), 2),
        "E_comm_kJ": round(t["energy_J"] / 1e3, 3),
        "E_total_kJ": round((e_proc_J + t["energy_J"]) / 1e3, 3),
        "T_ship_med_h": round(float(m["latency_med_h"]), 2),
        "T_item_med_h": round(float(np.median(lat)), 2) if lat.size else None,
        "T_item_p90_h": round(float(np.percentile(lat, 90)), 2) if lat.size else None,
        "T_item_max_h": round(float(lat.max()), 2) if lat.size else None,
        "relay_fraction_items": t["relay_fraction_items"],
        "levels": hist,
    }


def delta(row: dict, ref: dict) -> dict:
    keys = ("ship_recall", "info_preservation", "D_tx_MB", "E_comm_kJ", "T_ship_med_h", "T_item_max_h")
    return {k: round(row[k] - ref[k], 4) for k in keys}


# ----------------------------------------------------------------------------- detection rows
def detection_rows(res_dir: Path) -> dict:
    """The three perception ablations, read from the files wp26 / wp24 wrote. Nothing is typed in."""
    wp26 = json.loads((res_dir / "wp26_b0_b4.json").read_text())
    wp24 = json.loads((res_dir / "wp24_detection_eval.json").read_text())
    q, sw = wp26["detection_quality_same_scenes"], wp24["swaths"]["rows"]

    def pick(r):
        return {k: r[k] for k in ("n_gt", "precision", "recall", "F1", "n_detections_supplied")} | {
            "recall_small": r["recall_by_size"]["small_<32"]["recall"]} if "recall_by_size" in r else {
            k: r[k] for k in ("n_gt", "precision", "recall", "F1", "n_detections_supplied")}

    def row(name, full, without, source, inp, note):
        f, w = pick(full), pick(without)
        return {"ablation": name, "label": "REAL", "mode": "collated", "source": source, "input": inp,
                "full": f, "without": w,
                "delta": {k: round(w[k] - f[k], 4) for k in ("precision", "recall", "F1")}, "note": note}

    scenes = f"{wp26['inputs']['scenes']} scenes of {wp26['inputs']['per_side']}x{wp26['inputs']['per_side']} test tiles"
    swaths = f"{wp24['swaths']['n_swaths']} swaths, {wp24['swaths']['n_ships']} ships"
    return {
        "without_sahi": row("without SAHI", q["B2"], q["B1"], "wp26_b0_b4.json: B2 vs B1", scenes,
                            "B1 = one detector call on the whole scene, shrunk to the detector's input"),
        "without_tile_overlap": row("without tile overlap", sw["sahi_B2"], sw["no_overlap"],
                                    "wp24_detection_eval.json: swaths.sahi_B2 vs no_overlap", swaths,
                                    "same windows and fusion, overlap set to zero"),
        "without_detection_fusion": row("without detection fusion", sw["sahi_B2"], sw["sahi_no_fusion"],
                                        "wp24_detection_eval.json: swaths.sahi_B2 vs sahi_no_fusion", swaths,
                                        "same overlapping windows, duplicates across windows not merged"),
    }


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="the code/ directory (default: this script's)")
    ap.add_argument("--day-tiles", type=int, default=40_000)
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rerun-detection", action="store_true",
                    help="run wp26_b0_b4.py and wp24_detection_eval.py first (needs the dataset and weights)")
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/results")
    args = ap.parse_args()
    res_dir, t0 = ROOT / "results", time.perf_counter()
    out = args.out or res_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.rerun_detection:
        for script in ("wp26_b0_b4.py", "wp24_detection_eval.py"):
            print(f"--- re-measuring: {script}")
            subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / script)], check=True)

    # ---- one day, one contact plan: wp23's (so "full" must reproduce its P0-P3 row) -------------
    tiles = pd.read_csv(res_dir / "wp6_tiles.csv")
    attach_coast_sizes(tiles, res_dir)
    ships = pd.read_csv(res_dir / "wp6_ships.csv")
    wl, stats = workload_from_catalogue(tiles, ships, RealWorkloadConfig(
        tiles_per_day=args.day_tiles, det_thr=args.det_thr, seed=args.seed))
    lod = LoDConfig(conf_low=args.det_thr, size_model=load_size_model(res_dir / "wp6_size_model.json"))
    pcfg = PriorityConfig(p1_conf=lod.conf_high)
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    direct = CommsConfig(link_share=args.link_share, relay_enabled=False)
    relay = CommsConfig(link_share=args.link_share, relay_enabled=True)
    links = build_links(relay, start, 36.0)
    storage = args.storage_gb * 1e9
    e_tile = EnergyModel.from_results(res_dir).e_proc_per_tile_J("cpu_onnx")["E_prop"]
    e_proc_J = e_tile * stats.tiles

    def share_links(s: float) -> Links:
        k = s / args.link_share
        return Links([replace(p, capacity_bytes=p.capacity_bytes * k) for p in links.primary_passes],
                     links.isl, links.relay_passes, primary_share=s, relay_share=links.relay_share)

    def run(policy, ccfg=direct, lk=links, cfg=pcfg):
        items, assigned = encode(wl, lod, cfg, policy)
        return evaluate(items, assigned, wl, lod, lk, start, ccfg, e_proc_J, storage)

    print(f"=== WP30 paper ablations [{SIM}] ===")
    print(f"day: {stats.tiles} tiles, {stats.ships} ships ({stats.detected} detected), "
          f"{len(links.primary_passes)} passes, link share {args.link_share}\n")

    base = {name: run(name) for name in POLICIES}                    # direct path, nominal link
    with_relay = {name: run(name, relay) for name in POLICIES}
    full, full_relay = base["full"], with_relay["full"]

    # ---- gates: nothing below is trusted unless these hold --------------------------------------
    wp23 = json.loads((res_dir / "wp23_semantic_compare.json").read_text())["semantic_schemes"]["priority"]
    gates = {
        "full_reproduces_wp23_P0P3_row": {
            "wp23": {"ship_recall": wp23["ship_recall"], "MB_sent": wp23["MB_sent"]},
            "here": {"ship_recall": full["ship_recall"], "MB_sent": full["D_tx_MB"]},
            "passed": abs(full["ship_recall"] - wp23["ship_recall"]) < 5e-4
                      and abs(full["D_tx_MB"] - wp23["MB_sent"]) < 0.1},
        "cap_at_P1_equals_fixed_P1": {"passed": base["no_roi"] == base["fixed_P1"]},
        "relay_is_a_reroute": {
            "passed": full_relay["ship_recall"] == full["ship_recall"]
                      and abs(full_relay["D_tx_MB"] - full["D_tx_MB"]) < 0.5},
        "no_ROI_sends_only_P1": {"passed": base["no_roi"]["levels"]["P2"] == 0
                                           and base["no_roi"]["levels"]["P3"] == 0},
    }
    ok = all(g["passed"] for g in gates.values())

    downlink = {
        "without_adaptive_semantic_downlink": {
            "ablation": "without adaptive semantic downlink", "label": SIM, "mode": "run",
            "definition": "every detection that clears the onboard cut is sent at one fixed level",
            "full": full, "without": {k: base[k] for k in ("fixed_P1", "fixed_P2", "fixed_P3")},
            "delta": {k: delta(base[k], full) for k in ("fixed_P1", "fixed_P2", "fixed_P3")}},
        "without_roi_transmission": {
            "ablation": "without ROI transmission", "label": SIM, "mode": "run",
            "definition": "adaptive levels capped at P1: metadata only",
            "full": full, "without": base["no_roi"], "delta": delta(base["no_roi"], full),
            "note": "equal to fixed_P1 by construction (gate cap_at_P1_equals_fixed_P1)"},
        "without_optional_relay": {
            "ablation": "without optional relay", "label": "latency SIM; energy ASSUMPTION powers",
            "mode": "run", "definition": "the full adaptive policy; relay path disabled in sat7.comms",
            "full": full_relay, "without": full, "delta": delta(full, full_relay)},
    }

    # ---- curves ---------------------------------------------------------------------------------
    rows = []

    def add(curves, series, knob, value, path, r):
        xy = {"D_tx_vs_recall": ("D_tx_MB", "ship_recall"),
              "D_tx_vs_info_preservation": ("D_tx_MB", "info_preservation"),
              "E_total_vs_T_total": ("T_item_med_h", "E_total_kJ")}
        for c in curves:
            x, y = xy[c]
            rows.append({"curve": c, "series": series, "knob": knob, "knob_value": value, "path": path,
                         "x_name": x, "x": r[x], "y_name": y, "y": r[y],
                         **{k: v for k, v in r.items() if k != "levels"}, "label": SIM})

    d_curves = ("D_tx_vs_recall", "D_tx_vs_info_preservation")
    for name in POLICIES:
        if name == "no_roi":
            continue                                                 # the same day as fixed_P1
        for s in (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, args.link_share):
            r = base[name] if s == args.link_share else run(name, replace(direct, link_share=s), share_links(s))
            add(d_curves, name, "link_share", s, "direct", r)
    for p1 in (0.25, 0.40, 0.55, 0.670, 0.80, 0.90, 1.01):           # 0.25: all P1 off-coast; 1.01: none
        add(d_curves, "full", "p1_conf", p1, "direct", run("full", cfg=replace(pcfg, p1_conf=p1)))
    for thr in (0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.670, 0.80):      # the detector's own cut: what
        wl_t, _ = workload_from_catalogue(tiles, ships, RealWorkloadConfig(   # actually moves recall
            tiles_per_day=args.day_tiles, det_thr=thr, seed=args.seed))
        lod_t = replace(lod, conf_low=thr)
        items, assigned = encode(wl_t, lod_t, replace(pcfg, p1_conf=max(thr, pcfg.p1_conf)), "full")
        add(("D_tx_vs_recall",), "full", "det_thr", thr, "direct",
            evaluate(items, assigned, wl_t, lod_t, links, start, direct, e_proc_J, storage))
    for name in POLICIES:
        if name == "no_roi":
            continue
        add(("E_total_vs_T_total",), name, "path", "direct", "direct", base[name])
        add(("E_total_vs_T_total",), name, "path", "relay", "relay", with_relay[name])
    for lam_e in (0.1, 1.0, 10.0, 100.0):
        add(("E_total_vs_T_total",), "full", "lam_E_per_J", lam_e, "relay",
            run("full", replace(relay, lam_E=lam_e)))
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "wp30_curves.csv", index=False)

    # ---- figure ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, c in zip(axes, ("D_tx_vs_recall", "D_tx_vs_info_preservation", "E_total_vs_T_total")):
        sub = curves[curves.curve == c]
        for (series, knob), g in sub.groupby(["series", "knob"], sort=False):
            if c == "E_total_vs_T_total" and knob == "path":
                ax.plot(g.x, g.y, "-", lw=1, color="0.6")
                for _, pt in g.iterrows():
                    ax.scatter(pt.x, pt.y, marker="o" if pt.path == "direct" else "^", s=45, zorder=3,
                               label=f"{series} ({pt.path})")
            else:
                ax.plot(g.x, g.y, {"link_share": "o-", "det_thr": "d:"}.get(knob, "s--"), ms=4, lw=1.2,
                        label=f"{series} ({knob})")
        ax.set_xlabel(sub.x_name.iloc[0].replace("_", " "))
        ax.set_ylabel(sub.y_name.iloc[0].replace("_", " "))
        ax.set_title(c.replace("_", " "))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    fig.suptitle(f"WP30 trade-off curves [{SIM}; relay energy ASSUMPTION powers]", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "wp30_curves.png", dpi=140)

    report = {
        "label": f"detection rows REAL (collated); downlink rows and curves {SIM} (pure P0-P3, modeled "
                 "crop sizes, measured coastal tiles); relay latency SIM, energy ASSUMPTION powers. "
                 "Recall is at the workload's 15 % cloud fraction -- never quote it bare.",
        "workload": {"tiles": stats.tiles, "ships": stats.ships, "detected": stats.detected,
                     "passes": len(links.primary_passes), "link_share": args.link_share,
                     "det_thr": args.det_thr, "p1_conf": pcfg.p1_conf, "seed": args.seed,
                     "E_proc_kJ_per_day": round(e_proc_J / 1e3, 2),
                     "E_proc_note": "cpu_onnx build, ASSUMPTION powers (sat7.energy); equal in every "
                                    "downlink variant"},
        "policies": {k: v[1] for k, v in POLICIES.items()},
        "info_preservation": {"definition": "sum w*credit(delivered level) / sum w over detected ships",
                              "credit_ASSUMPTION": {f"P{k}": v for k, v in INFO_CREDIT.items()}},
        "gates": gates, "gates_passed": ok,
        "ablations": detection_rows(res_dir) | downlink,
        "all_policies_direct": base, "all_policies_with_relay": with_relay,
        "curves": {"file": "wp30_curves.csv", "rows": len(curves),
                   "points_per_curve": curves.groupby("curve").size().to_dict()},
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out / "wp30_ablations.json").write_text(json.dumps(report, indent=2, default=float))

    # ---- print ----------------------------------------------------------------------------------
    print(f"{'detection ablation':28s} {'recall':>15s} {'precision':>15s}  source")
    for r in list(report["ablations"].values())[:3]:
        print(f"{r['ablation']:28s} {r['full']['recall']:.3f} -> {r['without']['recall']:.3f} "
              f"{r['full']['precision']:.3f} -> {r['without']['precision']:.3f}  {r['source']}")
    print(f"\n{'downlink policy':10s} {'recall':>7s} {'info':>6s} {'D_tx MB':>8s} {'E_comm kJ':>10s} "
          f"{'T_ship h':>9s} {'T_max h':>8s}   levels")
    for name, r in base.items():
        print(f"{name:10s} {r['ship_recall']:7.4f} {r['info_preservation']:6.3f} {r['D_tx_MB']:8.2f} "
              f"{r['E_comm_kJ']:10.3f} {r['T_ship_med_h']:9.2f} {r['T_item_max_h']:8.2f}   {r['levels']}")
    r = full_relay
    print(f"{'full+relay':10s} {r['ship_recall']:7.4f} {r['info_preservation']:6.3f} {r['D_tx_MB']:8.2f} "
          f"{r['E_comm_kJ']:10.3f} {r['T_ship_med_h']:9.2f} {r['T_item_max_h']:8.2f}   "
          f"relay carried {100 * r['relay_fraction_items']:.1f} % of items")
    for name, g in gates.items():
        print(f"gate {name}: {'ok' if g['passed'] else 'FAILED'}")
    print(f"\nSaved wp30_ablations.json, wp30_curves.csv ({len(curves)} rows), wp30_curves.png in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
