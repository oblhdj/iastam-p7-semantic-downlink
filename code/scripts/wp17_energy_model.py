"""WP17 -- the per-stage energy model the accepted paper promised (B-progression E_proc/E_comm).

START_HERE.md section 5 item 2 lists a per-stage energy model as unbuilt: E_proc = sum(P_k * T_k)
across preprocess / tile / detect / fuse / semantic, plus E_comm = P_tx * D_tx / R_tx, and the
energy-saving ratio ES = 1 - E_prop / E_base. This builds it. No detector is run here and no torch
is imported: every per-stage TIME is reused from an earlier REAL measurement, and the downlink
side reuses the existing SGP4 orbit/link model -- this script only does the arithmetic.

What is REAL vs invented (the project rule: every invented constant is swept, cf. report 11):
  * T_k  (stage times)      REAL -- cited below, with the device each was measured on.
  * downlink capacity / R_tx  SIM -- the orbit sim's own output (results/passes.csv) + sat7.orbit.
  * D_tx (bytes sent/day)   SIM -- the scheduler's real-detection result (report 05: 108.0 MB/day).
  * P_k  (stage powers)     ASSUMPTION -- invented, physically-motivated, and SWEPT here. P_gpu is
    bounded by a real spec (RTX 5060 Laptop TGP 35-115 W, the laptop every timing was taken on --
    START_HERE flags that this is NOT flight hardware), but the active draw during a single
    inference is below TGP and is itself an assumption. P_tx is the one genuinely missing number:
    the link model (sat7.orbit.LinkConfig / rate_bps) carries SNR and data rate but no transmit
    POWER, so P_tx is taken as representative of published S-band cubesat transmitters (~8-30 W DC
    while keying) and swept; it is ASSUMPTION, not a datasheet citation.

The headline is deliberately robust: for the DEFAULT onboard build (learned gate + ONNX FP32
detector, both on CPU -- START_HERE says ONNX FP32 is the onboard build), every processing stage
runs on the same processor, so P_cpu CANCELS in ES_proc and the processing saving is a pure time
ratio that needs no power assumption at all. The power assumptions only enter the absolute joules
and the processing-vs-comms-vs-bent-pipe comparison.

    .venv/Scripts/python.exe scripts/wp17_energy_model.py

Outputs: results/wp17_energy_model.json, results/wp17_energy_stages.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sat7.orbit import LINK_PRESETS, rate_bps

# ------------------------------------------------------------------- ASSUMPTION stage powers (W)
# Invented, physically-motivated, SWEPT below. See module docstring for provenance of each.
P_DEFAULT = {"cpu": 28.0, "gpu": 60.0, "tx": 15.0}
P_SWEEP = {"cpu": (15.0, 45.0), "gpu": (35.0, 115.0), "tx": (8.0, 30.0)}  # (low, high)

MB = 1e6


def _load(path: Path):
    if not path.exists():
        raise SystemExit(f"energy model: source file {path} is missing -- cannot read live. "
                         f"Run the WP that produces it before trusting any energy figure.")
    return json.loads(path.read_text())


def load_stage_times(results: Path):
    """Read every measured stage TIME live from the result file that owns it. 'The .csv/.json
    files always win' (START_HERE 7); several of these have moved before. Returns (T, provenance).

    Each figure is pulled from its authoritative producer, not hardcoded:
      detector GPU   wp1_metrics.json  inference_ms_per_tile_gpu
      detector CPU   wp1_export.json   latency_ms_per_tile.onnx_fp32_cpu  (the onboard build)
      learned gate   wp3_gate.json     infer_ms_per_tile / infer_ms_per_tile_cpu (PURE forward)
      classic filter wp11_integration.json  stage_ms_per_tile.prefilter
    Two figures are NOT in any results file and are flagged as such, not silently substituted:
      preprocess/JPEG-decode (report 03's ~1,400 tiles/s, never emitted to a file) and the
      semantic stage (an ASSUMPTION, swept). Both are SHARED stages that cancel in ES_proc.
    """
    gate = _load(results / "wp3_gate.json")
    det_m = _load(results / "wp1_metrics.json")
    det_x = _load(results / "wp1_export.json")
    integ = _load(results / "wp11_integration.json")
    T = {
        "preprocess_decode_cpu": 0.71,
        "classic_prefilter_cpu": float(integ["stage_ms_per_tile"]["prefilter"]),
        "gate_gpu": float(gate["infer_ms_per_tile"]),
        "gate_cpu": float(gate["infer_ms_per_tile_cpu"]),
        "detect_gpu": float(det_m["inference_ms_per_tile_gpu"]),
        "detect_cpu_onnx": float(det_x["latency_ms_per_tile"]["onnx_fp32_cpu"]),
        "fuse_cpu": 0.0,
        "semantic_cpu": 0.5,
    }
    prov = {
        "preprocess_decode_cpu": "UNSOURCED (REAL-derived): report 03 ~1,400 tiles/s warm JPEG "
                                 "decode; not emitted to any results file. Shared stage, cancels in ES_proc.",
        "classic_prefilter_cpu": "REAL live: wp11_integration.json stage_ms_per_tile.prefilter",
        "gate_gpu": "REAL live: wp3_gate.json infer_ms_per_tile (pure forward, GPU)",
        "gate_cpu": "REAL live: wp3_gate.json infer_ms_per_tile_cpu",
        "detect_gpu": "REAL live: wp1_metrics.json inference_ms_per_tile_gpu",
        "detect_cpu_onnx": "REAL live: wp1_export.json latency_ms_per_tile.onnx_fp32_cpu (onboard build)",
        "fuse_cpu": "definitional: single-tile frame has no cross-slice fusion (B2/SAHI adds it)",
        "semantic_cpu": "ASSUMPTION (invented, not measured): LoD encode + schedule decision, CPU",
    }
    unsourced = ["preprocess_decode_cpu (not in any results file)",
                 "semantic_cpu (ASSUMPTION, not a measurement)"]
    return T, prov, unsourced


def load_downlink(results: Path):
    """Read MB/day live from wp6_real_table.csv (it has moved: 117.9 -> 108.0 after conf_high 0.670).

    d_raw  = the bent-pipe row's MB_offered (the full raw volume offered per day)
    d_sent = the 'Ours: LoD + value-greedy' row's MB_sent (what the semantic downlink actually sends)
    Also returns that row's own data_reduction_x, which the sanity gate reproduces.
    """
    df = pd.read_csv(results / "wp6_real_table.csv")
    raw = df[df.strategy.str.contains("Bent pipe", case=False, na=False)]
    ours = df[df.strategy.str.contains("value-greedy", case=False, na=False)]
    if raw.empty or ours.empty:
        raise SystemExit("energy model: wp6_real_table.csv has no 'Bent pipe' and/or 'value-greedy' "
                         f"row -- strategies present: {list(df.strategy)}")
    d_raw = float(raw.iloc[0]["MB_offered"])
    d_sent = float(ours.iloc[0]["MB_sent"])
    csv_reduction = float(ours.iloc[0]["data_reduction_x"])
    return d_sent, d_raw, csv_reduction


def e_proc_per_tile(cfg: str, P: dict, T: dict) -> dict:
    """Per-tile processing energy (J) by stage, for one pipeline config. Detector counted once per
    tile in BOTH baseline and proposed (report 13's convention), so it cancels in ES_proc and the
    gate's extra empty-tile skipping is an uncounted *additional* saving (report 04)."""
    dec = P["cpu"] * T["preprocess_decode_cpu"] / 1000      # shared, cancels
    sem = P["cpu"] * T["semantic_cpu"] / 1000
    if cfg == "cpu_onnx":   # the stated onboard build: gate + ONNX FP32 detector, all on CPU
        gate, detect = P["cpu"] * T["gate_cpu"] / 1000, P["cpu"] * T["detect_cpu_onnx"] / 1000
        classic = P["cpu"] * T["classic_prefilter_cpu"] / 1000
    elif cfg == "gpu":      # report 12's measured budget: GPU gate + GPU detector
        gate, detect = P["gpu"] * T["gate_gpu"] / 1000, P["gpu"] * T["detect_gpu"] / 1000
        classic = P["cpu"] * T["classic_prefilter_cpu"] / 1000   # the classic stage is CPU either way
    else:
        raise ValueError(cfg)
    base = {"preprocess": dec, "gate_or_prefilter": classic, "detect": detect, "fuse": 0.0, "semantic": sem}
    prop = {"preprocess": dec, "gate_or_prefilter": gate, "detect": detect, "fuse": 0.0, "semantic": sem}
    return {"baseline": base, "proposed": prop,
            "E_base": sum(base.values()), "E_prop": sum(prop.values())}


def sahi_variant(proc: dict, results: Path, tiles_per_day: int, e_comm_J: float, e_bent_J: float):
    """The same model with the detector called once per SAHI window instead of once per tile.

    A labelled VARIANT: the default block is untouched and stays the one that describes the simulated
    day, whose detections come from one detector call per native 768 px tile. Three choices, stated:

      * detector calls per tile -- read live from the same-input run (wp26): 25 windows per 4x4-tile
        scene = 1.5625. Applied to the baseline and the proposed pipeline alike.
      * the gate stays at ONE call per tile. That is how it is trained and run everywhere it runs
        (wp3, wp3_score_tiles, wp11): one 128 px view of a whole 768 px tile. No per-window gate
        exists in the repo; the per-window figure is reported as a sensitivity only.
      * fusion (global-coordinate lift + cross-window NMS) is charged 0 ms. ASSUMPTION: no timing
        of it exists anywhere in the repo.
    Returns None when wp26_b0_b4.json is absent (the default model does not need it).
    """
    path = results / "wp26_b0_b4.json"
    if not path.exists():
        return None
    w26 = json.loads(path.read_text(encoding="utf-8"))
    windows = float(w26["checks"]["3_B2_sahi_fusion"]["windows_per_scene"])
    tiles_per_scene = int(w26["inputs"]["per_side"]) ** 2
    calls = windows / tiles_per_scene

    def build(gate_calls: float) -> dict:
        base, prop = dict(proc["baseline"]), dict(proc["proposed"])
        base["detect"], prop["detect"] = base["detect"] * calls, prop["detect"] * calls
        prop["gate_or_prefilter"] = prop["gate_or_prefilter"] * gate_calls     # the classic stage is per tile
        e_base, e_prop = sum(base.values()), sum(prop.values())
        day = e_prop * tiles_per_day
        return {"baseline": base, "proposed": prop, "E_base": e_base, "E_prop": e_prop,
                "ES_proc": 1 - e_prop / e_base, "E_proc_day_J": day, "E_base_day_J": e_base * tiles_per_day,
                "E_total_J": day + e_comm_J}
    v, w = build(1.0), build(calls)
    return {
        "label": "VARIANT, not the default: the detector is called once per SAHI window instead of once per "
                 "tile. Stage times REAL (laptop RTX 5060 / CPU, not flight hw); powers ASSUMPTION; fusion "
                 "time ASSUMPTION (0 ms, never measured) -> every joule here is an ESTIMATE",
        "config": "cpu_onnx",
        "applies_to": "a pipeline that slices scenes with SAHI (the paper's B2 / B3). The simulated day takes "
                      "its detections from one detector call per native 768 px tile, which is what the "
                      "default block above costs.",
        "detector_calls_per_tile": calls,
        "detector_calls_source": f"results/wp26_b0_b4.json: {windows:g} windows per scene / {tiles_per_scene} "
                                 f"tiles per scene (window 768, overlap 0.20)",
        "gate_calls_per_tile": 1.0,
        "gate_note": "once per tile, as the gate is trained and run everywhere it runs (wp3, wp3_score_tiles, "
                     "wp11): one 128 px view of a whole 768 px tile. No per-window gate exists in the repo.",
        "fuse_cpu_ms": 0.0,
        "fuse_note": "ASSUMPTION: the global-coordinate lift + cross-window NMS has never been timed anywhere "
                     "in the repo, so there is no evidence for or against 0 ms",
        "E_proc_per_tile_J": {"baseline": v["baseline"], "proposed": v["proposed"],
                              "E_base": round(v["E_base"], 4), "E_prop": round(v["E_prop"], 4),
                              "ES_proc": round(v["ES_proc"], 4)},
        "E_per_day_kJ": {
            "tiles_per_day": tiles_per_day,
            "E_proc_proposed": round(v["E_proc_day_J"] / 1000, 2),
            "E_proc_baseline": round(v["E_base_day_J"] / 1000, 2),
            "E_comm_proposed_semantic": round(e_comm_J / 1000, 2),
            "E_total_proposed": round(v["E_total_J"] / 1000, 2),
            "proc_vs_comm_ratio_proposed": round(v["E_proc_day_J"] / e_comm_J, 2),
            "ES_total_vs_bent_pipe": round(1 - v["E_total_J"] / e_bent_J, 4)},
        "sensitivity_if_the_gate_ran_once_per_window": {
            "note": "NOT how the gate runs; shown only because the choice moves ES_proc",
            "gate_calls_per_tile": calls, "E_prop_J_per_tile": round(w["E_prop"], 4),
            "E_proc_proposed_kJ_day": round(w["E_proc_day_J"] / 1000, 2), "ES_proc": round(w["ES_proc"], 4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results",
                    help="directory the stage-time and MB/day source files are read from")
    ap.add_argument("--passes", type=Path, default=ROOT / "results" / "passes.csv")
    ap.add_argument("--tiles-per-day", type=int, default=40_000, help="ASSUMPTION: imager duty cycle")
    ap.add_argument("--d-sent-mb", type=float, default=None, help="override; default reads wp6_real_table.csv")
    ap.add_argument("--d-raw-mb", type=float, default=None, help="override; default reads wp6_real_table.csv")
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--gate-tol", type=float, default=0.01, help="sanity-gate tolerance on the data-reduction cross-check")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    # ---- read every measured figure LIVE from its source file (no hardcoded stage times / MB/day)
    T, prov, unsourced = load_stage_times(args.results)
    d_sent_csv, d_raw_csv, csv_reduction = load_downlink(args.results)
    d_sent = args.d_sent_mb if args.d_sent_mb is not None else d_sent_csv
    d_raw = args.d_raw_mb if args.d_raw_mb is not None else d_raw_csv

    # ---- downlink: reuse the orbit sim's own per-pass capacity + contact time (results/passes.csv)
    pz = pd.read_csv(args.passes)
    contact_s = float(pz.duration_min.sum()) * 60.0
    cap_mb_day = float(pz.capacity_MB.sum())
    R_eff_Bps = cap_mb_day * MB / contact_s                 # effective bytes/s averaged over contact
    link = LINK_PRESETS[args.link]
    R_zenith_bps = float(rate_bps(np.array([500.0]), 500.0, link)[0])  # peak (zenith) rate from the model
    tx_time = lambda d_mb: d_mb * MB / R_eff_Bps            # E_comm = P_tx * D_tx / R_tx

    # ---- SANITY GATE: refuse to report ES until two independent checks pass (wp16 pattern).
    #  (a) the data-reduction factor this model derives from the two MB/day figures must reproduce
    #      wp6_real_table.csv's OWN separately-computed data_reduction_x -- catches reading the wrong
    #      rows, or a refreshed CSV where offered/sent moved apart.
    #  (b) the effective link rate averaged over a contact cannot exceed the model's zenith peak --
    #      a physical bound on the SIM downlink, independent of the energy arithmetic.
    model_reduction = d_raw / d_sent
    gate = {
        "check_a_data_reduction": {"model_d_raw/d_sent": round(model_reduction, 2),
                                   "wp6_real_table_data_reduction_x": round(csv_reduction, 2),
                                   "rel_err": round(abs(model_reduction - csv_reduction) / csv_reduction, 5)},
        "check_b_rate_bound": {"R_eff_Mbps": round(R_eff_Bps * 8 / 1e6, 3),
                               "R_zenith_Mbps": round(R_zenith_bps / 1e6, 3)},
    }
    if abs(model_reduction - csv_reduction) / csv_reduction > args.gate_tol:
        raise SystemExit(f"[sanity] FAILED (a): model reduction {model_reduction:.1f}x != "
                         f"wp6_real_table data_reduction_x {csv_reduction:.1f}x -- d_raw/d_sent were "
                         f"read from inconsistent rows, or the CSV moved. Fix before trusting ES.")
    if R_eff_Bps * 8 > R_zenith_bps * 1.001:
        raise SystemExit(f"[sanity] FAILED (b): effective rate {R_eff_Bps*8/1e6:.2f} Mbps exceeds the "
                         f"zenith peak {R_zenith_bps/1e6:.2f} Mbps -- impossible; passes.csv or the link "
                         f"model is inconsistent.")
    gate["passed"] = True
    print(f"[sanity] data-reduction {model_reduction:.1f}x == wp6 {csv_reduction:.1f}x "
          f"(rel err {gate['check_a_data_reduction']['rel_err']:.1e}); "
          f"R_eff {R_eff_Bps*8/1e6:.2f} <= zenith {R_zenith_bps/1e6:.2f} Mbps -> OK\n")

    P = P_DEFAULT
    # ---- processing energy, both configs
    proc = {c: e_proc_per_tile(c, P, T) for c in ("cpu_onnx", "gpu")}
    es_proc = {c: 1 - proc[c]["E_prop"] / proc[c]["E_base"] for c in proc}

    # ---- per-day totals (default config = cpu_onnx, the stated onboard build)
    tpd = args.tiles_per_day
    Eproc_prop_day = proc["cpu_onnx"]["E_prop"] * tpd
    Eproc_base_day = proc["cpu_onnx"]["E_base"] * tpd
    Ecomm_prop_day = P["tx"] * tx_time(d_sent)             # semantic downlink
    Ecomm_bent_day = P["tx"] * tx_time(d_raw)              # bent pipe: send the raw volume
    E_prop_total = Eproc_prop_day + Ecomm_prop_day
    E_base_bentpipe = Ecomm_bent_day                        # B0: no onboard processing, send raw
    es_total_vs_bentpipe = 1 - E_prop_total / E_base_bentpipe

    # ---- sensitivity: ES_proc(gpu) and ES_total at the P_k sweep extremes (report 11 pattern)
    def recompute(**over):
        Pp = {**P, **over}
        pr = {c: e_proc_per_tile(c, Pp, T) for c in ("cpu_onnx", "gpu")}
        esp = {c: 1 - pr[c]["E_prop"] / pr[c]["E_base"] for c in pr}
        ep = pr["cpu_onnx"]["E_prop"] * tpd + Pp["tx"] * tx_time(d_sent)
        eb = Pp["tx"] * tx_time(d_raw)
        return esp, 1 - ep / eb
    sweep = {}
    for k, (lo, hi) in P_SWEEP.items():
        key = {"cpu": "cpu", "gpu": "gpu", "tx": "tx"}[k]
        esp_lo, est_lo = recompute(**{key: lo})
        esp_hi, est_hi = recompute(**{key: hi})
        sweep[f"P_{k}"] = {"range_W": [lo, hi],
                           "ES_proc_gpu": [round(esp_lo["gpu"], 4), round(esp_hi["gpu"], 4)],
                           "ES_total": [round(est_lo, 4), round(est_hi, 4)]}

    # ---- report
    rep = {
        "label_note": "T_k REAL (laptop RTX 5060, not flight hw); downlink SIM (orbit sim); "
                      "P_k ASSUMPTION (swept). ES_proc for the default CPU-ONNX build is P-independent. "
                      "All stage times and MB/day are read LIVE from their source files (see provenance).",
        "stage_times_ms": T,
        "stage_time_provenance": prov,
        "unsourced_figures": unsourced,
        "sanity_gate": gate,
        "powers_W_assumption": P,
        "downlink": {
            "source": "results/passes.csv (SGP4 orbit sim) + sat7.orbit.rate_bps; "
                      "D_sent/D_raw live from results/wp6_real_table.csv",
            "passes_per_day": int(len(pz)), "contact_min_day": round(contact_s / 60, 1),
            "capacity_MB_day": round(cap_mb_day, 1),
            "R_eff_Mbps": round(R_eff_Bps * 8 / 1e6, 3),
            "R_zenith_Mbps_model": round(R_zenith_bps / 1e6, 3),
            "D_sent_MB_day": round(d_sent, 2), "D_raw_MB_day": round(d_raw, 2),
            "capacity_used_by_semantic": round(d_sent / cap_mb_day, 4),
            "bent_pipe_days_of_contact_needed": round(d_raw / cap_mb_day, 1),
        },
        "E_proc_per_tile_J": {
            "cpu_onnx_DEFAULT": {"baseline": proc["cpu_onnx"]["baseline"],
                                 "proposed": proc["cpu_onnx"]["proposed"],
                                 "E_base": round(proc["cpu_onnx"]["E_base"], 4),
                                 "E_prop": round(proc["cpu_onnx"]["E_prop"], 4),
                                 "ES_proc": round(es_proc["cpu_onnx"], 4),
                                 "ES_proc_note": "P_cpu cancels (all-CPU) -> = time ratio, "
                                                 "power-independent; LOWER BOUND (gate also skips "
                                                 "40.2% of empty tiles, report 04, uncounted here)"},
            "gpu_report12_budget": {"baseline": proc["gpu"]["baseline"],
                                    "proposed": proc["gpu"]["proposed"],
                                    "E_base": round(proc["gpu"]["E_base"], 4),
                                    "E_prop": round(proc["gpu"]["E_prop"], 4),
                                    "ES_proc": round(es_proc["gpu"], 4),
                                    "ES_proc_note": "depends on P_cpu/P_gpu (classic stage is CPU, "
                                                    "detector GPU); energy saving < the 6x TIME saving "
                                                    "because the eliminated CPU stage draws less than "
                                                    "the retained GPU detector"},
        },
        "E_per_day_kJ": {
            "tiles_per_day": tpd,
            "E_proc_proposed": round(Eproc_prop_day / 1000, 2),
            "E_proc_baseline": round(Eproc_base_day / 1000, 2),
            "E_comm_proposed_semantic": round(Ecomm_prop_day / 1000, 2),
            "E_comm_bent_pipe_raw": round(Ecomm_bent_day / 1000, 2),
            "E_total_proposed": round(E_prop_total / 1000, 2),
            "proc_vs_comm_ratio_proposed": round(Eproc_prop_day / Ecomm_prop_day, 2),
            "ES_total_vs_bent_pipe": round(es_total_vs_bentpipe, 4),
        },
        "sensitivity_P_sweep": sweep,
    }
    sahi = sahi_variant(proc["cpu_onnx"], args.results, tpd, Ecomm_prop_day, E_base_bentpipe)
    if sahi is not None:                                    # appended last: the default block is unchanged
        rep["sahi_variant"] = sahi
    (args.out / "wp17_energy_model.json").write_text(json.dumps(rep, indent=2))

    # stage CSV (default config)
    rows = []
    for which in ("baseline", "proposed"):
        for stage, j in proc["cpu_onnx"][which].items():
            rows.append({"config": "cpu_onnx", "pipeline": which, "stage": stage, "E_J_per_tile": j})
    pd.DataFrame(rows).to_csv(args.out / "wp17_energy_stages.csv", index=False)

    # ---- print
    print("=== WP17 per-stage energy model (REAL times, SIM downlink, ASSUMPTION powers) ===\n")
    print(f"downlink (orbit sim): {len(pz)} passes, {contact_s/60:.1f} min contact, "
          f"{cap_mb_day:.0f} MB/day capacity, R_eff {R_eff_Bps*8/1e6:.2f} Mbps")
    print(f"semantic downlink sends {d_sent:.0f} MB = {d_sent/cap_mb_day:.1%} of "
          f"capacity; a bent pipe would need {d_raw/cap_mb_day:.0f}x the contact (infeasible)\n")
    print("E_proc per tile (J):")
    for c, name in (("cpu_onnx", "DEFAULT  gate+ONNX-FP32, all CPU"), ("gpu", "GPU budget (report 12)")):
        print(f"  {name:34s}  base {proc[c]['E_base']:.3f}  prop {proc[c]['E_prop']:.3f}  "
              f"ES_proc {es_proc[c]:+.1%}")
    print(f"\nper day @ {tpd:,} tiles:  E_proc prop {Eproc_prop_day/1000:.1f} kJ  "
          f"base {Eproc_base_day/1000:.1f} kJ   |  E_comm semantic {Ecomm_prop_day/1000:.1f} kJ  "
          f"bent-pipe {Ecomm_bent_day/1000:.0f} kJ")
    print(f"proc:comm ratio (proposed) {Eproc_prop_day/Ecomm_prop_day:.1f}:1  "
          f"-> once you stop sending pixels, COMPUTE is the bigger energy cost, not the radio")
    print(f"ES_total vs bent pipe: {es_total_vs_bentpipe:.1%} "
          f"(dominated by the {d_raw/d_sent:.0f}x data cut)")
    print("\nsensitivity (ES robust across the P_k sweep):")
    for k, v in sweep.items():
        print(f"  {k} {v['range_W']} W -> ES_proc_gpu {v['ES_proc_gpu']}, ES_total {v['ES_total']}")
    if sahi is not None:
        d, s = sahi["E_per_day_kJ"], sahi["E_proc_per_tile_J"]
        print(f"\nVARIANT (not the default) -- SAHI, {sahi['detector_calls_per_tile']:g} detector calls per tile, "
              f"gate once per tile, fusion 0 ms [ASSUMPTION]:")
        print(f"  E_proc {d['E_proc_proposed']} kJ/day (default {Eproc_prop_day / 1000:.2f}), ES_proc "
              f"{s['ES_proc']:+.1%} (default {es_proc['cpu_onnx']:+.1%}), proc:comm "
              f"{d['proc_vs_comm_ratio_proposed']}:1, ES_total {d['ES_total_vs_bent_pipe']:.1%}")
    print(f"\nSaved wp17_energy_model.json, wp17_energy_stages.csv in {args.out}")


if __name__ == "__main__":
    main()
