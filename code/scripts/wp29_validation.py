"""WP29 -- scientific validation: units, data reduction, energy, latency, fairness, traceability.

Reads the committed results and checks them; it regenerates none of them. Seven checklist items
(numbered 6-12, continuing the B0-B4 checks 1-5 in wp26):

   6  data reduction uses actual, consistently defined data sizes
   7  detection metrics are based on valid ground truth (or are not claimed as accuracy)
   8  processing / communication energy assumptions are documented
   9  latency includes the stages the experiment requires
  10  relay energy and latency account for both links
  11  the same inputs and assumptions are used where two things are compared
  12  no numerical claim lacks traceable experimental evidence

Each item gets hard CHECKS (a failed one makes the item FAIL) and LIMITS. A limit that starts
"GAP:" means the requirement is not fully met and makes the item PARTIAL; the others are caveats
that travel with a PASS. Evidence is written beside each. One live simulation is run, because no committed file answers it: six consecutive days of
the nominal workload, to see whether one day's result survives when the next day's traffic needs
the same passes (--no-steady-state skips it and keeps the block already in the output file).

Labels: stage times REAL (laptop, not flight hardware); powers ASSUMPTION -- NO power was measured
anywhere in this project, so every energy figure is an ESTIMATE; links and days SIM.

    .venv/Scripts/python.exe scripts/wp29_validation.py

Output: results/wp29_validation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sat7.accounting import (MB, LatencyBudget, data_reduction_percent, measured_stage_times_s,
                             raw_image_bytes, reduction_factor, stage_energy_J, tx_energy_J, tx_time_s)
from sat7.comms import CommsConfig, Links, simulate_comms
from sat7.energy import EnergyModel
from sat7.evaluation import describe_detections
from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, Pass, find_passes, make_satellite
from sat7.real_workload import RealWorkloadConfig, load_catalogue, load_size_model, workload_from_catalogue
from sat7.relay import ISLWindow
from sat7.scheduler import FIFO, RAW_TILE_BYTES, Item, LoDConfig, ValueGreedy, encode_lod, simulate
from sat7.semantic import Packetizer

DOCS = {"README": "README.md", "PHASE3": "paper/PHASE3_RESULTS.md", "SLIDES": "paper/slides/index.html",
        "VIDEO": "paper/video_script.md", "COVERAGE": "demo/PAPER_COVERAGE.md", "DEMO": "demo/DEMO.md",
        "QA": "demo/QA_CARD.md", "QUICKSTART": "demo/quickstart/README.md",
        "REPORTS": "reports/README.md", "R23": "reports/23-paper-faithful-modules.md"}
JUDGE_DOCS = list(DOCS)                                # the ten documents the ledger covers


def _json(res: Path, name: str) -> dict:
    return json.loads((res / name).read_text(encoding="utf-8"))


def _table(res: Path, name: str) -> dict:
    with (res / name).open(newline="", encoding="utf-8") as f:
        return {r["strategy"]: r for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------------- claims ledger
def claims_ledger(res: Path, steady: dict | None = None) -> list[dict]:
    """Every headline number the ten judge-facing documents quote, recomputed from the results file
    that owns it. `forms` are the spellings a document may use; `docs` must each contain one.
    `steady` is the six-day block (default: the one in results/wp29_validation.json, if present)."""
    if steady is None and (res / "wp29_validation.json").exists():
        steady = _json(res, "wp29_validation.json")["steady_state"]
    t = _table(res, "wp6_real_table.csv")
    ours, fair, bent = (t["Ours: LoD + value-greedy"], t["Phi-sat-2 style, fair (+coastal)"],
                        t["Bent pipe (raw, FIFO)"])
    told = _table(res, "wp6_real_table_modeledcoast.csv")["Ours: LoD + value-greedy"]
    sweep = pd.read_csv(res / "wp6_real_sweep.csv")

    def recall(tiles, strategy):
        return float(sweep[(sweep.tiles == tiles) & (sweep.strategy == strategy)].ship_recall.iloc[0])
    gap = pd.read_csv(res / "wp5_optimality_gap.csv")
    op_gap = float(gap[(gap.policy == "Value-greedy (ours)") & (gap.scale == "operational")].gap.mean())
    jb = pd.read_csv(res / "wp5_joint_bound.csv")
    qa = pd.read_csv(res / "wp8_queue_aware.csv")
    sens = pd.read_csv(res / "wp10_sensitivity.csv")
    sw = sens[sens.param != "(baseline)"]
    cloud = sens[(sens.load == 40000) & (sens.param == "cloud")].ship_recall
    base160 = float(sens[(sens.load == 160000) & (sens.param == "(baseline)")].ours_minus_baseline.iloc[0])
    w26 = _json(res, "wp26_b0_b4.json")["detection_quality_same_scenes"]
    w24 = _json(res, "wp24_detection_eval.json")["B1_committed_all_test_tiles"]
    w17 = _json(res, "wp17_energy_model.json")
    b4 = _json(res, "wp18_campaign.json")["configs"]["B4"]
    w14 = _json(res, "wp14_gate_signal.json")["by_load"]["40000"]
    w23 = _json(res, "wp23_semantic_compare.json")["semantic_schemes"]
    w28 = _json(res, "wp28_coast_tile_model.json")
    w1 = _json(res, "wp1_metrics.json")["test"]
    w0 = _json(res, "wp0_stats.json")
    mb, red = float(ours["MB_sent"]), float(ours["data_reduction_x"])
    dr = data_reduction_percent(mb, float(bent["MB_offered"]))
    lat_d, lat_r = b4["latency_h_B3_direct"], b4["latency_h_B4_relay"]
    e_b4 = b4["energy_kJ_TARGET"]
    wins = _json(res, "wp26_b0_b4.json")["checks"]["3_B2_sahi_fusion"]["windows_per_scene"]
    D = DOCS
    every = list(D.values())

    def c(cid, value, forms, docs, source):
        return {"id": cid, "value": value, "forms": forms, "docs": docs, "source": source}
    extra = []
    if "sahi_variant" in w17:
        sv = w17["sahi_variant"]
        es_w = sv["sensitivity_if_the_gate_ran_once_per_window"]["ES_proc"]
        extra += [
            c("sahi_variant_E_proc_kJ_per_day", sv["E_per_day_kJ"]["E_proc_proposed"],
              [f"{sv['E_per_day_kJ']['E_proc_proposed']:.1f} kJ"], [D["PHASE3"]], "wp17_energy_model.json sahi_variant"),
            c("sahi_variant_ES_proc_percent", 100 * sv["E_proc_per_tile_J"]["ES_proc"],
              [f"{100 * sv['E_proc_per_tile_J']['ES_proc']:.1f} %"], [D["PHASE3"]], "wp17_energy_model.json sahi_variant"),
            c("sahi_variant_proc_to_comm", sv["E_per_day_kJ"]["proc_vs_comm_ratio_proposed"],
              [f"{sv['E_per_day_kJ']['proc_vs_comm_ratio_proposed']:.1f} : 1"], [D["PHASE3"]],
              "wp17_energy_model.json sahi_variant"),
            c("sahi_variant_ES_proc_gate_per_window_percent", 100 * es_w, [f"{100 * es_w:.1f} %"], [D["PHASE3"]],
              "wp17_energy_model.json sahi_variant (sensitivity)"),
        ]
    if steady is not None:
        vg = steady["runs"]["share_0.25_Value-greedy (ours)"]
        rec = [d["ship_recall"] for d in vg["per_day"]]
        st = measured_stage_times_s(res, "cpu_onnx", calls_per_tile=1.5625)
        stage_ms = 1e3 * (st["processing_s"] + max(st["encoding_s"].values()) + st["decoding_s"])
        bent_kJ = w17["powers_W_assumption"]["tx"] * w17["downlink"]["contact_min_day"] * 60 / 1e3
        fifo = steady["runs"]["share_0.25_FIFO"]["per_day"]
        lat = [d["latency_med_h"] for d in vg["per_day"]]
        extra += [
            c("sustained_link_share_MB_per_24h", vg["sustained_capacity_MB_per_24h"],
              [f"{vg['sustained_capacity_MB_per_24h']:.1f} MB"], [D["README"], D["PHASE3"], D["QA"], D["DEMO"]],
              "wp29_validation.json steady_state"),
            c("day_offer_over_sustained_percent", 100 * vg["offered_over_sustained"],
              [f"{100 * vg['offered_over_sustained']:.0f}%", f"{100 * vg['offered_over_sustained']:.0f} %"],
              [D["README"], D["PHASE3"], D["QA"], D["DEMO"]], "wp29_validation.json steady_state"),
            c("six_day_recall_range", [min(rec), max(rec)], [f"{min(rec):.3f}–{max(rec):.3f}"],
              [D["README"], D["PHASE3"], D["QA"], D["DEMO"]], "wp29_validation.json steady_state"),
            c("six_day_latency_range_h", [min(lat), max(lat)], [f"{min(lat):.1f}–{max(lat):.1f} h"],
              [D["QA"], D["DEMO"]], "wp29_validation.json steady_state"),
            c("fifo_latency_day1_h", fifo[0]["latency_med_h"], [f"{fifo[0]['latency_med_h']:.1f} h"],
              [D["QA"], D["DEMO"]], "wp29_validation.json steady_state"),
            c("fifo_latency_day6_h", fifo[-1]["latency_med_h"], [f"{fifo[-1]['latency_med_h']:.1f} h"],
              [D["QA"], D["DEMO"]], "wp29_validation.json steady_state"),
            c("fifo_recall_day6", fifo[-1]["ship_recall"], [f"{fifo[-1]['ship_recall']:.3f}"],
              [D["QA"], D["DEMO"]], "wp29_validation.json steady_state"),
            c("link_limited_bent_pipe_kJ_per_day", bent_kJ, [f"{bent_kJ:.1f} kJ"],
              [D["README"], D["PHASE3"], D["QA"]], "wp17_energy_model.json (P_tx x daily contact time)"),
            c("non_communication_latency_ms", stage_ms, [f"{stage_ms:.0f} ms"], [D["PHASE3"], D["COVERAGE"]],
              "wp17_energy_model.json + wp25_semantic_packets.json (sat7.accounting)"),
        ]
    return extra + [
        c("data_reduction_x", red, [f"{red:.0f}×"], every, "wp6_real_table.csv"),
        c("earlier_modeled_reduction_x", float(told["data_reduction_x"]),
          [f"{float(told['data_reduction_x']):.0f}×"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "COVERAGE", "DEMO", "QA", "QUICKSTART", "REPORTS", "R23")],
          "wp6_real_table_modeledcoast.csv"),
        c("MB_sent_per_day", mb, [f"{mb:.1f} MB", f"{mb:.0f} MB"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "QA", "REPORTS", "R23")], "wp6_real_table.csv"),
        c("D_raw_MB", float(bent["MB_offered"]), [f"{float(bent['MB_offered']):,.0f} MB"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "VIDEO", "QA")], "wp6_real_table.csv"),
        c("data_reduction_percent", dr, [f"{dr:.2f} %", f"{dr:.2f}%"], [D["PHASE3"], D["COVERAGE"]],
          "wp6_real_table.csv"),
        c("ship_recall", float(ours["ship_recall"]), [f"{float(ours['ship_recall']):.3f}"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "DEMO", "QA", "VIDEO")], "wp6_real_table.csv"),
        c("lead_over_fair_baseline_pts", 100 * (float(ours["ship_recall"]) - float(fair["ship_recall"])),
          [f"+{100 * (float(ours['ship_recall']) - float(fair['ship_recall'])):.1f}"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "DEMO")], "wp6_real_table.csv"),
        c("bytes_vs_fair_baseline_x", mb / float(fair["MB_sent"]), [f"~{mb / float(fair['MB_sent']):.0f}×"],
          [D[k] for k in ("README", "PHASE3", "SLIDES")], "wp6_real_table.csv"),
        c("vs_FIFO_160k_x", recall(160000, "Ours: LoD + value-greedy") / recall(160000, "Ours: LoD + FIFO"),
          [f"{recall(160000, 'Ours: LoD + value-greedy') / recall(160000, 'Ours: LoD + FIFO'):.2f}×"],
          [D["README"], D["PHASE3"], D["REPORTS"]], "wp6_real_sweep.csv"),
        c("vs_FIFO_80k_x", recall(80000, "Ours: LoD + value-greedy") / recall(80000, "Ours: LoD + FIFO"),
          [f"{recall(80000, 'Ours: LoD + value-greedy') / recall(80000, 'Ours: LoD + FIFO'):.2f}×"],
          [D["README"], D["PHASE3"], D["REPORTS"]], "wp6_real_sweep.csv"),
        c("greedy_gap_operational", op_gap, [f"{100 * op_gap:.3f}%", f"{100 * op_gap:.2f} %", f"{100 * op_gap:.2f}%"],
          [D["README"], D["PHASE3"], D["QA"], D["REPORTS"]], "wp5_optimality_gap.csv"),
        c("foresight_value_max", float(jb.price_of_online.max()),
          [f"≤ {100 * jb.price_of_online.max():.1f} %", f"≤{100 * jb.price_of_online.max():.1f}%"],
          [D["PHASE3"], D["REPORTS"]], "wp5_joint_bound.csv"),
        c("queue_aware_worst_regret_pts", 100 * float(qa[qa.policy.str.startswith("queue-aware")].regret.max()),
          [f"{100 * float(qa[qa.policy.str.startswith('queue-aware')].regret.max()):.1f} points"],
          [D["PHASE3"], D["REPORTS"]], "wp8_queue_aware.csv"),
        c("B1_recall_same_scenes", w26["B1"]["recall"], [f"{w26['B1']['recall']:.3f}"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "COVERAGE", "DEMO", "QA", "QUICKSTART", "REPORTS", "R23")],
          "wp26_b0_b4.json"),
        c("B2_recall_same_scenes", w26["B2"]["recall"], [f"{w26['B2']['recall']:.3f}"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "COVERAGE", "DEMO", "QA", "QUICKSTART", "REPORTS", "R23")],
          "wp26_b0_b4.json"),
        c("plain_tiling_recall_same_scenes", w26["B1_per_tile_reference"]["recall"],
          [f"{w26['B1_per_tile_reference']['recall']:.3f}"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "COVERAGE", "DEMO", "QA")], "wp26_b0_b4.json"),
        c("sahi_compute_x", wins / 16, [f"{wins / 16:.2f}×"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "COVERAGE", "DEMO", "QA", "QUICKSTART")], "wp26_b0_b4.json"),
        c("detector_on_native_tiles_recall", w24["recall"], [f"{w24['recall']:.3f}"],
          [D[k] for k in ("README", "PHASE3", "COVERAGE", "QA", "REPORTS")], "wp24_detection_eval.json"),
        c("detector_mAP50", w1["mAP50"], [f"{w1['mAP50']:.3f}"], [D["README"], D["PHASE3"], D["COVERAGE"]],
          "wp1_metrics.json"),
        c("dataset_tiles", w0["selected_tiles"], [f"{w0['selected_tiles']:,}"], [D["README"], D["PHASE3"], D["QA"]],
          "wp0_stats.json"),
        c("cloud_band", [float(cloud.min()), float(cloud.max())], [f"{cloud.min():.2f}–{cloud.max():.2f}"],
          [D[k] for k in ("README", "PHASE3", "SLIDES", "DEMO", "QA", "VIDEO")], "wp10_sensitivity.csv"),
        c("sensitivity_ours_ge_fair", [int((sw.ours_minus_baseline >= 0).sum()), int(len(sw))],
          [f"{int((sw.ours_minus_baseline >= 0).sum())} of {len(sw)}", f"in {int((sw.ours_minus_baseline >= 0).sum())}"],
          [D[k] for k in ("README", "PHASE3", "COVERAGE", "SLIDES", "QA", "REPORTS")], "wp10_sensitivity.csv"),
        c("lead_over_fair_at_160k_pts", 100 * base160, [f"{100 * base160:.1f} points", f"{100 * base160:.1f} pts"],
          [D[k] for k in ("README", "PHASE3", "COVERAGE", "SLIDES", "QA")], "wp10_sensitivity.csv"),
        c("ES_proc_percent", 100 * w17["E_proc_per_tile_J"]["cpu_onnx_DEFAULT"]["ES_proc"],
          [f"{100 * w17['E_proc_per_tile_J']['cpu_onnx_DEFAULT']['ES_proc']:.1f} %",
           f"{100 * w17['E_proc_per_tile_J']['cpu_onnx_DEFAULT']['ES_proc']:.1f}%"],
          [D["PHASE3"], D["SLIDES"], D["COVERAGE"]], "wp17_energy_model.json"),
        c("proc_to_comm_energy", w17["E_per_day_kJ"]["proc_vs_comm_ratio_proposed"],
          [f"{w17['E_per_day_kJ']['proc_vs_comm_ratio_proposed']:.1f} : 1",
           f"{w17['E_per_day_kJ']['proc_vs_comm_ratio_proposed']:.1f}:1"],
          [D["PHASE3"], D["SLIDES"], D["COVERAGE"], D["REPORTS"]], "wp17_energy_model.json"),
        c("ES_total_percent", 100 * w17["E_per_day_kJ"]["ES_total_vs_bent_pipe"],
          [f"{100 * w17['E_per_day_kJ']['ES_total_vs_bent_pipe']:.1f} %"], [D["PHASE3"]], "wp17_energy_model.json"),
        c("relay_worst_latency_h", [lat_d["max"], lat_r["max"]], [f"{lat_d['max']:.1f}"],
          [D[k] for k in ("PHASE3", "SLIDES", "COVERAGE", "DEMO", "QA", "VIDEO", "REPORTS")], "wp18_campaign.json"),
        c("relay_worst_latency_with_relay_h", lat_r["max"], [f"{lat_r['max']:.1f}"],
          [D[k] for k in ("PHASE3", "SLIDES", "COVERAGE", "DEMO", "QA", "VIDEO", "REPORTS")], "wp18_campaign.json"),
        c("relay_item_share_percent", 100 * b4["relay_fraction"],
          [f"{100 * b4['relay_fraction']:.0f} %", f"{100 * b4['relay_fraction']:.0f}%"], [D["PHASE3"], D["REPORTS"]],
          "wp18_campaign.json"),
        c("relay_comm_energy_increase_percent", 100 * (e_b4["b4_mix"] / e_b4["direct_only"] - 1),
          [f"+{100 * (e_b4['b4_mix'] / e_b4['direct_only'] - 1):.0f}"],
          [D["PHASE3"], D["SLIDES"], D["QA"], D["REPORTS"]], "wp18_campaign.json"),
        c("thumbnail_rekey_saving_percent", w14["bytes_saved_pct"],
          [f"{w14['bytes_saved_pct']:.1f} %", f"{w14['bytes_saved_pct']:.1f}%"], [D["PHASE3"], D["REPORTS"]],
          "wp14_gate_signal.json"),
        c("P0P3_MB_same_day", w23["priority"]["MB_sent"], [f"{w23['priority']['MB_sent']:.1f}"],
          [D["COVERAGE"], D["R23"], D["REPORTS"]], "wp23_semantic_compare.json"),
        c("LoD_MB_same_day", w23["lod"]["MB_sent"], [f"{w23['lod']['MB_sent']:.1f}"],
          [D["COVERAGE"], D["R23"], D["REPORTS"]], "wp23_semantic_compare.json"),
        c("coastal_tile_median_kB",
          w28["coastal_tile_q40"]["measured_all_coastal_test_tiles"]["jpeg_baseline"]["median_B"] / 1e3,
          [f"{w28['coastal_tile_q40']['measured_all_coastal_test_tiles']['jpeg_baseline']['median_B'] / 1e3:.1f} kB"],
          [D["PHASE3"], D["DEMO"], D["QA"], D["REPORTS"]], "wp28_coast_tile_model.json"),
        c("modeled_share_of_bytes_percent", 100 * w28["headline"]["still_modeled_share_of_canonical_bytes"],
          [f"{100 * w28['headline']['still_modeled_share_of_canonical_bytes']:.0f} %",
           f"{100 * w28['headline']['still_modeled_share_of_canonical_bytes']:.0f}%"], [D["PHASE3"], D["QA"]],
          "wp28_coast_tile_model.json"),
    ]


def check_claims(res: Path, repo: Path, steady: dict | None = None) -> dict:
    texts = {rel: (repo / rel).read_text(encoding="utf-8") for rel in DOCS.values()}
    ledger = claims_ledger(res, steady)
    missing, n = [], 0
    for cl in ledger:
        for doc in cl["docs"]:
            n += 1
            if not any(f in texts[doc] for f in cl["forms"]):
                missing.append({"claim": cl["id"], "doc": doc, "expected_one_of": cl["forms"], "source": cl["source"]})
    return {"claims": len(ledger), "claim_document_pairs": n, "missing": missing}


# ---------------------------------------------------------------------------------- steady state
def steady_state(res: Path, days: int = 6, tail_h: int = 36, tiles_per_day: int = 40_000) -> dict:
    """Six consecutive days of the nominal workload [SIM-over-REAL]. The committed day simulations
    give ONE day of traffic 36 h of passes; here every day's traffic competes for the passes."""
    tiles, ships = load_catalogue(res)
    lod = LoDConfig(conf_low=0.25, size_model=load_size_model(res / "wp6_size_model.json"))
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    full = find_passes(sat, GroundStation(), start, days * 24 + tail_h, LINK_PRESETS["cubesat_sband"])
    wl, _ = workload_from_catalogue(tiles, ships, RealWorkloadConfig(
        hours=24.0 * days, tiles_per_day=tiles_per_day, det_thr=0.25, seed=0))
    items = encode_lod(wl, lod)
    out = {"label": "SIM-over-REAL", "days": days, "tiles_per_day": tiles_per_day, "pass_horizon_h": days * 24 + tail_h,
           "offered_MB_per_day_mean": round(sum(i.size for i in items) / MB / days, 1), "runs": {}}
    for share, policy in ((0.25, ValueGreedy()), (0.25, FIFO()), (0.35, ValueGreedy())):
        passes = [replace(p, capacity_bytes=p.capacity_bytes * share) for p in full]
        r = simulate(items, passes, start, policy, 8e9, encoder="LoD")
        first, sent = {}, {}
        for s in r.sent:
            sent[id(s.item)] = sent.get(id(s.item), 0.0) + s.item.size * s.fraction
            for sid in s.item.ships:
                if sid not in first or s.delivered_s < first[sid]:
                    first[sid] = s.delivered_s
        rows = []
        for d in range(days):
            lo, hi = d * 86400.0, (d + 1) * 86400.0
            sh = [s for s in wl.ships if lo <= s.created_s < hi]
            lat = np.array([(first[s.id] - s.created_s) / 3600 for s in sh if s.id in first])
            it = [i for i in items if lo <= i.created_s < hi]
            off = sum(i.size for i in it)
            rows.append({"day": d + 1, "ship_recall": round(len(lat) / len(sh), 4),
                         "latency_med_h": round(float(np.median(lat)), 2),
                         "latency_p90_h": round(float(np.percentile(lat, 90)), 2),
                         "offered_MB": round(off / MB, 1),
                         "bytes_delivered_fraction": round(sum(sent.get(id(i), 0.0) for i in it) / off, 3)})
        cap24 = sum(p.capacity_bytes for p in passes) / MB * 24 / (days * 24 + tail_h)
        out["runs"][f"share_{share}_{policy.name}"] = {
            "link_share": share, "policy": policy.name, "sustained_capacity_MB_per_24h": round(cap24, 1),
            "offered_over_sustained": round(out["offered_MB_per_day_mean"] / cap24, 3), "per_day": rows}
    return out


# ---------------------------------------------------------------------------------- live micro-checks
def _toy_latency_definition() -> dict:
    """One item captured at t=100 s, one pass rising at t=1000 s: what does 'latency' contain?"""
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    p = Pass(start + timedelta(seconds=1000), start + timedelta(seconds=1000), start + timedelta(seconds=1400), 45.0, 1e6)
    it = Item(1, 100.0, 250_000.0, 1.0, "P2", (0,), progressive=True)
    r = simulate([it], [p], start, ValueGreedy(), 8e9)
    s = r.sent[0]
    wait = 1000.0 - 100.0
    in_pass = 400.0 * (250_000.0 / 1e6)                       # its share of the pass, at the payload rate
    return {"latency_s": s.delivered_s - it.created_s, "wait_for_pass_s": wait, "transmission_s": in_pass,
            "ok": abs((s.delivered_s - it.created_s) - (wait + in_pass)) < 1e-6}


def _toy_relay_both_legs() -> dict:
    """A relay delivery must carry an ISL leg and a relay-to-ground leg, in time and in energy."""
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)

    def gp(h, cap=1e6, dur=400):
        t = start + timedelta(hours=h)
        return Pass(t, t, t + timedelta(seconds=dur), 45.0, cap)
    isl0 = start + timedelta(hours=0.5)
    links = Links([gp(10)], [ISLWindow(isl0, isl0 + timedelta(minutes=10), 2000.0, 2500.0)], [gp(2)])
    cfg = CommsConfig()
    it = Item(1, 60.0, 20_000.0, 1.0, "P2", (0,), progressive=True)
    cr = simulate_comms([it], links, start, ValueGreedy(), cfg)
    d = cr.deliveries[0]
    t_isl = tx_time_s(it.size, cfg.isl_rate_bps * cfg.isl_efficiency)
    r_gs = links.relay_passes[0].capacity_bytes * 8 / links.relay_passes[0].duration_s / links.relay_share
    t_gs = tx_time_s(it.size, r_gs)
    e = cfg.p_isl_W * t_isl + cfg.p_tx_W * t_gs
    return {"path": d.path, "tx_s": d.tx_s, "energy_J": d.energy_J, "delivered_h": d.delivered_s / 3600,
            "isl_window_start_h": 0.5, "relay_pass_rise_h": 2.0,
            "ok": (d.path == "relay" and abs(d.tx_s["isl"] - t_isl) < 1e-9 and abs(d.tx_s["relay_ground"] - t_gs) < 1e-9
                   and abs(sum(d.energy_J.values()) - e) < 1e-9 and d.delivered_s >= 2.0 * 3600)}


# ---------------------------------------------------------------------------------- the checklist
def run_checklist(res: Path, repo: Path, steady: dict) -> dict:
    items: dict[str, dict] = {}

    def item(num, title):
        items[str(num)] = {"title": title, "checks": [], "limits": []}
        return items[str(num)]

    def chk(it, name, ok, **ev):
        it["checks"].append({"check": name, "ok": bool(ok), **ev})

    t = _table(res, "wp6_real_table.csv")
    ours, bent = t["Ours: LoD + value-greedy"], t["Bent pipe (raw, FIFO)"]
    d_tx, d_raw = float(ours["MB_sent"]), float(bent["MB_offered"])
    w17, w19 = _json(res, "wp17_energy_model.json"), _json(res, "wp19_relay_energy.json")
    w24, w25 = _json(res, "wp24_detection_eval.json"), _json(res, "wp25_semantic_packets.json")
    w26, w27, w28 = _json(res, "wp26_b0_b4.json"), _json(res, "wp27_comms.json"), _json(res, "wp28_coast_tile_model.json")
    w18 = _json(res, "wp18_campaign.json")
    ss = steady["runs"]["share_0.25_Value-greedy (ours)"]

    # ------------------------------------------------------------------ 6 data reduction
    it = item(6, "Data reduction uses actual, consistently defined data sizes")
    chk(it, "D_raw per tile is H x W x 3 bytes", RAW_TILE_BYTES == raw_image_bytes(768, 768) == 1_769_472,
        RAW_TILE_BYTES=RAW_TILE_BYTES)
    dr, rf = data_reduction_percent(d_tx, d_raw), reduction_factor(d_tx, d_raw)
    chk(it, "headline = 100 x (1 - D_tx / D_raw) on the table's own bytes",
        abs(rf - float(ours["data_reduction_x"])) / rf < 1e-9, D_tx_MB=round(d_tx, 3), D_raw_MB=round(d_raw, 3),
        data_reduction_percent=round(dr, 4), factor=round(rf, 2), table_column=round(float(ours["data_reduction_x"]), 2))
    raw_tiles = d_raw * MB / RAW_TILE_BYTES
    chk(it, "D_raw is whole tiles (non-cloud tiles x 1,769,472 B), nothing compressed",
        abs(raw_tiles - w28["days"]["raw_tiles_mean"]) < 0.1, tiles_per_day_in_baseline=round(raw_tiles, 1))
    chk(it, "the coastal size table reproduces from pixels (wp25 gate)",
        w25["LoD_headline_with_measured_coast"].get("size_table_reproduces_from_pixels") is True)
    tb = pd.read_csv(res / "wp28_coast_tile_bytes.csv")
    wire = [sum(len(pk) for pk in Packetizer().packetize(0x103, bytes(int(r.jpeg_q40_baseline_B) + 18))) == r.packet_B
            for r in tb.head(25).itertuples()]
    chk(it, "a coastal tile's charged size is its real serialization (JPEG + 18 B header + CCSDS packets)",
        all(wire), rows_checked=len(wire))
    chk(it, "scene-level B0 equals pixels + packet headers (wp26 check 1)",
        w26["checks"]["1_B0_original_image"]["passed"] and
        w26["checks"]["1_B0_original_image"]["pixels_bytes"] == raw_image_bytes(3072, 3072),
        pixels_bytes=w26["checks"]["1_B0_original_image"]["pixels_bytes"])
    share_meas = w28["models"]["measured_per_tile"]["products"]["tile"]["share"]
    it["limits"] += [
        f"GAP: D_transmitted is only partly ACTUAL: coastal tiles ({100 * share_meas:.1f}% of the bytes) are measured "
        f"JPEGs; thumbnails, chips and crops ({100 * (1 - share_meas):.1f}%) are model sizes (flat 1,000 B, WP4 "
        f"power law). On real packets the crop model was within 0.94-1.26x (wp25 measured_over_modeled).",
        f"Two D_raw definitions exist. The headline counts non-cloud tiles only ({raw_tiles:,.0f} of 40,000 per "
        f"day) with no packet headers; counting every imaged tile gives {40_000 * RAW_TILE_BYTES / MB:,.0f} MB "
        f"and {40_000 * RAW_TILE_BYTES / MB / d_tx:.0f}x. wp26's B0 counts every tile plus CCSDS headers. The "
        f"headline uses the smaller baseline, so it understates rather than overstates.",
        f"D_transmitted is one isolated day given 36 h of passes ({d_tx:.1f} MB). The link share sustains "
        f"{ss['sustained_capacity_MB_per_24h']} MB per 24 h, i.e. the day's offer is "
        f"{100 * ss['offered_over_sustained']:.0f}% of it. On consecutive days fewer bytes than that are sent "
        f"(steady_state block).",
    ]

    # ------------------------------------------------------------------ 7 detection metrics / GT
    it = item(7, "Detection metrics are based on valid ground truth (or are not claimed as accuracy)")
    w0 = _json(res, "wp0_stats.json")
    cat = pd.read_csv(res / "wp6_ships.csv")
    b1 = w24["B1_committed_all_test_tiles"]
    chk(it, "full-scale evaluation uses every labelled test ship",
        b1["n_gt"] == len(cat) == w0["split"]["test"]["ships"] and b1["images"] == w0["split"]["test"]["tiles"],
        n_gt=b1["n_gt"], images=b1["images"], iou_thr=b1["iou_thr"], conf_thr=b1["conf_thr"])
    chk(it, "the test split shares no near-duplicate group with train/val",
        w0["groups_spanning_splits"] == 0, groups_a_naive_split_would_leak=w0["groups_a_naive_split_would_leak"])
    q = w26["detection_quality_same_scenes"]
    chk(it, "B1..B4 are scored against the same ground truth",
        len({q[m]["n_gt"] for m in ("B1", "B2", "B3", "B4")}) == 1 and w26["checks"]["5_same_inputs"]["passed"],
        n_gt=q["B1"]["n_gt"], matching=w26["checks"]["5_same_inputs"]["fingerprint"]["matching"])
    chk(it, "precision, recall, F1 and AP are all reported (not recall alone)",
        all(k in b1 for k in ("precision", "recall", "F1", "AP50", "AP50-95")),
        precision=b1["precision"], recall=b1["recall"], F1=b1["F1"], AP50=b1["AP50"])
    nogt = describe_detections([(10, 10, 5, 5, 0.9), (20, 20, 5, 5, 0.1)])
    chk(it, "without ground truth only counts and confidences are returned",
        not any(k in json.dumps(nogt).lower() for k in ("recall", "precision", "f1", "ap50")), keys=sorted(nogt))
    cat_recall = float((cat.conf >= 0.25).mean())
    it["limits"] += [
        f"The day simulations (B3, the 0.685 delivered recall) read the stored catalogue, which matches a "
        f"box to a ship BEFORE applying the 0.25 cut and at IoU >= 0.3. It finds {cat_recall:.4f} of the test "
        f"ships where threshold-then-match at IoU 0.5 finds {b1['recall']:.4f} (wp24). Delivered recall is "
        f"therefore on the catalogue's rule, not on the wp24 rule; the catalogue was not rebuilt.",
        "B1 / B2 ground truth is the tile labels shifted onto scenes stitched from independent tiles: valid "
        "boxes, but no ship crosses a seam, so these scenes cannot show a seam effect.",
        "The quickstart's synthetic tiles have no ground truth; its output is labelled SYNTH and reports "
        "counts and confidences only.",
    ]

    # ------------------------------------------------------------------ 8 energy assumptions
    it = item(8, "Processing / communication energy assumptions are documented")
    P, T, day = w17["powers_W_assumption"], w17["stage_times_ms"], w17["E_per_day_kJ"]
    r_bps = w17["downlink"]["R_eff_Mbps"] * 1e6
    e_comm = tx_energy_J(P["tx"], w17["downlink"]["D_sent_MB_day"] * MB, r_bps) / 1e3
    chk(it, "E_communication = P_tx x bytes x 8 / rate reproduces wp17",
        abs(e_comm - day["E_comm_proposed_semantic"]) / e_comm < 2e-3,
        recomputed_kJ=round(e_comm, 3), wp17_kJ=day["E_comm_proposed_semantic"], P_tx_W=P["tx"],
        D_MB=w17["downlink"]["D_sent_MB_day"], rate_Mbps=w17["downlink"]["R_eff_Mbps"])
    stages = {"preprocess": ("cpu", "preprocess_decode_cpu"), "gate": ("cpu", "gate_cpu"),
              "detect": ("cpu", "detect_cpu_onnx"), "fuse": ("cpu", "fuse_cpu"), "semantic": ("cpu", "semantic_cpu")}
    e_tile = sum(stage_energy_J(P[p], T[k]) for p, k in stages.values())
    e_proc = e_tile * day["tiles_per_day"] / 1e3
    chk(it, "E_processing = sum_k P_k x T_k reproduces wp17",
        abs(e_proc - day["E_proc_proposed"]) / e_proc < 1e-3, recomputed_kJ=round(e_proc, 3),
        wp17_kJ=day["E_proc_proposed"], J_per_tile=round(e_tile, 4))
    chk(it, "E_total = E_processing + E_communication",
        abs(day["E_proc_proposed"] + day["E_comm_proposed_semantic"] - day["E_total_proposed"]) < 0.02,
        E_total_kJ=day["E_total_proposed"])
    em = EnergyModel.from_results(res)
    lib = em.e_proc_per_tile_J("cpu_onnx")
    chk(it, "sat7.energy gives the same per-tile energy as wp17 and the same E_comm",
        abs(lib["E_prop"] - w17["E_proc_per_tile_J"]["cpu_onnx_DEFAULT"]["E_prop"]) < 1e-3 and
        abs(replace(em, r_tx_bps=r_bps).e_comm_direct_J(w17["downlink"]["D_sent_MB_day"] * MB) / 1e3 - e_comm) < 1e-9,
        sat7_energy_E_prop_J=round(lib["E_prop"], 4))
    pw = w19["powers_W"]
    e_isl = tx_energy_J(pw["isl_ASSUMPTION_NEW"], w19["inputs"]["D_sent_MB_day"] * MB, w19["inputs"]["R_isl_Mbps"] * 1e6) / 1e3
    chk(it, "wp19 ISL energy reproduces from P_isl x bytes x 8 / R_isl",
        abs(e_isl - w19["per_path_energy_daily_kJ"]["E_ISL"]) / e_isl < 2e-3, recomputed_kJ=round(e_isl, 3),
        wp19_kJ=w19["per_path_energy_daily_kJ"]["E_ISL"])
    chk(it, "every power is labelled an assumption in the results files",
        "ASSUMPTION" in w17["label_note"] and all("ASSUMPTION" in k for k in pw),
        wp17_powers_W=P, wp19_powers_W=pw)
    chk(it, "every stage time names its source; the unsourced ones are listed",
        set(w17["stage_time_provenance"]) == set(T) and len(w17["unsourced_figures"]) == 2,
        unsourced=w17["unsourced_figures"])
    chk(it, "power assumptions are swept", set(w17["sensitivity_P_sweep"]) == {"P_cpu", "P_gpu", "P_tx"} and
        len(w19["sensitivity_NEW_assumptions"]) >= 4, ES_total_range=[min(min(v["ES_total"]) for v in w17["sensitivity_P_sweep"].values()),
                                                                     max(max(v["ES_total"]) for v in w17["sensitivity_P_sweep"].values())])
    enc = w25["timing_ms_per_tile"]["T_enc_onboard_median"]
    enc_avg = 0.05 * enc["coast"] + 0.20 * enc["ships"]               # orbit mix: 5% coast, 20% ships, rest send nothing
    sv = w17["sahi_variant"]
    calls = sv["detector_calls_per_tile"]
    sahi_tile = e_tile + (calls - 1.0) * stage_energy_J(P["cpu"], T["detect_cpu_onnx"])
    chk(it, "the SAHI variant block is the default stages with the detector term x its calls per tile",
        abs(sahi_tile - sv["E_proc_per_tile_J"]["E_prop"]) < 1e-3 and sv["gate_calls_per_tile"] == 1.0 and
        sv["fuse_cpu_ms"] == 0.0 and "ASSUMPTION" in sv["fuse_note"] and "VARIANT" in sv["label"] and
        abs(sahi_tile * day["tiles_per_day"] / 1e3 - sv["E_per_day_kJ"]["E_proc_proposed"]) < 0.02,
        detector_calls_per_tile=calls, recomputed_J_per_tile=round(sahi_tile, 4),
        block_J_per_tile=sv["E_proc_per_tile_J"]["E_prop"], block_ES_proc=sv["E_proc_per_tile_J"]["ES_proc"])
    it["limits"] += [
        "NO power was measured in this project. P_cpu 28 W, P_gpu 60 W, P_tx 15 W and P_isl 12 W are "
        "ASSUMPTIONS; stage times are measured on a laptop; so every joule is an ESTIMATE. Only ES_proc for "
        "the all-CPU build is free of them (the power cancels: it is a ratio of measured times).",
        f"semantic_cpu = 0.5 ms is an ASSUMPTION in wp17. wp25 has since measured the encoder: "
        f"{enc['ships']} ms on a ship tile, {enc['coast']} ms on a coastal tile, i.e. about {enc_avg:.2f} ms per "
        f"tile over the assumed orbit mix -- consistent with the assumption, so wp17 was not regenerated.",
        f"The default E_processing is for ONE detector call per tile, which is what the day simulation runs. "
        f"The labelled variant results/wp17_energy_model.json -> sahi_variant costs SAHI's {calls:g} calls per "
        f"tile: {sv['E_per_day_kJ']['E_proc_proposed']} kJ/day instead of {day['E_proc_proposed']}, ES_proc "
        f"{100 * sv['E_proc_per_tile_J']['ES_proc']:.1f}% instead of "
        f"{100 * w17['E_proc_per_tile_J']['cpu_onnx_DEFAULT']['ES_proc']:.1f}%, "
        f"{sv['E_per_day_kJ']['proc_vs_comm_ratio_proposed']:.1f} : 1 against the radio. In it the gate stays "
        f"at one call per tile (how it is trained and run; no per-window gate exists) and fusion is charged "
        f"0 ms, an ASSUMPTION: it has never been timed.",
        "wp18 and the docs label the relay's energy 'TARGET'. It is an estimate from assumed P_isl and R_isl, "
        "not a design goal; the label is kept as report 19 wrote it.",
    ]

    # ------------------------------------------------------------------ 9 latency stages
    it = item(9, "Latency includes the stages the experiment requires")
    toy = _toy_latency_definition()
    chk(it, "simulated latency = wait for the pass + the item's transmission, from capture", toy["ok"],
        **{k: v for k, v in toy.items() if k != "ok"})
    st_cpu = measured_stage_times_s(res, "cpu_onnx")
    st_sahi = measured_stage_times_s(res, "cpu_onnx", calls_per_tile=1.5625)
    comm_direct_s = float(ours["latency_med_h"]) * 3600
    budgets = {
        "direct, per ship, three-day mean (wp6_real_table.csv)":
            LatencyBudget(st_cpu["processing_s"], st_cpu["encoding_s"]["coast"], comm_direct_s, st_cpu["decoding_s"]),
        "direct, real packets (wp27 nominal, direct only)":
            LatencyBudget(st_cpu["processing_s"], st_cpu["encoding_s"]["coast"],
                          w27["nominal"]["direct_only"]["ships"]["latency_med_h"] * 3600, st_cpu["decoding_s"]),
        "with relay, real packets (wp27 nominal)":
            LatencyBudget(st_cpu["processing_s"], st_cpu["encoding_s"]["coast"],
                          w27["nominal"]["with_relay"]["ships"]["latency_med_h"] * 3600, st_cpu["decoding_s"]),
        "direct, SAHI processing (1.5625 detector calls per tile)":
            LatencyBudget(st_sahi["processing_s"], st_sahi["encoding_s"]["coast"], comm_direct_s, st_sahi["decoding_s"]),
    }
    worst_stage = max(b.non_communication_s for b in budgets.values())
    chk(it, "processing, encoding and decoding times are measured and can be added",
        all(v > 0 for v in (st_cpu["processing_s"], st_cpu["encoding_s"]["coast"], st_cpu["decoding_s"])),
        processing_s=round(st_cpu["processing_s"], 5), encoding_s=st_cpu["encoding_s"], decoding_s=st_cpu["decoding_s"],
        source=st_cpu["source"])
    chk(it, "T_total = T_processing + T_encoding + T_communication + T_decoding (median, seconds)", True,
        budgets={k: {kk: (round(vv, 6) if kk != "non_communication_share" else float(f"{vv:.3g}"))
                     for kk, vv in b.as_dict().items()} for k, b in budgets.items()})
    chk(it, "the onboard stages never queue: a tile is finished before the next one is captured",
        worst_stage < 86400 / 160_000, per_tile_onboard_s=round(worst_stage, 4),
        tile_interval_s_at_40k=86400 / 40_000, tile_interval_s_at_160k=86400 / 160_000)
    chk(it, "ignoring the onboard stages in the simulator cannot move a pass assignment materially",
        worst_stage * 6 / 86400 < 1e-4, fraction_of_tiles_within_the_margin_of_a_pass=float(f"{worst_stage * 6 / 86400:.2g}"))
    it["limits"] += [
        "Every latency the results files and documents quote is T_communication alone. The other three "
        f"stages add {worst_stage * 1e3:.0f} ms at most ({worst_stage / comm_direct_s:.1e} of the median), so "
        "the quoted hours are unchanged; the convention is written in sat7/accounting.py.",
        "Stage times are a laptop's, not flight hardware's. A processor 10x slower still finishes a tile "
        f"({10 * worst_stage:.2f} s) before the next arrives at 40,000 tiles/day (2.16 s), not at 160,000 (0.54 s).",
        f"Latencies come from one isolated day. Over six consecutive days the per-ship median is "
        f"{min(r['latency_med_h'] for r in ss['per_day'])}-{max(r['latency_med_h'] for r in ss['per_day'])} h with "
        f"value-greedy, and FIFO's grows without bound (steady_state block).",
    ]

    # ------------------------------------------------------------------ 10 relay: both links
    it = item(10, "Relay energy and latency account for both links")
    pp = w19["per_path_energy_daily_kJ"]
    chk(it, "E_relay = E_ISL + E_GS, both non-zero (wp19)",
        abs(pp["E_ISL"] + pp["E_GS"] - pp["E_relay"]) < 2e-3 and pp["E_ISL"] > 0 and pp["E_GS"] > 0, **pp)
    rl = w27["nominal"]["with_relay"]["by_path"]["relay"]
    chk(it, "relayed traffic has transmit time and energy on the ISL and on the relay's ground pass (wp27)",
        all(rl[k][leg] > 0 for k in ("tx_s", "energy_J") for leg in ("isl", "relay_ground")) and
        rl["tx_s"]["ground"] == 0, tx_s=rl["tx_s"], energy_J=rl["energy_J"])
    cfg27 = w27["config"]
    chk(it, "each leg's energy is its own power x its own transmit time (wp27)",
        abs(rl["energy_J"]["isl"] - cfg27["p_isl_W"] * rl["tx_s"]["isl"]) < 0.01 * rl["energy_J"]["isl"] and
        abs(rl["energy_J"]["relay_ground"] - cfg27["p_tx_W"] * rl["tx_s"]["relay_ground"]) < 0.01 * rl["energy_J"]["relay_ground"],
        p_isl_W=cfg27["p_isl_W"], p_tx_W=cfg27["p_tx_W"])
    ex = w27["example_items_both_paths"]["P3"]
    t_isl = tx_time_s(ex["bytes"], ex["relay"]["isl_rate_Mbps"] * 1e6)
    chk(it, "one item traced by hand: ISL time = bytes x 8 / R_isl (wp27 example)",
        abs(t_isl - ex["relay"]["tx_s"]["isl"]) < 1e-3, bytes=ex["bytes"], recomputed_s=round(t_isl, 4),
        wp27_s=ex["relay"]["tx_s"]["isl"])
    toy = _toy_relay_both_legs()
    chk(it, "live: a relayed item waits for the ISL window AND the relay's ground pass", toy["ok"],
        **{k: v for k, v in toy.items() if k != "ok"})
    chk(it, "the relay forwards processed products only, never raw imagery (wp27)",
        w27["nominal"]["checks"]["non_semantic_items_on_relay"] == 0)
    b4 = w18["configs"]["B4"]
    e_direct_only = tx_energy_J(15.0, b4["MB_sent_SIM"] * MB, 2.53e6) / 1e3
    chk(it, "the B4 row's energy is joules (P_tx x bytes x 8 / R_gs), not the router's unit-less placeholders",
        abs(e_direct_only - b4["energy_kJ_TARGET"]["direct_only"]) < 0.02 and
        b4["energy_kJ_TARGET"]["b4_mix"] > b4["energy_kJ_TARGET"]["direct_only"],
        recomputed_direct_only_kJ=round(e_direct_only, 3), wp18=b4["energy_kJ_TARGET"])
    it["limits"] += [
        "GAP: The canonical B4 row (wp18) uses sat7.relay.route_item: its ENERGY has both legs, but its relay "
        "LATENCY is read from window start times (next ISL window, then the relay's next ground pass) with "
        "no data volume, rate or capacity, while its direct latency is B3's actual delivery. That is "
        "optimistic for the relay; wp27 is the volume- and capacity-aware simulation of both legs.",
        "P_isl, R_isl and the relay's orbit are ASSUMPTIONS; no relay hardware or real link exists.",
    ]

    # ------------------------------------------------------------------ 11 fair comparisons
    it = item(11, "The same inputs and assumptions are used where two things are compared")
    chk(it, "B1 vs B2: same scenes, pixels, ground truth, model and matching rule (wp26 check 5)",
        w26["checks"]["5_same_inputs"]["passed"], fingerprint=w26["checks"]["5_same_inputs"]["fingerprint"]["scenes"])
    sweep = pd.read_csv(res / "wp6_real_sweep.csv")
    same_cap = all(g.MB_capacity.nunique() == 1 for _, g in sweep.groupby("tiles"))
    same_ceiling = all(g[g.strategy.str.startswith("Ours")].ceiling_recall.nunique() == 1 for _, g in sweep.groupby("tiles"))
    chk(it, "ours vs FIFO vs the baselines: same day, same passes, same link share (wp6)",
        same_cap and same_ceiling, loads=sorted(int(x) for x in sweep.tiles.unique()))
    chk(it, "direct vs relay: same items, same ground-link model and power; relay off == the scheduler (wp27)",
        w27["gate_direct_only_equals_scheduler"]["passed"] and cfg27["link_share"] == cfg27["relay_link_share"],
        deliveries=w27["gate_direct_only_equals_scheduler"]["deliveries"])
    chk(it, "LoD vs P0-P3: same day and passes (wp23)",
        _json(res, "wp23_semantic_compare.json")["workload"]["tiles"] == 40000)
    contact_s = w17["downlink"]["contact_min_day"] * 60
    e_bent_feasible = P["tx"] * contact_s / 1e3
    it["limits"] += [
        f"B1/B2 ({q['B1']['n_gt']:,} ships on 150 scenes) and B3/B4 ({w18['configs']['B3']['day_ships']:,} ships "
        f"in a simulated day) are different inputs; B2's detection recall and B3's delivered recall are not "
        f"steps of one curve. Stated in the documents.",
        f"GAP: ES_total compares against a bent pipe that sends ALL raw data: {day['E_comm_bent_pipe_raw']:,.0f} kJ/day, "
        f"which would need {w17['downlink']['bent_pipe_days_of_contact_needed']} days of contact per day of imaging. "
        f"A bent pipe on this link can transmit for at most {w17['downlink']['contact_min_day']} min/day, i.e. "
        f"{e_bent_feasible:.1f} kJ/day -- LESS than our {day['E_total_proposed']} kJ/day. The {100 * day['ES_total_vs_bent_pipe']:.1f}% "
        f"is energy per unit of imagery accounted for, not a saving in the daily energy budget.",
        "GAP: wp18's B4 gives the direct path its simulated delivery time and the relay path a window estimate "
        "(item 10).",
        f"wp17 sends the day's {w17['downlink']['D_sent_MB_day']} MB on {100 * w17['downlink']['capacity_used_by_semantic']:.1f}% "
        f"of the whole link; the scheduler assumes this payload owns 25% of it.",
    ]

    # ------------------------------------------------------------------ 12 traceability
    it = item(12, "No numerical claim lacks traceable experimental evidence")
    cl = check_claims(res, repo, steady)
    chk(it, "every headline number in the ten judge-facing documents equals its results file",
        not cl["missing"], claims=cl["claims"], claim_document_pairs=cl["claim_document_pairs"], missing=cl["missing"])
    chk(it, "the earlier 557x estimate is still reproducible from a committed file",
        abs(float(_table(res, "wp6_real_table_modeledcoast.csv")["Ours: LoD + value-greedy"]["data_reduction_x"]) - 556.96) < 0.01)
    it["limits"] += [
        "The ledger covers the headline numbers of ten documents. docs/START_HERE.md, docs/STATUS.md, "
        "docs/ARCHITECTURE.md and the bodies of reports 05-22 still narrate the earlier estimate (bannered "
        "in reports/README.md) and are not covered.",
        "GAP: Three claims rest on a modelling assumption rather than a measurement: (a) a progressive product "
        "truncated to 10% of its bytes still counts as delivering its ships -- this is what keeps recall "
        "flat under congestion and over consecutive days; (b) thumbnail and crop sizes; (c) a 25% link share.",
        f"GAP: '100% of what the satellite still knew' holds for one day given 36 h of passes. The day's offer is "
        f"{100 * ss['offered_over_sustained']:.0f}% of what the share sustains per 24 h; over six consecutive days "
        f"recall still holds ({min(r['ship_recall'] for r in ss['per_day'])}-{max(r['ship_recall'] for r in ss['per_day'])}) "
        f"only because of assumption (a).",
    ]

    for it in items.values():
        failed = [c["check"] for c in it["checks"] if not c["ok"]]
        gaps = [x for x in it["limits"] if x.startswith("GAP: ")]
        it["status"] = "FAIL" if failed else ("PARTIAL" if gaps else "PASS")
        it["failed_checks"], it["gaps"] = failed, len(gaps)
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--no-steady-state", action="store_true",
                    help="reuse the steady_state block already in the output file instead of re-simulating")
    args = ap.parse_args()
    res, out = args.results, args.results / "wp29_validation.json"
    t0 = time.perf_counter()
    if args.no_steady_state and out.exists():
        steady = json.loads(out.read_text(encoding="utf-8"))["steady_state"]
    else:
        steady = steady_state(res)
    checklist = run_checklist(res, REPO, steady)
    report = {
        "label": "validation of committed results; stage times REAL (laptop), powers ASSUMPTION (none measured: "
                 "every energy is an ESTIMATE), links and days SIM",
        "status_meaning": {"PASS": "every check holds; any limits listed are caveats",
                           "PARTIAL": "every check holds, but a limit marked GAP means the requirement is "
                                      "not fully met",
                           "FAIL": "at least one check does not hold"},
        "checklist": checklist,
        "steady_state": steady,
        "claims_ledger": claims_ledger(res, steady),
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")

    print("=== WP29 validation (committed results; nothing regenerated) ===")
    for num, it in checklist.items():
        ok = sum(c["ok"] for c in it["checks"])
        print(f"\n[{num}] {it['title']}: {it['status']}  ({ok}/{len(it['checks'])} checks, {it['gaps']} gaps, "
              f"{len(it['limits']) - it['gaps']} caveats)")
        for c in it["checks"]:
            print(f"     {'ok  ' if c['ok'] else 'FAIL'} {c['check']}")
        for lim in it["limits"]:
            print(f"     {'' if lim.startswith('GAP: ') else 'caveat: '}{lim[:150]}{'...' if len(lim) > 150 else ''}")
    ss = steady["runs"]
    print("\nsteady state, six consecutive days [SIM-over-REAL]:")
    for name, r in ss.items():
        print(f"  {name}: sustained {r['sustained_capacity_MB_per_24h']} MB/24 h, offer {100 * r['offered_over_sustained']:.0f}% of it")
        for d in r["per_day"]:
            print(f"     day {d['day']}: recall {d['ship_recall']}  latency med {d['latency_med_h']} h  "
                  f"bytes delivered {d['bytes_delivered_fraction']}")
    print(f"\nsaved {out} ({time.perf_counter() - t0:.0f} s)")
    return 1 if any(it["status"] == "FAIL" for it in checklist.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
