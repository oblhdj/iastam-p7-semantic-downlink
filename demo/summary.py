"""IASTAM P7 -- demo summary.

Reads the authoritative result files and prints one presentation-ready account of the
whole system: the problem, the headline, the B0->B4 campaign, the end-to-end integrity
check, and the honesty labels. Pure standard library -- runs in ANY python, no GPU, no
torch, no re-computation. This is the "slide behind the demo": the numbers it prints are
the committed outputs in code/results/, not fresh ones.

    python demo/summary.py

If you want to regenerate those outputs live first, run demo/run_demo.sh (needs the GPU
venv). This script just reports whatever is currently in code/results/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "code" / "results"

RULE = "=" * 78
THIN = "-" * 78


def _load_json(name: str) -> dict | None:
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_table(name: str) -> list[dict]:
    p = RESULTS / name
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def _row(rows: list[dict], key: str) -> dict | None:
    for r in rows:
        if r.get("strategy") == key:
            return r
    return None


def _f(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main() -> None:
    table = _load_table("wp6_real_table.csv")
    camp = _load_json("wp18_campaign.json")
    integ = _load_json("wp11_integration.json")
    w0 = _load_json("wp0_stats.json")

    ours = _row(table, "Ours: LoD + value-greedy")
    fair = _row(table, "Phi-sat-2 style, fair (+coastal)")
    bent = _row(table, "Bent pipe (raw, FIFO)")

    print("\n" + RULE)
    print("  IASTAM 6.0  -  Problem 7  -  Semantic Downlink for Maritime Detection")
    print("  Transmitting INFORMATION, not raw data")
    print(RULE)

    # ---------------------------------------------------------------- 1. the problem
    print("\n[1] THE PROBLEM  (B0 = raw reference)")
    raw_mb = 60124.0
    if camp and "configs" in camp and "B0" in camp["configs"]:
        raw_mb = _f(camp["configs"]["B0"]["raw_volume_MB_day_at_40k"], raw_mb)
    sent_mb = _f(ours["MB_sent"]) if ours else 108.0
    print(f"    A small satellite images ~{raw_mb:,.0f} MB of ocean per day.")
    print(f"    Its radio link carries only a couple hundred MB per day.")
    print(f"    => ~99.7% of what it sees is thrown away.  The usual fix: compress harder.")

    # ---------------------------------------------------------------- 2. the headline
    print("\n[2] OUR HEADLINE  (B3 = full onboard pipeline)")
    if ours:
        red = _f(ours["data_reduction_x"])
        rec = _f(ours["ship_recall"])
        print(f"    Data reduction vs raw : {red:,.0f}x   ({raw_mb:,.0f} MB  ->  {sent_mb:.1f} MB / day)")
        print(f"    Ships delivered       : {rec:.3f}   "
              f"(!) at an ASSUMED 15% cloud -- band 0.40-0.82 over 0-50% cloud; never quote bare")
        if fair:
            dpts = (rec - _f(fair["ship_recall"])) * 100
            print(f"    vs fair Phi-sat-2 base: +{dpts:.1f} points of recall for ~7x the bytes")
        if bent:
            print(f"    vs bent-pipe FIFO     : {rec:.3f} vs {_f(bent['ship_recall']):.3f} "
                  f"at the same byte budget (raw pixels collapse the link)")

    # ---------------------------------------------------------------- 3. B0->B4 campaign
    print("\n[3] B0 -> B4 CAMPAIGN  (the progression the accepted paper promised)")
    if camp and "configs" in camp:
        c = camp["configs"]
        print(f"    {'cfg':<4}{'onboard':<34}{'recall':<9}{'label':<16}key number")
        print("    " + THIN[:74])
        def line(cfg, onboard, rec, label, note):
            rec_s = f"{rec:.3f}" if isinstance(rec, (int, float)) else str(rec)
            print(f"    {cfg:<4}{onboard:<34}{rec_s:<9}{label:<16}{note}")
        if "B0" in c:
            line("B0", "none (raw image)", "-", c["B0"].get("label", ""),
                 f"{_f(c['B0']['raw_volume_MB_day_at_40k']):,.0f} MB/day raw")
        if "B1" in c:
            line("B1", "detector", _f(c["B1"]["recall"]["overall"]), c["B1"].get("label", ""),
                 "YOLO on whole tiles")
        if "B2" in c:
            line("B2", "detector + SAHI + fusion", _f(c["B2"]["recall"]["overall"]),
                 c["B2"].get("label", ""), f"{_f(c['B2']['compute_vs_regular_tiling_x']):.2f}x compute")
        if "B3" in c:
            line("B3", "detector+gate+LoD+scheduler", _f(c["B3"]["ship_recall"]),
                 c["B3"].get("label", ""), f"{_f(c['B3']['reduction_vs_B0_x']):,.0f}x reduction, {_f(c['B3']['MB_sent']):.0f} MB")
        if "B4" in c:
            b4 = c["B4"]
            cut = _f(b4.get("worst_case_latency_cut_h"))
            dmax = _f(b4["latency_h_B3_direct"]["max"]); rmax = _f(b4["latency_h_B4_relay"]["max"])
            line("B4", "+ inter-satellite relay", _f(b4["ship_recall_SIMoverREAL"]),
                 "latency SIM", f"worst latency {dmax:.1f}h -> {rmax:.1f}h (reroute, recall=B3)")
        gates = c.get("B4", {}).get("sanity_gates", {})
        if gates:
            print(f"    sanity gates: {'ALL PASS' if gates.get('passed') else 'CHECK'} "
                  f"(B4 reproduces B3 recall/MB; relay only changes latency+energy)")
    else:
        print("    (wp18_campaign.json not found -- run demo/run_demo.sh to generate it)")

    # ---------------------------------------------------------------- 4. integrity check
    print("\n[4] END-TO-END INTEGRITY  (real JPEGs through the real chain, wp11)")
    if integ:
        cc = integ.get("catalogue_cross_check", {})
        flips = cc.get("decision_flips", {})
        print(f"    {integ.get('tiles', '?')} real tiles pushed through "
              f"prefilter -> gate -> detector -> LoD -> scheduler -> ground.")
        print(f"    Reproduces the stored catalogue WP6-WP10 replay: "
              f"context agreement {cc.get('context', {}).get('agree', '?')}, "
              f"decision flips @conf_high = {flips.get('conf_high', '?')}")
        sm = integ.get("stage_ms_per_tile", {})
        if sm:
            print(f"    Onboard cost / tile (laptop RTX 5060): "
                  f"prefilter {sm.get('prefilter', 0):.1f} ms, gate {sm.get('gate', 0):.2f} ms, "
                  f"detector {sm.get('detector', 0):.1f} ms")
        thin = integ.get("thin_context_pools", {})
        if thin:
            for ctx, d in thin.items():
                print(f"    (!) honesty: a '{ctx}' pool of {d['real_tiles']} real tiles is "
                      f"resampled {d['reuse_x']:.0f}x -- much of the cloud-ship loss rests on these")
    else:
        print("    (wp11_integration.json not found -- run demo/run_demo.sh to generate it)")

    # ---------------------------------------------------------------- 5. honesty
    print("\n[5] HOW TO READ THE NUMBERS  (defend these or lose points)")
    print("    - Every figure is labelled REAL / SIM / LIT / TARGET / ASSUMPTION.")
    print("    - ONBOARD timing is measured on a LAPTOP RTX 5060. 'Runs onboard' is a")
    print("      TARGET mapped to a flight processor [LIT], NOT a measured prototype.")
    print("    - Recall is always quoted WITH the cloud fraction (0.685 @ 15%; band 0.40-0.82).")
    print("    - Negative results are reported: INT8 rejected, classic gate failed,")
    print("      blanket SAHI a poor trade, our own first baseline was unfair.")
    if w0:
        print(f"    - Trained on {w0.get('selected_tiles', '?'):,} real Airbus tiles, "
              f"leakage-free split ({w0.get('groups_a_naive_split_would_leak', '?'):,} groups a naive split would leak, ours leaks 0).")

    print("\n" + RULE)
    print("  Full narrative: reports/  |  deep state: docs/STATUS.md  |  map: HANDOFF.md")
    print(RULE + "\n")


if __name__ == "__main__":
    main()
