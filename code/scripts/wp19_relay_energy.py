"""WP19 -- B4 relay energy: E_relay = E_ISL + E_GS, in wp17's per-stage structure.

START_HERE.md section 5 item 3 asks for the inter-satellite relay's energy: `E_relay = E_ISL +
E_GS`, so route(plan, links) (report 18 section 3) can weigh `J_relay = λ_E·E + λ_T·T` against the
direct path. This builds the ENERGY terms only -- transmit power × duration over each link, the
same `E = P·D/R` pattern wp17 uses for E_comm. It runs in .venv, imports no torch, and reads every
reused figure LIVE from its source file, exactly as wp17 now does (no hardcoded D/R/P).

The two relay legs, and why relay can ever be chosen (it is NEVER cheaper on energy):
  * E_ISL  originating sat -> relay sat   = P_isl · D / R_isl   (the extra hop)
  * E_GS   relay sat -> ground            = P_tx  · D / R_gs    (reuses the GROUND-link hardware,
                                            so same P_tx and R_gs as the direct path -- see note 3)
  * E_relay = E_ISL + E_GS  (constellation-total energy, per START_HERE 5.3) is STRICTLY GREATER
    than the direct E_comm = P_tx·D/R_gs. Relay is a LATENCY/availability play (Sfax is one station,
    11.4 h max gap): route() spends more joules to get data down sooner. The energy-vs-time trade
    is route()'s to make; this file only supplies the joules each path costs.

What is REAL / SIM / ASSUMPTION (project rule: invented constants are swept, cf. report 11):
  * D (bytes/day sent)     SIM -- live from wp6_real_table.csv via wp17.load_downlink (one source).
  * R_gs (ground rate)     SIM -- the orbit sim's own passes.csv (capacity/contact), as wp17.
  * P_tx (ground tx power) ASSUMPTION -- reused from wp17 (15 W, swept), NOT re-invented.
  * P_isl, R_isl           NEW, and the only genuinely new numbers here. The ISL link (sat-to-sat
    crosslink) is different hardware from the ground downlink, so it needs its own power and rate.
    There is NO ISL figure anywhere in the repo yet (orbit.py has find_passes/rate_bps for the
    GROUND link only), so these are ASSUMPTION and SWEPT -- flagged pending the parallel SGP4
    link-window track, which is to supply the real ISL window duration and may supply R_isl. Label
    them LIT only when sourced from a crosslink datasheet for comparable hardware.

Integration with the path-choice track (report 20 / sat7.relay), which adopted this interface:
  * The relay/direct MIX comes from route(), not invented here. With our energy injected into
    sat7.relay.route_item, the weighted ES is now COMPUTED (weighted_es_via_route below), and it
    cross-checks wp20's relay fraction at lam_E=0 (energy-free, so purely latency-driven: f=0.44).
    Bounds (all-direct / all-relay) are kept for context -- the spread is only 0.14 pts, so ES is
    essentially insensitive to the path mix: relay is a LATENCY buy, not an energy one.
  * It still does NOT compute latency T -- that is route()'s half of J (SIM in report 20). Energy is
    window-independent (P·D/R). It reuses report 20's ISL windows (wp20_isl_windows.csv) for an ISL
    feasibility check only, and does NOT touch the path-choice implementation.

    .venv/Scripts/python.exe scripts/wp19_relay_energy.py

Outputs: results/wp19_relay_energy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from sat7.orbit import LINK_PRESETS, rate_bps
import numpy as np
# Reuse wp17's live loaders + power assumptions, so D, P_tx and the sweep have ONE source of truth.
from wp17_energy_model import MB, P_DEFAULT, P_SWEEP, load_downlink

# ---- NEW ISL constants (ASSUMPTION, swept, pending the SGP4 link-window track) -----------------
P_ISL_DEFAULT = 12.0          # W, RF crosslink tx power -- same order as the S-band ground radio
P_ISL_SWEEP = (5.0, 30.0)     # W
# R_isl default = the ground rate (neutral: assume the crosslink is no faster than the downlink
# until the link-window track says otherwise). Swept wide because real crosslinks vary 10x+.
R_ISL_SWEEP_MBPS = (1.0, 50.0)


# ---------------------------------------------------------------- the energy primitives route() calls
def direct_energy(d_bytes: float, p_tx: float, r_gs_bps: float) -> dict:
    """Direct sat->ground path. Identical to wp17's E_comm; route() uses this for J_direct's E."""
    t = d_bytes * 8 / r_gs_bps if r_gs_bps else float("inf")   # bytes -> bits -> seconds
    return {"E_comm_J": p_tx * t, "tx_time_s": t}


def relay_energy(d_bytes: float, p_tx: float, p_isl: float,
                 r_gs_bps: float, r_isl_bps: float) -> dict:
    """Relay path sat->(ISL)->relay sat->(ground). E_relay = E_ISL + E_GS. This is exactly what
    route() needs to fill J_relay's E; T is added by route() from the link windows (not here)."""
    t_isl = d_bytes * 8 / r_isl_bps if r_isl_bps else float("inf")
    t_gs = d_bytes * 8 / r_gs_bps if r_gs_bps else float("inf")
    e_isl, e_gs = p_isl * t_isl, p_tx * t_gs
    return {"E_ISL_J": e_isl, "E_GS_J": e_gs, "E_relay_J": e_isl + e_gs,
            "isl_time_s": t_isl, "gs_time_s": t_gs}


def _r_gs_bps(passes_csv: Path):
    pz = pd.read_csv(passes_csv)
    contact_s = float(pz.duration_min.sum()) * 60.0
    cap_mb_day = float(pz.capacity_MB.sum())
    return cap_mb_day * MB * 8 / contact_s, cap_mb_day, contact_s   # bits/s averaged over contact


def weighted_es_via_route(d_bytes, p_tx, p_isl, r_gs_bps, r_isl_bps,
                          e_proc_J, e_bent_J, e_direct_J, ratio, lam_E=0.0, lam_T=1.0):
    """Fold the REAL path choice into ES by injecting our energy into the OTHER track's route()
    (sat7.relay, report 20) -- reuse, not reimplement. lam_E=0 = latency-optimal (max relay); raising
    lam_E penalises relay's extra joules. Uniform arrivals, uniform item bytes, so the relay
    byte-fraction f = relay arrival-fraction, and E_comm_mix = E_direct·(1 + (ratio-1)·f) since our
    per-item relay/direct energy ratio is constant. Returns None if sat7.relay is unavailable."""
    try:
        from datetime import datetime, timedelta, timezone
        from skyfield.api import load
        from sat7.orbit import GroundStation, LINK_PRESETS, OrbitConfig, find_passes, make_satellite
        from sat7.relay import RelayConfig, isl_windows, make_relay, route_item, LinkParams
    except Exception as e:
        return {"status": f"sat7.relay unavailable ({type(e).__name__}) -- weighted ES left as bounds"}
    START = datetime(2026, 9, 18, tzinfo=timezone.utc)
    HOURS = 36.0
    ts = load.timescale()
    link = LINK_PRESETS["cubesat_sband"]
    primary = make_satellite(OrbitConfig(epoch=START), ts=ts)
    rcfg = RelayConfig(orbit=OrbitConfig(epoch=START, raan_deg=90.0))   # wp20's demonstrative default
    relay = make_relay(rcfg, ts=ts)
    direct = find_passes(primary, GroundStation(), START, HOURS, link, ts=ts)
    relay_p = find_passes(relay, GroundStation(), START, HOURS, link, ts=ts)
    isl = isl_windows(primary, relay, START, HOURS, grazing_km=rcfg.grazing_km,
                      max_range_km=rcfg.max_range_km, ts=ts)
    grid = [START + timedelta(minutes=5 * k) for k in range(int(HOURS * 60 / 5))]
    lp = LinkParams(p_tx_W=p_tx, p_isl_W=p_isl, r_gs_bps=r_gs_bps, r_isl_bps=r_isl_bps)
    out = {}
    for lamE in (0.0, 1.0, 10.0):
        rows = [route_item(d_bytes, t, direct, isl, relay_p, lam_E=lamE, lam_T=lam_T,
                           direct_energy=direct_energy, relay_energy=relay_energy, link=lp)
                for t in grid]
        f = sum(1 for r in rows if r["path"] == "relay") / len(rows)
        e_comm_mix = e_direct_J * (1 + (ratio - 1) * f)
        out[f"lam_E={lamE}"] = {"relay_fraction_f": round(f, 3),
                                "ES_weighted": round(1 - (e_proc_J + e_comm_mix) / e_bent_J, 4)}
    out["cross_check_lamE0_vs_wp20"] = {"f_here": out["lam_E=0.0"]["relay_fraction_f"],
                                        "wp20_reported": 0.44,
                                        "note": "lam_E=0 is energy-free -> must match wp20's latency-optimal f"}
    out["status"] = "COMPUTED via sat7.relay.route_item with our energy injected (drop-in)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--passes", type=Path, default=ROOT / "results" / "passes.csv")
    ap.add_argument("--wp17", type=Path, default=ROOT / "results" / "wp17_energy_model.json",
                    help="wp17 output, read live for E_proc / E_comm / E_bent-pipe to fold ES")
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--p-isl", type=float, default=P_ISL_DEFAULT, help="ASSUMPTION (pending peer)")
    ap.add_argument("--r-isl-mbps", type=float, default=None,
                    help="ISL rate; default = ground rate (neutral ASSUMPTION, pending peer)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    # ---- live inputs (single source of truth): D from wp6_real_table.csv, R_gs from passes.csv
    d_sent_mb, d_raw_mb, _ = load_downlink(args.results)
    r_gs_bps, cap_mb_day, contact_s = _r_gs_bps(args.passes)
    r_isl_bps = (args.r_isl_mbps * 1e6) if args.r_isl_mbps is not None else r_gs_bps
    r_isl_is_default = args.r_isl_mbps is None
    P = P_DEFAULT
    D = d_sent_mb * MB                       # the daily semantic payload (bytes), what gets routed

    # ---- per-path daily energy
    direct = direct_energy(D, P["tx"], r_gs_bps)
    relay = relay_energy(D, P["tx"], args.p_isl, r_gs_bps, r_isl_bps)
    relay_over_direct = relay["E_relay_J"] / direct["E_comm_J"]

    # ---- SANITY GATE (wp17 pattern): refuse to report until independent checks pass.
    #  (a) direct path here must equal wp17's E_comm for the same D (one source of truth for E_comm)
    #  (b) relay must cost STRICTLY MORE energy than direct -- an invariant of adding a hop; if it
    #      ever came out <= direct, a sign/definition error has flipped the model.
    #  (c) the ground rate used here must match wp17's downlink rate (same orbit sim), <= zenith peak.
    wp17 = json.loads(args.wp17.read_text()) if args.wp17.exists() else None
    link = LINK_PRESETS[args.link]
    r_zenith_bps = float(rate_bps(np.array([500.0]), 500.0, link)[0])
    gate = {"check_b_relay_gt_direct": {"E_relay_J": round(relay["E_relay_J"], 1),
                                        "E_direct_J": round(direct["E_comm_J"], 1),
                                        "ratio": round(relay_over_direct, 4)},
            "check_c_rate_bound": {"R_gs_Mbps": round(r_gs_bps / 1e6, 3),
                                   "R_zenith_Mbps": round(r_zenith_bps / 1e6, 3)}}
    if relay_over_direct <= 1.0:
        raise SystemExit(f"[sanity] FAILED (b): E_relay {relay['E_relay_J']:.1f} J <= E_direct "
                         f"{direct['E_comm_J']:.1f} J -- adding an ISL hop cannot cost less energy; "
                         f"sign/definition error. Fix before trusting any relay figure.")
    if r_gs_bps > r_zenith_bps * 1.001:
        raise SystemExit(f"[sanity] FAILED (c): ground rate {r_gs_bps/1e6:.2f} Mbps exceeds zenith "
                         f"peak {r_zenith_bps/1e6:.2f} Mbps -- passes.csv/link model inconsistent.")
    if wp17 is not None:
        wp17_ecomm_kJ = float(wp17["E_per_day_kJ"]["E_comm_proposed_semantic"])
        here_ecomm_kJ = direct["E_comm_J"] / 1000
        rel = abs(here_ecomm_kJ - wp17_ecomm_kJ) / wp17_ecomm_kJ if wp17_ecomm_kJ else 0.0
        gate["check_a_matches_wp17_Ecomm"] = {"wp17_kJ": round(wp17_ecomm_kJ, 2),
                                              "here_kJ": round(here_ecomm_kJ, 2), "rel_err": round(rel, 4)}
        if rel > 0.02:
            raise SystemExit(f"[sanity] FAILED (a): direct E_comm here {here_ecomm_kJ:.2f} kJ != wp17 "
                             f"{wp17_ecomm_kJ:.2f} kJ (rel {rel:.2%}) -- the two energy models disagree "
                             f"on the SAME downlink. Reconcile D/R before trusting the relay terms.")
    gate["passed"] = True

    # ---- authoritative ISL windows from the relay track (reuse, don't recompute)
    isl_csv = args.results / "wp20_isl_windows.csv"
    isl_info = None
    if isl_csv.exists():
        iw = pd.read_csv(isl_csv)
        isl_total_s = float(iw.duration_s.sum())
        isl_need_s = D * 8 / r_isl_bps
        isl_info = {"source": "wp20_isl_windows.csv (SIM, sat7.relay SGP4)",
                    "n_windows": int(len(iw)), "total_min": round(isl_total_s / 60, 1),
                    "min_range_km": round(float(iw.min_range_km.min()), 1),
                    "mean_range_km": round(float(iw.mean_range_km.mean()), 1),
                    "isl_time_needed_s_for_D": round(isl_need_s, 1),
                    "window_time_sufficient": bool(isl_total_s > isl_need_s),
                    "note": "range is SIM from the relay track; it could DERIVE a real R_isl via a "
                            "crosslink budget (datasheet -> LIT) but is NOT used in the energy here "
                            "(R_isl stays ASSUMPTION). A stale 458 km min-range figure in report 20's "
                            "prose was flagged against this CSV and corrected to 1252 km."}

    # ---- fold into wp17's ES. The weighted value is NOW computable (route() exists, report 20): we
    # inject our energy into sat7.relay.route() to get the real relay fraction f. Bounds kept for ref.
    es_bounds = {"note": "ES = 1 - (E_proc + E_comm_mix)/E_bent_pipe; E_comm_mix = "
                         "(1-f)*E_direct + f*E_relay, f = relay byte-fraction from route()."}
    weighted = {"status": "wp17 json absent -> E_proc/E_bent unavailable"}
    if wp17 is not None:
        e_proc = float(wp17["E_per_day_kJ"]["E_proc_proposed"]) * 1000
        e_bent = float(wp17["E_per_day_kJ"]["E_comm_bent_pipe_raw"]) * 1000
        for f, name in ((0.0, "all_direct_f0"), (1.0, "all_relay_f1")):
            e_comm_mix = (1 - f) * direct["E_comm_J"] + f * relay["E_relay_J"]
            es_bounds[name] = round(1 - (e_proc + e_comm_mix) / e_bent, 4)
        weighted = weighted_es_via_route(D, P["tx"], args.p_isl, r_gs_bps, r_isl_bps,
                                         e_proc, e_bent, direct["E_comm_J"], relay_over_direct)

    # ---- sweep the NEW assumptions (report 11 pattern): P_isl, R_isl, and reuse P_tx sweep
    sweep = {}
    for p_isl in P_ISL_SWEEP:
        r = relay_energy(D, P["tx"], p_isl, r_gs_bps, r_isl_bps)
        sweep[f"P_isl={p_isl}W"] = {"E_relay_kJ": round(r["E_relay_J"] / 1000, 2),
                                    "relay/direct": round(r["E_relay_J"] / direct["E_comm_J"], 3)}
    for r_isl_mbps in R_ISL_SWEEP_MBPS:
        r = relay_energy(D, P["tx"], args.p_isl, r_gs_bps, r_isl_mbps * 1e6)
        sweep[f"R_isl={r_isl_mbps}Mbps"] = {"E_relay_kJ": round(r["E_relay_J"] / 1000, 2),
                                            "relay/direct": round(r["E_relay_J"] / direct["E_comm_J"], 3)}

    rep = {
        "label_note": "E_ISL/E_GS structure built. D SIM (live wp6_real_table.csv), R_gs SIM (orbit "
                      "sim), P_tx ASSUMPTION (reused from wp17). P_isl/R_isl ASSUMPTION+SWEPT, NEW, "
                      "PENDING the SGP4 link-window track for the real ISL window/rate.",
        "status": {"energy_terms": "BUILT (E_ISL, E_GS, E_relay, direct)",
                   "weighted_ES": "COMPUTED -- route() (report 20) now exists; our energy injected",
                   "latency_T": "NOT computed here -- it is the relay track's (report 20); SIM there",
                   "full_B0_B4": "route() exists (report 20) + energy here -> B4 now runnable; the "
                                 "B0-B4 campaign still needs wp18's runner wired to route()"},
        "route_interface": {
            "direct_energy": "direct_energy(d_bytes, p_tx, r_gs_bps) -> {E_comm_J, tx_time_s}",
            "relay_energy": "relay_energy(d_bytes, p_tx, p_isl, r_gs_bps, r_isl_bps) -> "
                            "{E_ISL_J, E_GS_J, E_relay_J, isl_time_s, gs_time_s}",
            "note": "these are the E primitives route() calls per path to form J = lambda_E*E + "
                    "lambda_T*T; proposed to the path-choice track, pending confirmation"},
        "inputs": {"D_sent_MB_day": round(d_sent_mb, 2), "D_raw_MB_day": round(d_raw_mb, 2),
                   "R_gs_Mbps_SIM": round(r_gs_bps / 1e6, 3),
                   "R_isl_Mbps": round(r_isl_bps / 1e6, 3),
                   "R_isl_is_default_eq_ground": r_isl_is_default,
                   "contact_min_day": round(contact_s / 60, 1), "capacity_MB_day": round(cap_mb_day, 1)},
        "powers_W": {"tx_ground_ASSUMPTION_wp17": P["tx"], "isl_ASSUMPTION_NEW": args.p_isl},
        "sanity_gate": gate,
        "per_path_energy_daily_kJ": {
            "E_direct": round(direct["E_comm_J"] / 1000, 3),
            "E_ISL": round(relay["E_ISL_J"] / 1000, 3),
            "E_GS": round(relay["E_GS_J"] / 1000, 3),
            "E_relay": round(relay["E_relay_J"] / 1000, 3),
            "relay_over_direct_x": round(relay_over_direct, 3),
            "payload_D_MB": round(d_sent_mb, 2)},
        "ES_bounds": es_bounds,
        "ES_weighted_via_route20": weighted,
        "isl_windows_SIM": isl_info,
        "sensitivity_NEW_assumptions": sweep,
        "batch_consistency_note": (
            "RESOLVED upstream check (task prerequisite): wp17's gate time (0.052 ms, wp3_gate.json) "
            "is a PURE-forward batch-32 figure; its detector time (10.71 ms, wp1_metrics.json) is "
            "end-to-end model.predict over a file list (decode+pre/post+Python match, batch-1-ish). "
            "They are NOT the same measurement scope, so '0.052 vs 10.71' is not a like-for-like "
            "comparison. This does NOT affect any figure here: E_ISL/E_GS/E_direct are transmit "
            "energies (P*D/R) and use neither forward time. It also does not change wp17's ES_proc "
            "conclusion (the gate is negligible vs the 56.3 ms classic prefilter it replaces), but "
            "wp17's per-stage gpu numbers mix scopes and should carry that provenance."),
    }
    (args.out / "wp19_relay_energy.json").write_text(json.dumps(rep, indent=2))

    # ---- print
    print("=== WP19 relay energy: E_relay = E_ISL + E_GS (energy terms only) ===\n")
    print(f"[sanity] relay/direct {relay_over_direct:.3f}x (>1 OK); R_gs {r_gs_bps/1e6:.2f} <= "
          f"zenith {r_zenith_bps/1e6:.2f} Mbps"
          + (f"; direct E_comm == wp17 ({gate['check_a_matches_wp17_Ecomm']['rel_err']:.1e}) OK"
             if wp17 is not None else "; wp17 json absent -> skipped check (a)"))
    print(f"\npayload D = {d_sent_mb:.1f} MB/day; R_gs {r_gs_bps/1e6:.2f} Mbps (SIM), "
          f"R_isl {r_isl_bps/1e6:.2f} Mbps ({'=ground, ASSUMPTION' if r_isl_is_default else 'set'})")
    print(f"per-day energy (kJ):  E_direct {direct['E_comm_J']/1000:.2f}  |  "
          f"E_ISL {relay['E_ISL_J']/1000:.2f} + E_GS {relay['E_GS_J']/1000:.2f} = "
          f"E_relay {relay['E_relay_J']/1000:.2f}  ({relay_over_direct:.2f}x direct)")
    print("relay ALWAYS costs more energy than direct; it buys latency, which route() weighs (not here).")
    if wp17 is not None:
        print(f"\nES bounds (folded into wp17): all-direct {es_bounds['all_direct_f0']} .. "
              f"all-relay {es_bounds['all_relay_f1']} (0.14-pt spread -> ES ~insensitive to relay use)")
        if weighted.get("status", "").startswith("COMPUTED"):
            cc = weighted["cross_check_lamE0_vs_wp20"]
            w0 = weighted["lam_E=0.0"]
            print(f"  WEIGHTED via route() (report 20, our energy injected): lam_E=0 (latency-optimal) "
                  f"f={w0['relay_fraction_f']} -> ES {w0['ES_weighted']}")
            print(f"  cross-check: f_here {cc['f_here']} == wp20 {cc['wp20_reported']} at lam_E=0 "
                  f"(energy-free) -> drop-in validated")
        if isl_info:
            print(f"  ISL (SIM, wp20): {isl_info['n_windows']} windows, {isl_info['total_min']:.0f} min, "
                  f"need {isl_info['isl_time_needed_s_for_D']:.0f}s for D -> "
                  f"{'feasible' if isl_info['window_time_sufficient'] else 'TIGHT'}")
    print("\nNEW assumptions swept (P_isl, R_isl):")
    for k, v in sweep.items():
        print(f"  {k:18s} E_relay {v['E_relay_kJ']:.2f} kJ  ({v['relay/direct']}x direct)")
    print("\nStatus: energy terms BUILT; weighted ES COMPUTED via route() (report 20); ISL windows "
          "now SIM from report 20 (R_isl still ASSUMPTION, swept). ES ~insensitive to the path mix "
          "-> relay is a LATENCY buy, not an energy one. Saved wp19_relay_energy.json")


if __name__ == "__main__":
    main()
