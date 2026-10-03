"""WP20 -- B4 inter-satellite relay: ISL windows, path choice, and the direct-only sanity gate.

Implements the relay seam report 18 section 3 reserved (route(plan, links)) and START_HERE section
5 item 3, using sat7.relay (which reuses sat7.orbit's SGP4 propagator -- no new orbital mechanics).

What this answers NOW, and what it defers:
  * ISL (sat-to-relay) visibility windows and the LATENCY of direct vs relay delivery -- SIM, from
    the real orbit propagation. These are reportable today.
  * The ENERGY of each path (E_direct / E_ISL / E_GS / E_relay) -- TARGET. Supplied later by the
    energy track via route()'s callables; here they are labeled placeholders. So the final
    J = lam_E*E + lam_T*T decision is TARGET wherever lam_E != 0; the pure-latency choice (lam_E=0)
    is a SIM result.
  * lam_E, lam_T -- ASSUMPTION (invented trade-off knobs), swept.

Sanity gate (refuses to report before it passes, wp16 pattern): with ISL disabled the path choice
must collapse to the existing ground-station-only simulation -- same passes, same latency -- that
WP5b/WP6/WP8/wp18 use (find_passes(primary, Sfax, start, 36h)). If it cannot reproduce that
baseline exactly, the relay plumbing is wrong and no relay number can be trusted.

    .venv/Scripts/python.exe scripts/wp20_relay_path_choice.py

Outputs: results/wp20_relay_path_choice.json, results/wp20_isl_windows.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from skyfield.api import load

from sat7.orbit import GroundStation, LINK_PRESETS, OrbitConfig, find_passes, make_satellite
from sat7.relay import RelayConfig, isl_windows, make_relay, route_item

# The canonical setup used by WP5b/WP6/WP8/wp11/wp18 -- matched exactly so the gate reproduces them.
START = datetime(2026, 9, 18, tzinfo=timezone.utc)
HOURS = 36.0


def latency_series(times, direct, isl, relay, lam_E, lam_T, isl_enabled, **kw):
    """Chosen path + latency (s) for an item arriving at each grid time."""
    rows = [route_item(1.0, t, direct, isl, relay, lam_E=lam_E, lam_T=lam_T,
                       isl_enabled=isl_enabled, **kw) for t in times]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--raan-offset", type=float, default=90.0,
                    help="ASSUMPTION: relay plane RAAN offset (deg). Small offsets keep continuous "
                         "ISL but the SAME ground track (relay useless); a large offset gives "
                         "complementary Sfax coverage (the point of a relay). 90 = demonstrative default.")
    ap.add_argument("--grid-min", type=float, default=5.0, help="arrival-time grid step (minutes)")
    ap.add_argument("--lam-t", type=float, default=1.0 / 3600, help="ASSUMPTION latency weight (per s)")
    ap.add_argument("--lam-e", type=float, default=1.0, help="ASSUMPTION energy weight (per E unit)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    ts = load.timescale()
    link = LINK_PRESETS[args.link]
    primary = make_satellite(OrbitConfig(epoch=START), ts=ts)
    relay_cfg = RelayConfig(orbit=OrbitConfig(epoch=START, raan_deg=args.raan_offset))
    relay = make_relay(relay_cfg, ts=ts)

    direct = find_passes(primary, GroundStation(), START, HOURS, link, ts=ts)   # WP5b/WP6 baseline
    relay_p = find_passes(relay, GroundStation(), START, HOURS, link, ts=ts)
    isl = isl_windows(primary, relay, START, HOURS, grazing_km=relay_cfg.grazing_km,
                      max_range_km=relay_cfg.max_range_km, ts=ts)
    print(f"=== WP20 relay path choice (link {args.link}, relay RAAN +{args.raan_offset:.0f} deg) ===")
    print(f"SIM windows over {HOURS:.0f} h: primary {len(direct)} ground passes, "
          f"relay {len(relay_p)} ground passes, {len(isl)} ISL windows\n")

    # ---- SANITY GATE: ISL disabled must reproduce the ground-only baseline (WP5b/WP6/wp18) exactly
    fresh = find_passes(make_satellite(OrbitConfig(epoch=START), ts=ts), GroundStation(),
                        START, HOURS, link, ts=ts)
    same_passes = (len(direct) == len(fresh) and
                   all(abs(a.capacity_bytes - b.capacity_bytes) < 1.0 and a.rise == b.rise
                       for a, b in zip(direct, fresh)))
    grid = [START + timedelta(minutes=args.grid_min * k)
            for k in range(int(HOURS * 60 / args.grid_min))]
    base_wait = []                                    # ground-only latency, computed independently
    for t in grid:
        nxt = next((p.rise for p in direct if p.rise >= t), None)
        base_wait.append((nxt - t).total_seconds() if nxt else float("inf"))
    directonly = latency_series(grid, direct, isl, relay_p, args.lam_e, args.lam_t, isl_enabled=False)
    collapse_ok = all(r["path"] == "direct" and
                      (r["T_s"] == b or (r["T_s"] == float("inf") and b == float("inf")))
                      for r, b in zip(directonly, base_wait))
    if not (same_passes and collapse_ok):
        raise SystemExit(f"[sanity] FAILED: direct-only did not reproduce the ground-only baseline "
                         f"(passes match={same_passes}, latency match={collapse_ok}). Fix before "
                         f"trusting any relay number.")
    print(f"[sanity] ISL-disabled collapses to the ground-only baseline exactly: "
          f"{len(direct)} passes reproduced, latency identical at all {len(grid)} arrival times -> OK\n")

    # ---- SIM latency result: direct-only vs direct+relay (pure latency, lam_E=0 -> energy-free)
    finite = lambda xs: [x for x in xs if np.isfinite(x)]
    withrelay = latency_series(grid, direct, isl, relay_p, lam_E=0.0, lam_T=1.0, isl_enabled=True)
    d_h = np.array(finite(base_wait)) / 3600
    r_h = np.array(finite([x["T_s"] for x in withrelay])) / 3600
    relay_used = sum(1 for x in withrelay if x["path"] == "relay")
    lat = {
        "direct_only_median_h": round(float(np.median(d_h)), 3),
        "direct_only_p90_h": round(float(np.percentile(d_h, 90)), 3),
        "direct_only_max_h": round(float(d_h.max()), 3),
        "with_relay_median_h": round(float(np.median(r_h)), 3),
        "with_relay_p90_h": round(float(np.percentile(r_h, 90)), 3),
        "with_relay_max_h": round(float(r_h.max()), 3),
        "relay_chosen_frac_latency_optimal": round(relay_used / len(grid), 3),
    }
    print("SIM latency to downlink (energy-free, lam_E=0 -> pure latency):")
    print(f"  direct only  : median {lat['direct_only_median_h']:.2f} h, p90 "
          f"{lat['direct_only_p90_h']:.2f} h, max {lat['direct_only_max_h']:.2f} h")
    print(f"  direct+relay : median {lat['with_relay_median_h']:.2f} h, p90 "
          f"{lat['with_relay_p90_h']:.2f} h, max {lat['with_relay_max_h']:.2f} h  "
          f"(relay picked for {lat['relay_chosen_frac_latency_optimal']:.0%} of arrivals)")
    print(f"  worst-case latency cut by relay: "
          f"{lat['direct_only_max_h'] - lat['with_relay_max_h']:+.2f} h\n")

    # ---- TARGET path-choice: full J with placeholder energies, swept over lam_E (ASSUMPTION knob)
    sweep = {}
    for lamE in (0.0, 1.0, 5.0, 10.0, 20.0, 50.0):
        rows = latency_series(grid, direct, isl, relay_p, lam_E=lamE, lam_T=args.lam_t, isl_enabled=True)
        sweep[f"lam_E={lamE}"] = round(sum(1 for r in rows if r["path"] == "relay") / len(grid), 3)
    print("TARGET path choice -- relay-chosen fraction vs energy weight (placeholder E, lam_T "
          f"{args.lam_t:.2e}/s):")
    for k, v in sweep.items():
        print(f"  {k:12s} -> relay chosen {v:.0%}")

    pd.DataFrame([{"start": w.start.isoformat(), "end": w.end.isoformat(),
                   "duration_s": round(w.duration_s, 1), "min_range_km": round(w.min_range_km, 1),
                   "mean_range_km": round(w.mean_range_km, 1)} for w in isl]
                 ).to_csv(args.out / "wp20_isl_windows.csv", index=False)

    rep = {
        "label_note": "ISL windows + latency SIM (sat7.orbit SGP4); per-item energy E TARGET "
                      "(placeholder, from the energy track); lam_E/lam_T ASSUMPTION (swept).",
        "setup": {"start": START.isoformat(), "hours": HOURS, "link": args.link,
                  "relay_raan_offset_deg": args.raan_offset,
                  "relay_model": "SIM: 2nd satellite via sat7.orbit.make_satellite (reproducible "
                                 "Keplerian, RAAN-offset plane); no external TLE, no new orbital code"},
        "windows_SIM": {"primary_ground_passes": len(direct), "relay_ground_passes": len(relay_p),
                        "isl_windows": len(isl),
                        "isl_total_min": round(sum(w.duration_s for w in isl) / 60, 1),
                        "isl_min_range_km": round(min((w.min_range_km for w in isl), default=0.0), 1),
                        "isl_max_mean_range_km": round(max((w.mean_range_km for w in isl), default=0.0), 1)},
        "sanity_gate": {"ground_only_passes_reproduced": bool(same_passes),
                        "latency_collapses_to_baseline": bool(collapse_ok),
                        "n_arrival_times": len(grid), "passed": True,
                        "baseline": "find_passes(primary, Sfax, 2026-09-18, 36h) -- WP5b/WP6/wp18"},
        "latency_SIM_energy_free": lat,
        "path_choice_TARGET": {
            "rule": "J = lam_E*E + lam_T*T, choose min(J_direct, J_relay)",
            "E_status": "TARGET placeholder (E_direct=1, E_ISL=0.6, E_GS=1.0 units; NOT joules) -- "
                        "replace via route()'s direct_energy/relay_energy from the energy track",
            "lam_T_per_s_ASSUMPTION": args.lam_t,
            "relay_chosen_frac_vs_lam_E": sweep},
    }
    (args.out / "wp20_relay_path_choice.json").write_text(json.dumps(rep, indent=2))
    print(f"\nSaved wp20_relay_path_choice.json, wp20_isl_windows.csv in {args.out}")


if __name__ == "__main__":
    main()
