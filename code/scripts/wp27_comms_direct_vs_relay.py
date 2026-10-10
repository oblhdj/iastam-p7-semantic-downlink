"""WP27 -- direct vs relay downlink [SIM], with capacity and transmission time (audit E3).

SIM throughout: propagated orbits, geometric contact windows, a link-budget rate model and assumed
powers. No real satellite link is involved; every line printed and every JSON block says so.

wp20 / wp18-B4 chose a path per item from window START times only: the relay could carry any
amount instantly, so B4 was "a reroute, not extra capacity". sat7.comms simulates the two paths
with what a link actually has -- a rate, a duration, a capacity -- on the REAL packet sizes of the
P0-P3 semantic downlink (wp25: binary records, JPEG crops, CCSDS packets):

    direct   primary --ground pass--> Sfax
    relay    primary --ISL window--> relay satellite --relay's ground pass--> Sfax
             (the relay forwards encoded items; it never sees pixels or runs a detector)

  1 gate      relay disabled == sat7.scheduler.simulate, delivery for delivery
  2 nominal   40k tiles/day: direct-only vs relay -- bytes per path, ship latency, transmit time
              and energy per stage, how full each link is
  3 congested 160k tiles/day: the relay's ground passes are extra capacity, not just a shortcut
  4 sweeps    ISL rate, relay ground share, energy weight lam_E, deadline, link availability
  5 example   one item followed down both paths: bytes, rates, seconds and joules per stage

Labels: contact windows / latency / capacity SIM (sat7.orbit, SGP4); packet sizes REAL (wp25);
rates and powers ASSUMPTION (reports 17, 19), swept in 4.

    .venv/Scripts/python.exe scripts/wp27_comms_direct_vs_relay.py     # ~3 min, needs the split
Output: results/wp27_comms.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from sat7.comms import LABEL, CommsConfig, build_links, pass_rate_bps, simulate_comms
from sat7.priority import PriorityConfig
from sat7.real_workload import RealWorkloadConfig, load_catalogue, load_size_model, workload_from_catalogue
from sat7.scheduler import Item, LoDConfig, ValueGreedy, metrics, simulate
from sat7.semantic import EncoderConfig
from wp25_semantic_packets import Corpus, build_day

S = f"[{LABEL}]"


def ship_metrics(cr, wl, lod) -> dict:
    m = metrics(cr.result, wl, lod)
    return {k: round(float(m[k]), 4) for k in ("ship_recall", "dark_recall", "latency_med_h", "latency_p90_h",
                                              "MB_sent", "MB_offered")}


def block(cr, wl, lod) -> dict:
    return {"label": LABEL, "ships": ship_metrics(cr, wl, lod), "totals": cr.totals(),
            "by_path": cr.by_path(), "link_use": cr.link_use}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--day-tiles", type=int, default=40_000)
    ap.add_argument("--congested-tiles", type=int, default=160_000)
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    res_dir, t_all = ROOT / "results", time.perf_counter()
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    cfg = CommsConfig()
    links = build_links(cfg, start, 36.0)
    tiles, ships = load_catalogue(res_dir)
    corpus = Corpus(args.data, tiles, res_dir / "wp1_predictions.csv")
    lod = LoDConfig(conf_low=args.det_thr, size_model=load_size_model(res_dir / "wp6_size_model.json"))
    ecfg = EncoderConfig(det_thr=args.det_thr, priority=PriorityConfig(p1_conf=lod.conf_high))
    cache, storage = {}, args.storage_gb * 1e9

    def day(n_tiles):
        wl, _ = workload_from_catalogue(tiles, ships, RealWorkloadConfig(
            tiles_per_day=n_tiles, det_thr=args.det_thr, seed=args.seed))
        return wl, build_day(wl, corpus, ecfg, lod, cache, "measured")

    def run(items, c, lk=None):
        return simulate_comms(items, lk or links, start, ValueGreedy(), c, storage)

    report = {"label": f"{LABEL} -- simulated links (SGP4 orbits, geometric windows, link-budget rates, "
                       "assumed powers); packet sizes REAL (wp25); no real satellite connection",
              "config": {"ground_link": cfg.link.name, "link_efficiency": cfg.link.efficiency,
                         "link_share": cfg.link_share, "relay_link_share": cfg.relay_share,
                         "p_tx_W": cfg.p_tx_W, "isl_rate_bps": cfg.isl_rate_bps,
                         "isl_efficiency": cfg.isl_efficiency, "isl_effective_bps": cfg.isl_effective_bps,
                         "p_isl_W": cfg.p_isl_W, "relay_raan_deg": cfg.relay_raan_deg,
                         "lam_T_per_s": cfg.lam_T, "lam_E_per_J": cfg.lam_E},
              "contact_plan": {"label": LABEL, "hours": 36,
                               "primary_passes": [{"rise": p.rise.isoformat(timespec="minutes"),
                                                   "duration_s": round(p.duration_s),
                                                   "capacity_MB": round(p.capacity_bytes / 1e6, 2),
                                                   "payload_rate_Mbps": round(pass_rate_bps(p) / 1e6, 3),
                                                   "physical_rate_Mbps": round(
                                                       pass_rate_bps(p, links.primary_share) / 1e6, 3)}
                                                  for p in links.primary_passes],
                               "relay_passes": [{"rise": p.rise.isoformat(timespec="minutes"),
                                                 "duration_s": round(p.duration_s),
                                                 "capacity_MB": round(p.capacity_bytes / 1e6, 2),
                                                 "payload_rate_Mbps": round(pass_rate_bps(p) / 1e6, 3),
                                                 "physical_rate_Mbps": round(
                                                     pass_rate_bps(p, links.relay_share) / 1e6, 3)}
                                                for p in links.relay_passes],
                               "isl_windows": len(links.isl),
                               "isl_minutes": round(sum(w.duration_s for w in links.isl) / 60, 1),
                               "isl_capacity_MB_per_window_median": round(float(np.median(
                                   [cfg.isl_effective_bps * w.duration_s / 8e6 for w in links.isl])), 1)}}
    print(f"{S} contact plan over 36 h: {len(links.primary_passes)} primary passes "
          f"({sum(p.capacity_bytes for p in links.primary_passes) / 1e6:.1f} MB at share {cfg.link_share}), "
          f"{len(links.relay_passes)} relay passes ({sum(p.capacity_bytes for p in links.relay_passes) / 1e6:.1f} MB), "
          f"{len(links.isl)} ISL windows ({report['contact_plan']['isl_minutes']} min, "
          f"{cfg.isl_effective_bps / 1e6:.2f} Mbps effective)")

    # ---- 1. regression gate: no relay == scheduler.simulate
    wl, items = day(args.day_tiles)
    ref = simulate(items, links.primary_passes, start, ValueGreedy(), storage)
    off = run(items, replace(cfg, relay_enabled=False))
    same = (len(ref.sent) == len(off.result.sent) and abs(ref.bytes_sent - off.result.bytes_sent) < 1e-6
            and all(a.item is b.item and a.fraction == b.fraction and a.delivered_s == b.delivered_s
                    for a, b in zip(ref.sent, off.result.sent)))
    report["gate_direct_only_equals_scheduler"] = {"passed": bool(same), "deliveries": len(ref.sent),
                                                   "MB": round(ref.bytes_sent / 1e6, 3)}
    print(f"{S} gate: relay disabled == sat7.scheduler.simulate on {len(ref.sent)} deliveries: "
          f"{'PASS' if same else 'FAIL'}")
    if not same:
        raise SystemExit("gate failed: the comms simulator does not reduce to the scheduler")

    # ---- 2 + 3. nominal and congested, direct-only vs relay
    for name, n_tiles in (("nominal", args.day_tiles), ("congested", args.congested_tiles)):
        if name == "congested":
            wl, items = day(n_tiles)
        d = run(items, replace(cfg, relay_enabled=False))
        r = run(items, cfg)
        raw_on_relay = sum(1 for x in r.deliveries if x.path == "relay" and x.item.kind not in cfg.relay_kinds)
        over = {k: v["used"] > 1 + 1e-9 for k, v in r.link_use.items()}
        report[name] = {"tiles": n_tiles, "items": len(items),
                        "MB_offered": round(sum(i.size for i in items) / 1e6, 2),
                        "direct_only": block(d, wl, lod), "with_relay": block(r, wl, lod),
                        "checks": {"non_semantic_items_on_relay": raw_on_relay,
                                   "any_link_over_capacity": any(over.values())}}
        ds, rs = report[name]["direct_only"], report[name]["with_relay"]
        print(f"\n{S} {name}: {n_tiles} tiles/day, {len(items)} items, {report[name]['MB_offered']} MB offered")
        for tag, b in (("direct only", ds), ("with relay ", rs)):
            t = b["totals"]
            print(f"{S}   {tag}: delivered {t['MB_delivered']:8.2f} MB  ship recall {b['ships']['ship_recall']:.4f}  "
                  f"latency med {b['ships']['latency_med_h']:.2f} h / p90 {b['ships']['latency_p90_h']:.2f} h  "
                  f"tx {t['tx_s']:.0f} s  energy {t['energy_J'] / 1e3:.2f} kJ")
        bp = rs["by_path"]
        print(f"{S}   relay path: {bp['relay']['items']} items, {bp['relay']['MB']} MB; ISL {bp['relay']['tx_s']['isl']} s "
              f"/ {bp['relay']['energy_J']['isl'] / 1e3:.2f} kJ + relay ground {bp['relay']['tx_s']['relay_ground']} s "
              f"/ {bp['relay']['energy_J']['relay_ground'] / 1e3:.2f} kJ")
        print(f"{S}   link use: " + ", ".join(f"{k} {100 * v['used']:.1f}% of {v['capacity_MB']} MB"
                                              for k, v in rs["link_use"].items()))
        if name == "nominal":
            nominal_items, nominal_wl, nominal_d, nominal_r = items, wl, d, r
    congested_items, congested_wl = items, wl

    # ---- 4. sweeps (ASSUMPTIONS): each row is one change from the default
    def row(c, its, w, lk=None):
        cr = run(its, c, lk)
        m, t = ship_metrics(cr, w, lod), cr.totals()
        return {"ship_recall": m["ship_recall"], "latency_med_h": m["latency_med_h"],
                "latency_p90_h": m["latency_p90_h"], "MB_delivered": t["MB_delivered"],
                "relay_fraction_items": t["relay_fraction_items"], "energy_kJ": round(t["energy_J"] / 1e3, 2),
                "deadline_met_fraction": t["deadline_met_fraction"]}
    sweeps = {"label": LABEL}
    sweeps["isl_rate_Mbps_nominal"] = {str(r): row(replace(cfg, isl_rate_bps=r * 1e6), nominal_items, nominal_wl)
                                       for r in (0.01, 0.1, 0.5, 2.53, 10.0)}
    sweeps["relay_ground_share_congested"] = {}
    for sh in (0.0, 0.05, 0.10, 0.25, 0.50):
        c = replace(cfg, relay_link_share=sh)
        sweeps["relay_ground_share_congested"][str(sh)] = row(c, congested_items, congested_wl,
                                                              build_links(c, start, 36.0))
    sweeps["lam_E_per_J_nominal"] = {str(le): row(replace(cfg, lam_E=le), nominal_items, nominal_wl)
                                     for le in (0.0, 0.1, 1.0, 10.0, 100.0)}
    sweeps["deadline_h_nominal"] = {}
    for dl in (1.0, 2.0, 4.0, 6.0):
        sweeps["deadline_h_nominal"][str(dl)] = {
            "direct_only": row(replace(cfg, relay_enabled=False, deadline_s=dl * 3600), nominal_items, nominal_wl),
            "with_relay": row(replace(cfg, deadline_s=dl * 3600), nominal_items, nominal_wl)}
    sweeps["availability_nominal"] = {}
    for ga, ia in ((1.0, 1.0), (1.0, 0.5), (0.8, 1.0), (0.8, 0.5)):
        c = replace(cfg, ground_availability=ga, isl_availability=ia, availability_seed=1)
        sweeps["availability_nominal"][f"ground {ga} / isl {ia}"] = row(c, nominal_items, nominal_wl,
                                                                        build_links(c, start, 36.0))
    report["sweeps"] = sweeps
    print(f"\n{S} sweeps (one change from the default; rates and powers are ASSUMPTIONS):")
    for key, table in sweeps.items():
        if key == "label":
            continue
        print(f"{S}   {key}")
        for k, v in table.items():
            if "with_relay" in v:
                print(f"{S}     {k:>5} h: deadline met direct-only {v['direct_only']['deadline_met_fraction']} "
                      f"-> with relay {v['with_relay']['deadline_met_fraction']}")
            else:
                print(f"{S}     {k:>22}: recall {v['ship_recall']:.4f}  latency med {v['latency_med_h']:.2f} h  "
                      f"relay {100 * v['relay_fraction_items']:.1f}%  energy {v['energy_kJ']} kJ")

    # ---- 5. one item down both paths
    relay_by_id = {x.item.id: x for x in nominal_r.deliveries if x.path == "relay"}
    direct_by_id = {x.item.id: x for x in nominal_d.deliveries}
    ex = {}
    for kind in ("P3", "P2", "P1"):
        cand = [x for x in relay_by_id.values() if x.item.kind == kind and x.item.id in direct_by_id]
        if not cand:
            continue
        x = max(cand, key=lambda z: z.item.size)
        dd = direct_by_id[x.item.id]
        ex[kind] = {"label": LABEL, "bytes": int(x.item.size), "created_h": round(x.item.created_s / 3600, 3),
                    "direct": {"delivered_h": round(dd.delivered_s / 3600, 3),
                               "latency_h": round(dd.latency_s / 3600, 3),
                               "tx_s": {k: round(v, 4) for k, v in dd.tx_s.items()},
                               "energy_J": {k: round(v, 4) for k, v in dd.energy_J.items()},
                               "rate_Mbps": round(x.item.size * dd.fraction * 8 / dd.tx_s["ground"] / 1e6, 3)},
                    "relay": {"delivered_h": round(x.delivered_s / 3600, 3),
                              "latency_h": round(x.latency_s / 3600, 3),
                              "tx_s": {k: round(v, 4) for k, v in x.tx_s.items()},
                              "energy_J": {k: round(v, 4) for k, v in x.energy_J.items()},
                              "isl_rate_Mbps": round(x.item.size * 8 / x.tx_s["isl"] / 1e6, 3),
                              "relay_ground_rate_Mbps": round(x.item.size * 8 / x.tx_s["relay_ground"] / 1e6, 3)}}
    report["example_items_both_paths"] = ex
    print(f"\n{S} one item down both paths (nominal day):")
    for kind, e in ex.items():
        d, r = e["direct"], e["relay"]
        print(f"{S}   {kind} item, {e['bytes']:,} B, captured at {e['created_h']} h")
        print(f"{S}     direct: {e['bytes']:,} B x 8 / {d['rate_Mbps']} Mbps = {d['tx_s']['ground']} s, "
              f"{d['energy_J']['ground']} J; lands at {d['delivered_h']} h (latency {d['latency_h']} h)")
        print(f"{S}     relay : ISL {r['tx_s']['isl']} s at {r['isl_rate_Mbps']} Mbps ({r['energy_J']['isl']} J) "
              f"+ relay ground {r['tx_s']['relay_ground']} s at {r['relay_ground_rate_Mbps']} Mbps "
              f"({r['energy_J']['relay_ground']} J); lands at {r['delivered_h']} h (latency {r['latency_h']} h)")

    report["elapsed_s"] = round(time.perf_counter() - t_all, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wp27_comms.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{S} saved {args.out / 'wp27_comms.json'} ({report['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
