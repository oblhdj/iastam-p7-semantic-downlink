"""WP18 -- the B0-B3 campaign runner, built against the real entry points of report 18's spec.

`reports/18-campaign-runner-spec.md` fixed how the accepted paper's B0->B4 configs compose. This
executes the buildable part of that spec -- B0, B1, B2, B3 -- wiring each config to the module that
already exists, the way wp16's sanity gate validated the fusion before trusting it. **B4 is a stub**
that matches the reserved `route(plan, links)` interface exactly and raises NotImplementedError; it
is reported as TARGET, never as a filled-in number. Full B0-B4 end to end therefore stays blocked,
consistent with report 18 section 4 and START_HERE section 5 item 3 (no relay, no E_relay).

What each config is wired to (reuse, never reimplement -- the rule every wp script follows):
  B0  raw-volume reference      the raw bytes/day figure (SIM, report 05); no module runs
  B1  YOLO on a tile            a thin detect_fn over ultralytics YOLO.predict (the callable the
                                spec's B1 row anticipated, and the one B2 also needs)
  B2  SAHI + YOLO + fusion      sat7.b2_sahi_fusion.run_sahi(swath, detect_fn, SahiConfig(window=768))
  B3  + semantic policy         sat7.real_workload.workload_from_catalogue -> sat7.scheduler
                                (encode_lod -> simulate(ValueGreedy) -> metrics) + onboard_ceiling,
                                exactly the wp11_integration_demo chain, over the stored catalogue
  B4  + relay / adaptive comms  route(plan, links) STUB -> NotImplementedError (see report 18 section 3)

Labels (report 18 section 2, the project rule): B1/B2 recall REAL (detector on Airbus); B3 recall
SIM-over-REAL (scheduler sim over real detections) and never quoted without its cloud assumption
(START_HERE section 7); data/reduction SIM; energy from report 17 (ratios, not absolute joules).

INTERFACE MISMATCH this wiring surfaced (fed back into report 18, not patched around here): B1/B2
emit raw detections, but B3 consumes the per-tile *catalogue* frames (context, n_ships, false
alarms, matched confidences). Bridging them needs the catalogue-build adapter (run_prefilter +
flag_predictions + GT matching = wp11 stage 4 / wp6_build_catalogue), and B2's unit is a *swath*
while the catalogue is per *tile*. Report 12 proved the adapter reproduces the stored catalogue to
1e-4, so this run drives B3 from that stored catalogue (the proven path) and report 18 now names the
adapter as its own step rather than an arrow.

    .venv312/Scripts/python.exe -W ignore scripts/wp18_campaign_runner.py          # small-scale
    .venv312/Scripts/python.exe -W ignore scripts/wp18_campaign_runner.py --b1-tiles 500 --b2-swaths 4

Outputs: results/wp18_campaign.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
import numpy as np
import pandas as pd
import torch

from sat7.b2_sahi_fusion import SahiConfig, run_sahi
from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite
from sat7.real_workload import (RealWorkloadConfig, attach_coast_sizes, load_size_model,
                                workload_from_catalogue)
from sat7.scheduler import LoDConfig, ValueGreedy, encode_lod, metrics, simulate
from sat7.priority import PriorityConfig, encode_semantic   # P0-P3 scheme (selectable, defaults off)
from sat7.relay import LinkParams, RelayConfig, isl_windows, make_relay, route  # B4 (peer-owned)
from wp6_simulate_real import onboard_ceiling
from wp11_integration_demo import gt_boxes          # reuse the exact loader every WP uses
from wp12_seams import recall_of                    # reuse the exact greedy IoU matcher
from wp19_relay_energy import direct_energy, relay_energy  # B4 energy primitives (report 19, TARGET)

TILE = 768
B0_RAW_MB_DAY = 60_124.0                             # SIM, report 05 (raw offered per day at 40k tiles)
B0_DAY_TILES = 40_000                                # the tiles/day that 60,124 MB is measured at
B0_RAW_PER_TILE_MB = B0_RAW_MB_DAY / B0_DAY_TILES     # 1.503 MB/tile -> lets B0 scale-match any day


# ----------------------------------------------------------------------------- shared detector
def make_detect_fn(model, imgsz: int, conf: float, dev):
    """The thin detector callable B1 and B2 share: list of crops -> per-crop (cx,cy,w,h,conf)."""
    def detect_fn(crops):
        res = model.predict(crops, imgsz=imgsz, conf=conf, device=dev, verbose=False)
        return [[(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                 for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                for r in res]
    return detect_fn


def _recall_block(found: np.ndarray, sizes: np.ndarray) -> dict:
    def r(m):
        return round(float(found[m].mean()), 4) if m.any() else None
    return {"overall": round(float(found.mean()), 4), "small_<32": r(sizes < 32),
            "medium": r((sizes >= 32) & (sizes <= 96)), "large_>96": r(sizes > 96)}


# ----------------------------------------------------------------------------- B1
def run_b1(detect_fn, img_dir, lbl_dir, n_tiles, seed):
    """B1: plain YOLO per tile. REAL recall on real Airbus tiles."""
    files = sorted(img_dir.glob("*.jpg"))
    rng = np.random.default_rng(seed)
    pick = [files[i] for i in sorted(rng.choice(len(files), min(n_tiles, len(files)), replace=False))]
    found, sizes, n_det = [], [], 0
    for i in range(0, len(pick), 32):
        batch = pick[i:i + 32]
        crops = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in batch]
        per = detect_fn(crops)
        for f, dets in zip(batch, per):
            n_det += len(dets)
            gts = gt_boxes(lbl_dir / (f.stem + ".txt"))
            if gts:
                fnd = recall_of(dets, gts)
                found += fnd
                sizes += [max(g[2], g[3]) for g in gts]
    found, sizes = np.array(found, bool), np.array(sizes)
    return {"tiles": len(pick), "gt_ships": int(len(found)), "raw_detections": n_det,
            "recall": _recall_block(found, sizes)}


# ----------------------------------------------------------------------------- B2
def run_b2(detect_fn, manifest, ships_csv, img_dir, n_swaths, window):
    """B2: SAHI + fusion over real stitched swaths, via sat7.b2_sahi_fusion.run_sahi. REAL recall."""
    man = json.loads(manifest.read_text())
    ships = pd.read_csv(ships_csv)
    stride, L = man["stride_px"], man["swath_px"]
    cfg = SahiConfig(window=window)
    found, sizes, n_det, used = [], [], 0, 0
    for sw in man["swaths"][:n_swaths]:
        canvas = np.zeros((L, L, 3), np.uint8)
        for (r, c, name) in sw["source_tiles"]:
            t = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
            canvas[r * stride:r * stride + TILE, c * stride:c * stride + TILE] = t
        fused = run_sahi(canvas, detect_fn, cfg)          # swath-global detections (the B2 output)
        n_det += len(fused)
        g = ships[ships.swath_id == sw["swath_id"]]
        gts = list(zip(g.cx, g.cy, g.w, g.h))
        found += recall_of(fused, gts)
        sizes += list(g.size_px)
        used += 1
    found, sizes = np.array(found, bool), np.array(sizes)
    return {"swaths": used, "window_px": window, "gt_ships": int(len(found)),
            "fused_detections": n_det, "recall": _recall_block(found, sizes),
            "compute_vs_regular_tiling_x": 1.5625 if window == 768 else None}  # report 16


# ----------------------------------------------------------------------------- B3
def run_b3(cat_tiles, cat_ships, size_model, day_tiles, det_thr, link, link_share, storage_gb, seed,
           semantic="lod"):
    """B3: the real semantic-policy chain over the stored catalogue (= the wp11 chain).

    B1/B2 emit raw detections; this stage needs the per-tile catalogue frames, so it reuses the
    stored catalogue (report 12 proved live inference reproduces it to 1e-4). SIM-over-REAL recall.

    `semantic` selects the encoder: "lod" (the repo ladder, the canonical headline B3 reproduces) or
    "priority" (the paper's P0-P3 scheme, sat7.priority). The scheduler, passes and metrics are
    identical, so the two are directly comparable; the traceability gate in main() only applies to
    "lod" (the CSV it pins to is the LoD headline).
    """
    tiles = pd.read_csv(cat_tiles)
    attach_coast_sizes(tiles, Path(cat_tiles).parent)   # each coastal tile at its measured size (wp28)
    ships = pd.read_csv(cat_ships)
    cfg = RealWorkloadConfig(tiles_per_day=day_tiles, det_thr=det_thr, seed=seed)
    wl, stats = workload_from_catalogue(tiles, ships, cfg)
    lod = LoDConfig(conf_low=det_thr, size_model=load_size_model(size_model))
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    passes = find_passes(sat, GroundStation(), start, 36, LINK_PRESETS[link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * link_share) for p in passes]
    items = encode_semantic(wl, mode=semantic, lod=lod, cfg=PriorityConfig(p1_conf=lod.conf_high))
    res = simulate(items, passes, start, ValueGreedy(), storage_gb * 1e9, encoder=semantic.upper())
    mt = metrics(res, wl, lod)
    mt.update(onboard_ceiling(wl, lod, semantic))
    b0_matched_mb = stats.tiles * B0_RAW_PER_TILE_MB        # B0 raw at THIS run's tile count
    summary = {"semantic_scheme": semantic,
               "day_tiles": stats.tiles, "day_ships": stats.ships, "passes": len(passes),
               "ship_recall": round(float(mt["ship_recall"]), 4),
               "ceiling_recall": round(float(mt["ceiling_recall"]), 4),
               "MB_sent": round(float(mt["MB_sent"]), 1),
               "B0_raw_MB_scale_matched": round(b0_matched_mb, 0),
               "reduction_vs_B0_x": round(b0_matched_mb / float(mt["MB_sent"]), 1) if mt["MB_sent"] else None,
               "latency_med_h": round(float(mt["latency_med_h"]), 2),
               "cloud_assumption": f"link_share {link_share}; recall is at the workload's default cloud "
                                   f"fraction -- band 0.40-0.82 over 0-50% cloud, never quote bare"}
    # context B4 reroutes over: the scheduled downlink plan (what simulate() actually delivered) +
    # the primary passes + epoch. B4 cannot change recall/MB -- those are fixed here, upstream of route().
    ctx = {"res": res, "passes": passes, "start": start, "MB_sent": float(mt["MB_sent"]),
           "ship_recall": float(mt["ship_recall"])}
    return summary, ctx


# ----------------------------------------------------------------------------- B4
def run_b4(b3ctx, link_name, raan_offset, lam_E, lam_T, hours=36.0):
    """B4: reroute B3's delivered items via the optional relay where J = lam_E*E + lam_T*T favours it.

    REROUTE, not extra capacity (checked: scheduler.simulate owns eviction/capacity/aging UPSTREAM;
    sat7.relay.route is a post-B3 per-item path chooser that injects no passes). So B4's recall and
    MB-sent are B3's VERBATIM; only latency and energy move. Labels: ISL windows + latency SIM
    (orbit); per-item energy TARGET (report 19, pending real P_isl/R_isl); lam_E/lam_T ASSUMPTION.

    One modeled asymmetry (flagged by the energy track, latency is this track's): T_direct is B3's
    ACTUAL mid-pass delivered latency (via the direct_latency_s seam), while T_relay is still the
    window estimate (relay's next pass rise). So min(J) is slightly conservative against relay --
    relay's real mid-pass delivery would be marginally sooner. Fine for a first B0-B4; a relay-side
    delivered-time model is future work.
    """
    from skyfield.api import load
    ts = load.timescale()
    start = b3ctx["start"]
    direct = b3ctx["passes"]                                        # B3's primary ground passes
    primary = make_satellite(OrbitConfig(epoch=start), ts=ts)
    relay_cfg = RelayConfig(orbit=OrbitConfig(epoch=start, raan_deg=raan_offset))
    relay_sat = make_relay(relay_cfg, ts=ts)
    relay_p = find_passes(relay_sat, GroundStation(), start, hours, LINK_PRESETS[link_name], ts=ts)
    isl = isl_windows(primary, relay_sat, start, hours, grazing_km=relay_cfg.grazing_km, ts=ts)
    link = LinkParams()                                            # mirrors report 19 (energy track owns it)
    links = (direct, isl, relay_p)

    # plan = B3's scheduled downlink: delivered bytes (size*fraction) + arrival + ACTUAL direct latency
    sent = b3ctx["res"].sent
    plan = [{"bytes": s.item.size * s.fraction,
             "t_now": start + timedelta(seconds=s.item.created_s),
             "direct_latency_s": s.delivered_s - s.item.created_s} for s in sent]

    def run(lam_e, isl_on=True):
        return route(plan, links, lam_E=lam_e, lam_T=lam_T, isl_enabled=isl_on,
                     direct_energy=direct_energy, relay_energy=relay_energy, link=link)

    head = run(lam_E, isl_on=True)                                 # headline decision
    directonly = run(lam_E, isl_on=False)

    # ---- SANITY GATES (all must pass before any B4 number is trusted)
    b3_lat = [p["direct_latency_s"] for p in plan]
    mb_plan = sum(p["bytes"] for p in plan) / 1e6
    gate_a = (all(d["path"] == "direct" and abs(d["T_s"] - l) < 1e-6 for d, l in zip(directonly, b3_lat))
              and abs(mb_plan - b3ctx["MB_sent"]) < 1e-3)         # direct-only == B3 latency & MB exactly
    gate_b = abs(mb_plan - b3ctx["MB_sent"]) < 1e-3               # reroute: MB-sent unchanged (fraction-independent)
    grid = [start + timedelta(minutes=5 * k) for k in range(int(hours * 60 / 5))]
    gplan = [{"bytes": 1.0, "t_now": t} for t in grid]           # uniform arrivals, next-pass-wait direct
    gdec = route(gplan, links, lam_E=0.0, lam_T=1.0, isl_enabled=True,
                 direct_energy=direct_energy, relay_energy=relay_energy, link=link)
    f_grid = sum(1 for d in gdec if d["path"] == "relay") / len(grid)
    gate_c = abs(f_grid - 0.440) < 0.005                         # reproduce report 20's f at lam_E=0
    if not (gate_a and gate_b and gate_c):
        raise SystemExit(f"[B4 sanity] FAILED: (a) direct-only==B3 {gate_a}; (b) MB unchanged {gate_b}; "
                         f"(c) f=0.440 reproduced {gate_c} (got {f_grid:.3f}). Fix before trusting B4.")

    # ---- headline (lam_E = chosen; default 0 = pure latency-min, report 20's validated case)
    lat_h = np.array([d["T_s"] for d in head]) / 3600.0
    b3lat_h = np.array(b3_lat) / 3600.0
    n_relayed = sum(1 for d in head if d["path"] == "relay")
    f_relay = n_relayed / len(head)
    E_b4_kJ = sum(d["E"] for d in head) / 1000.0
    E_direct_kJ = sum(d["E"] for d in directonly) / 1000.0
    sweep = {f"lam_E={le:g}": round(sum(1 for d in run(le) if d["path"] == "relay") / len(head), 3)
             for le in (0.0, 50.0, 150.0, 500.0, 2000.0)}
    return {
        "onboard": "+relay/adaptive comms (reroute of B3's plan)", "status": "ran",
        "label": "latency SIM; energy TARGET (report 19); lambda ASSUMPTION",
        "design": "REROUTE not extra capacity -> recall & MB-sent are B3's verbatim (asserted)",
        "n_items_routed": len(head), "n_relayed": n_relayed,
        "relay_raan_offset_deg": raan_offset,
        "isl_windows": len(isl), "lam_E_headline": lam_E, "lam_T_per_s": lam_T,
        "ship_recall_SIMoverREAL": b3ctx["ship_recall"], "MB_sent_SIM": round(mb_plan, 1),
        "relay_fraction": round(f_relay, 3),
        "latency_h_B3_direct": {"median": round(float(np.median(b3lat_h)), 2),
                                "p90": round(float(np.percentile(b3lat_h, 90)), 2),
                                "max": round(float(b3lat_h.max()), 2)},
        "latency_h_B4_relay": {"median": round(float(np.median(lat_h)), 2),
                               "p90": round(float(np.percentile(lat_h, 90)), 2),
                               "max": round(float(lat_h.max()), 2)},
        "worst_case_latency_cut_h": round(float(b3lat_h.max() - lat_h.max()), 2),
        "energy_kJ_TARGET": {"direct_only": round(E_direct_kJ, 2), "b4_mix": round(E_b4_kJ, 2),
                             "note": "joules are report-19 placeholders pending real P_isl/R_isl"},
        "sanity_gates": {"a_directonly_reproduces_B3": bool(gate_a),
                         "b_reroute_MB_unchanged": bool(gate_b),
                         "c_report20_f0.440_reproduced": bool(gate_c), "f_grid": round(f_grid, 3),
                         "passed": True},
        "lam_E_sweep_relay_fraction_ASSUMPTION": sweep,
        "modeling_asymmetry": "T_direct = B3 actual mid-pass latency; T_relay = window estimate "
                              "(relay next-pass rise) -> slightly conservative against relay",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.pt")
    ap.add_argument("--manifest", type=Path, default=ROOT / "results" / "wp15_manifest.json")
    ap.add_argument("--swath-ships", type=Path, default=ROOT / "results" / "wp15_ships.csv")
    ap.add_argument("--cat-tiles", type=Path, default=ROOT / "results" / "wp6_tiles.csv")
    ap.add_argument("--cat-ships", type=Path, default=ROOT / "results" / "wp6_ships.csv")
    ap.add_argument("--size-model", type=Path, default=ROOT / "results" / "wp6_size_model.json")
    ap.add_argument("--b1-tiles", type=int, default=200, help="small-scale B1 sample")
    ap.add_argument("--b2-swaths", type=int, default=2, help="small-scale B2 swaths")
    ap.add_argument("--b2-window", type=int, default=768, help="SAHI window (768 = B2's decided setting)")
    ap.add_argument("--b3-day-tiles", type=int, default=40_000,
                    help="canonical operational load (wp6_real_table.csv headline); B3 reproduces its row")
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--semantic", choices=("lod", "priority"), default="lod",
                    help="B3 semantic encoder: 'lod' (repo ladder, the pinned headline) or 'priority' "
                         "(paper P0-P3, sat7.priority). The traceability gate applies only to 'lod'. "
                         "'priority' needs --tag: wp18_campaign.json is the canonical LoD campaign.")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. paper -> "
                    "wp18_campaign_paper.json)")
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25,
                    help="canonical link share (wp6_real_table.csv headline run)")
    ap.add_argument("--real-table", type=Path, default=ROOT / "results" / "wp6_real_table.csv",
                    help="CSV B3 is pinned to reproduce (traceability gate)")
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--b4-lam-e", type=float, default=0.0,
                    help="ASSUMPTION: B4 energy weight. 0 = headline (pure latency-min, report 20's "
                         "validated f=0.440 case); the sweep in the output covers the crossover")
    ap.add_argument("--b4-lam-t", type=float, default=1.0 / 3600, help="ASSUMPTION: B4 latency weight (per s)")
    ap.add_argument("--b4-raan", type=float, default=90.0,
                    help="ASSUMPTION: relay RAAN offset (deg); 90 = complementary coverage (report 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same-input", type=Path, default=ROOT / "results" / "wp26_b0_b4.json",
                    help="the same-input B0-B4 run (wp26_b0_b4.py): the canonical B1 and B2")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    if args.semantic != "lod" and not args.tag:
        ap.error("--semantic priority needs --tag (e.g. --tag paper): wp18_campaign.json is the "
                 "canonical LoD campaign")

    dev = 0 if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    detect_fn = make_detect_fn(model, 768, args.det_thr, dev)
    img_dir, lbl_dir = args.data / "images" / "test", args.data / "labels" / "test"
    print(f"=== WP18 B0-B3 campaign runner (small scale), device {dev} ===\n")
    t0 = time.perf_counter()

    configs = {}

    # B0 ---------------------------------------------------------------------
    configs["B0"] = {"onboard": "none", "label": "SIM", "status": "ran",
                     "raw_volume_MB_day_at_40k": B0_RAW_MB_DAY, "raw_MB_per_tile": round(B0_RAW_PER_TILE_MB, 3),
                     "source": "report 05 (scale-matched to B3's tile count for the reduction factor)"}
    print(f"B0 raw reference: {B0_RAW_MB_DAY:.0f} MB/day at 40k tiles = {B0_RAW_PER_TILE_MB:.3f} MB/tile [SIM]")

    # B1 / B2 -- canonical since 10 Oct 2026: the same-input run (wp26_b0_b4.py) -----------------
    # The paper's B1 is ONE detector call on the original (large) image and its B2 is SAHI on that
    # same image. wp26 measures both on the same 150 scenes of 4x4 real tiles; this runner reads
    # them, so the campaign carries one B1 and one B2. What this script used to report under those
    # names still runs below and is kept as "legacy_small_samples": a detector call per native
    # 768 px tile (no large image, so not the paper's B1) and SAHI on 2 swaths (47 ships).
    w26 = json.loads(args.same_input.read_text())
    q26, chk26, in26 = w26["detection_quality_same_scenes"], w26["checks"], w26["inputs"]

    def same_input(m, onboard, **extra):
        by = m["recall_by_size"]
        return {"scenes": m["images"], "scene_px": in26["scene_px"], "gt_ships": m["n_gt"],
                "detections": m["n_detections_supplied"],
                "recall": {"overall": m["recall"], "small_<32": by["small_<32"]["recall"],
                           "medium": by["medium_32-96"]["recall"], "large_>96": by["large_>96"]["recall"]},
                "precision": m["precision"], "F1": m["F1"], "AP50": m["AP50"],
                "conf_thr": m["conf_thr"], "iou_thr": m["iou_thr"], **extra,
                "onboard": onboard, "label": "REAL",
                "status": "read from results/wp26_b0_b4.json (same-input run)"}

    wins = chk26["3_B2_sahi_fusion"]["windows_per_scene"]
    b1 = same_input(q26["B1"], "detector, one call on the whole scene",
                    detector_calls_per_scene=chk26["2_B1_no_slicing"]["detector_calls_per_scene"])
    b2 = same_input(q26["B2"], "detector+SAHI+fusion", window_px=args.b2_window, windows_per_scene=wins,
                    compute_vs_regular_tiling_x=round(wins / in26["per_side"] ** 2, 4))
    configs["B1"], configs["B2"] = b1, b2
    print(f"B1 one call per {in26['scene_px']} px scene: recall {b1['recall']['overall']}, precision "
          f"{b1['precision']} on {b1['gt_ships']} ships [REAL, wp26]")
    print(f"B2 SAHI+fusion, same scenes: recall {b2['recall']['overall']}, precision {b2['precision']}, "
          f"{b2['compute_vs_regular_tiling_x']}x compute [REAL, wp26]")

    # the two earlier small samples: still measured, no longer called B1 / B2
    print(f"legacy: detector per native tile on {args.b1_tiles} tiles; SAHI on {args.b2_swaths} swaths ...",
          flush=True)
    old_b1 = run_b1(detect_fn, img_dir, lbl_dir, args.b1_tiles, args.seed)
    old_b2 = run_b2(detect_fn, args.manifest, args.swath_ships, img_dir, args.b2_swaths, args.b2_window)
    legacy = {
        "note": "what this runner reported as B1 / B2 until 10 Oct 2026. Different inputs from each other "
                "and from B3; matched at IoU >= 0.3. Not the paper's B1 / B2 -- kept as measured references.",
        "native_tiles": {**old_b1, "what": "one detector call per native 768 px tile (no large image)",
                         "label": "REAL", "full_scale": "results/wp24_detection_eval.json "
                                                        "B1_committed_all_test_tiles (8,173 ships)"},
        "swaths": {**old_b2, "what": "SAHI + fusion on stitched swaths", "label": "REAL",
                   "full_scale": "results/wp24_detection_eval.json swaths.rows.sahi_B2 (716 ships)"},
        "per_tile_on_the_same_scenes": {
            "recall": q26["B1_per_tile_reference"]["recall"],
            "precision": q26["B1_per_tile_reference"]["precision"],
            "gt_ships": q26["B1_per_tile_reference"]["n_gt"],
            "what": "one call per 768 px source tile of the wp26 scenes = a 16-window tiling without overlap"}}
    print(f"   native tiles recall {old_b1['recall']['overall']} ({old_b1['gt_ships']} ships); swaths recall "
          f"{old_b2['recall']['overall']} ({old_b2['gt_ships']} ships) [REAL, references only]")

    # B3 ---------------------------------------------------------------------
    print(f"B3 semantic policy: real scheduler chain over the stored catalogue ...", flush=True)
    b3, b3ctx = run_b3(args.cat_tiles, args.cat_ships, args.size_model, args.b3_day_tiles, args.det_thr,
                       args.link, args.link_share, args.storage_gb, args.seed, semantic=args.semantic)
    b3.update({"onboard": f"detector+gate+{args.semantic}+scheduler", "label": "SIM-over-REAL",
               "status": "ran"})
    b3["run_params"] = {"day_tiles": b3["day_tiles"], "link_share": args.link_share, "seed": args.seed,
                        "det_thr": args.det_thr, "mix": "orbit", "conf_high": 0.670,
                        "semantic": args.semantic,
                        "cloud_caveat": "recall at an assumed 15% cloud fraction; band 0.40-0.82 across 0-50%"}
    # ---- traceability gate: B3 must reproduce the named canonical CSV row within explicit tolerance.
    #      Only the LoD encoder is pinned to that CSV; the P0-P3 scheme is a different (comparable)
    #      policy with its own byte/recall profile, so the gate runs for --semantic lod only (see
    #      scripts/wp23_semantic_compare.py for the LoD-vs-P0-P3 head-to-head).
    if args.semantic == "lod":
        rt = pd.read_csv(args.real_table)
        vg = rt[rt.strategy.str.contains("value-greedy", case=False, na=False)].iloc[0]
        mb_csv, rec_csv, red_csv = float(vg["MB_sent"]), float(vg["ship_recall"]), float(vg["data_reduction_x"])
        MB_TOL, REC_TOL = 0.02, 0.01                        # MB to 2%; recall to the day-resampling spread
        mb_ok = abs(b3["MB_sent"] - mb_csv) / mb_csv <= MB_TOL
        rec_ok = abs(b3["ship_recall"] - rec_csv) <= REC_TOL
        if not (mb_ok and rec_ok):
            raise SystemExit(f"[B3 trace] FAILED: MB {b3['MB_sent']} vs CSV {mb_csv:.1f} (tol {MB_TOL:.0%} -> {mb_ok}); "
                             f"recall {b3['ship_recall']} vs CSV {rec_csv:.4f} (tol {REC_TOL} -> {rec_ok}). "
                             f"B3 is not reproducing wp6_real_table.csv 'Ours: LoD + value-greedy'.")
        b3["traceability"] = {"csv": "results/wp6_real_table.csv", "row": "Ours: LoD + value-greedy",
                              "MB_sent_csv": round(mb_csv, 1), "MB_tol_frac": MB_TOL,
                              "ship_recall_csv": round(rec_csv, 4), "recall_tol_abs": REC_TOL,
                              "reduction_csv_x": round(red_csv, 1),
                              "note": "MB reproduces to <2%; recall is a day-resampling realization within "
                                      "the documented [0.680, 0.690] spread (START_HERE, reports 05/11)"}
        configs["B3"] = b3
        print(f"   ship_recall {b3['ship_recall']} (ceiling {b3['ceiling_recall']}), "
              f"{b3['MB_sent']} MB sent, {b3['reduction_vs_B0_x']}x vs B0 [SIM-over-REAL]  "
              f"-- {b3['cloud_assumption'].split(';')[0]}")
        print(f"   [trace] reproduces wp6_real_table 'Ours: LoD + value-greedy' "
              f"(CSV {mb_csv:.1f} MB / recall {rec_csv:.4f}): MB within 2% {mb_ok}, recall within {REC_TOL} {rec_ok}")
    else:
        b3["traceability"] = {"skipped": "P0-P3 scheme is not pinned to the LoD headline CSV; "
                                         "see scripts/wp23_semantic_compare.py for the head-to-head"}
        configs["B3"] = b3
        print(f"   ship_recall {b3['ship_recall']} (ceiling {b3['ceiling_recall']}), "
              f"{b3['MB_sent']} MB sent, {b3['reduction_vs_B0_x']}x vs B0 [SIM-over-REAL, P0-P3]  "
              f"-- traceability gate skipped (not the LoD headline)")

    # B4 ---------------------------------------------------------------------
    print(f"B4 relay reroute via sat7.relay.route (lam_E={args.b4_lam_e}, RAAN +{args.b4_raan:.0f}) ...",
          flush=True)
    b4 = run_b4(b3ctx, args.link, args.b4_raan, args.b4_lam_e, args.b4_lam_t)
    configs["B4"] = b4
    print(f"   [sanity] direct-only==B3 {b4['sanity_gates']['a_directonly_reproduces_B3']}, "
          f"MB unchanged {b4['sanity_gates']['b_reroute_MB_unchanged']}, "
          f"f=0.440 reproduced {b4['sanity_gates']['c_report20_f0.440_reproduced']} "
          f"(got {b4['sanity_gates']['f_grid']}) -> OK")
    print(f"   recall {b4['ship_recall_SIMoverREAL']} & {b4['MB_sent_SIM']} MB = B3 verbatim (reroute); "
          f"relay {b4['relay_fraction']:.0%} of items; worst-case latency "
          f"{b4['latency_h_B3_direct']['max']}h -> {b4['latency_h_B4_relay']['max']}h "
          f"({b4['worst_case_latency_cut_h']:+}h) [latency SIM, energy TARGET]")

    report = {
        "label_note": "B0 SIM; B1/B2 REAL, read from the same-input run wp26 (150 scenes of 4x4 real "
                      "Airbus tiles); B3 SIM-over-REAL (state cloud; coastal tile bytes measured, other "
                      "products modeled); B4 latency SIM, energy TARGET (report 19), lambda ASSUMPTION; "
                      "energy from report 17 (ratios).",
        "scale": "B1/B2: 150 scenes, 1,164 ships (wp26). B3/B4: one simulated 40,000-tile day. "
                 "Different inputs -- B1/B2 are comparable with each other, not with B3's delivered recall.",
        "b0_b4_end_to_end": "RUNS -- B4 wired to sat7.relay.route (report 20 windows/choice + report "
                            "19 energy); B4 recall/MB are B3's (reroute), latency improved, energy TARGET",
        "energy_reference": "report 17 / results/wp17_energy_model.json (quote ratios, not joules)",
        "b4_design_question_resolved": "REROUTE not extra capacity -- scheduler.simulate owns "
                                       "eviction/capacity/aging upstream; route() changes only latency "
                                       "and energy. Confirmed by checking, asserted by gate (b).",
        "interface_mismatch_found": (
            "B1/B2 emit raw detections but B3 consumes per-tile catalogue frames, and B2's unit is a "
            "swath vs B3's per-tile catalogue -- the catalogue-build adapter (run_prefilter + "
            "flag_predictions + GT match = wp11 stage 4) is a required step the spec drew as an "
            "arrow. report 18 updated to name it; report 12 proved it reproduces the catalogue to 1e-4."),
        "b4_interface_extension": (
            "route_item gained direct_latency_s (peer-added, backward-compatible) so B4 uses B3's "
            "ACTUAL mid-pass delivered latency as the direct baseline -> gate (a) exact; wp20 numbers "
            "unmoved."),
        "configs": configs,
        "legacy_small_samples": legacy,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    out_name = f"wp18_campaign{'_' + args.tag if args.tag else ''}.json"
    (args.out / out_name).write_text(json.dumps(report, indent=2))

    print(f"\n{'config':5s} {'onboard':40s} {'key result':38s} label")
    def line(c, res):
        print(f"{c:5s} {configs[c]['onboard']:40s} {res:38s} {configs[c]['label'][:34]}")
    line("B0", f"{B0_RAW_MB_DAY:.0f} MB/day raw")
    line("B1", f"recall {b1['recall']['overall']} ({b1['gt_ships']} ships, wp26)")
    line("B2", f"recall {b2['recall']['overall']} @ {b2['compute_vs_regular_tiling_x']}x")
    line("B3", f"recall {b3['ship_recall']}, {b3['reduction_vs_B0_x']}x vs B0")
    line("B4", f"=B3 recall/MB; max lat {b4['latency_h_B3_direct']['max']}->{b4['latency_h_B4_relay']['max']}h")
    print(f"\nB0-B4 end-to-end: RUNS (B4 = reroute, latency SIM + energy TARGET). "
          f"Saved {out_name} in {args.out}")


if __name__ == "__main__":
    main()
