"""WP26 -- B0..B4 as selectable modes of one pipeline, on the SAME inputs, with the paper's checks.

wp18 ran the five configurations on five different inputs: B0 a constant (60,124 MB/day), B1 200
random tiles, B2 two swaths (47 ships), B3 the catalogue day built from B1-type per-tile detections
(no SAHI), B4 a reroute of that. Its own JSON calls it "a small-scale validation run". This runner
uses sat7.campaign: one chain, three switches (perception / payload / relay), and every mode sees:

  * the same scenes -- 4 x 4 real test tiles stitched (wp15 convention), contexts drawn by the orbit
    mix -- rendered ONCE per scene and handed to every mode as the same array (hashed);
  * the same detector, 0.25 onboard cut, threshold-then-match IoU 0.5 (sat7.evaluation);
  * the same packetization (sat7.semantic: records, JPEG crops, raw pixels, CCSDS packets);
  * the same day: scene instances, capture times, AIS (dark) flags, orbit passes, link share,
    storage, value-greedy scheduler (sat7.scheduler) -- hashed and recorded per mode.

Then it checks the paper's methodology and records the evidence:
  1 B0 transmits the original image   (decoded B0 packets == the rendered scene, byte for byte)
  2 B1 has no SAHI slicing stage       (exactly 1 detector call per scene, on the full frame)
  3 B2 uses SAHI and fuses correctly   (25 windows; no two fused boxes overlap at IoU >= 0.5;
                                        x_global = x0 + x_local exact against per-tile detection)
  4 B3 follows the selected policy     (every level == priority.classify; P0 never sent; one
                                        context per coastal tile; ground decode round trip)
  5 same inputs and assumptions        (identical fingerprints across every mode)

Labels: detection metrics REAL (trained detector, real tiles, real labels); bytes REAL (measured
serializations); delivered recall / latency SIM-over-REAL; B4's links SIM (sat7.comms: capacity,
transmit time and energy per stage; rates and powers ASSUMPTION); tile mix, AIS share, thresholds
ASSUMPTION.

    python scripts/wp26_b0_b4.py                      # all modes, 150 scenes (~5 min CPU, ONNX)
    python scripts/wp26_b0_b4.py --modes B2,B3 --scenes 20
    python scripts/wp26_b0_b4.py --modes B1,B2 --window 512 --overlap 0.20 --tag paper
                                                      # the paper's representative SAHI setting
Output: results/wp26_b0_b4.json   (numbers and hashes only -- no pixels); with --tag,
results/wp26_b0_b4_<tag>.json. A window or overlap other than 768 / 0.20 needs --tag, so the
canonical file cannot be overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import csv
import numpy as np

from sat7.b2_sahi_fusion import _iou
from sat7.campaign import MODES, ONBOARD_CTX, TILE, downlink, perceive, render, sample_scenes
from sat7.datasets import parse_yolo_lines
from sat7.evaluation import evaluate_detections, match_pairs
from sat7.imagery import load_image
from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite
from sat7.perception import PerceptionConfig, detect_image, letterbox, load_detector, slice_count
from sat7.priority import PriorityConfig, classify
from sat7.comms import LABEL, CommsConfig, build_links, simulate_comms
from sat7.scheduler import Item, LoDConfig, Ship, ValueGreedy, Workload, WorkloadConfig, metrics, simulate
from sat7.semantic import (EncoderConfig, Ids, Packetizer, decode_downlink, encode_jpeg,
                           raw_packet_bytes)


class Counted:
    """Wraps any detect_fn and counts detector calls (one per crop) -- evidence for checks 2 and 3."""
    def __init__(self, fn):
        self.fn, self.calls, self.size = fn, 0, getattr(fn, "size", TILE)

    def __call__(self, crops):
        self.calls += len(crops)
        return self.fn(crops)


def shape_only(px: int):
    """A zero-cost array with a scene's shape: the encoder needs H x W only when a memo serves crops."""
    return np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (px, px, 3))


class Memo:
    """Serves each distinct crop's detections once, so two code paths can be handed the SAME
    per-crop output. The coordinate gate then tests the lift arithmetic alone, on any backend."""
    def __init__(self, fn):
        self.fn, self.store = fn, {}

    def __call__(self, crops):
        keys = [hashlib.blake2b(np.ascontiguousarray(c).tobytes(), digest_size=16).digest() for c in crops]
        todo = [i for i, k in enumerate(keys) if k not in self.store]
        if todo:
            for i, out in zip(todo, self.fn([crops[i] for i in todo])):
                self.store[keys[i]] = out
        return [self.store[k] for k in keys]


def batch_vs_single(batched: list, single: list, acc: dict) -> None:
    """How much the DETECTOR itself differs between one batched call and single calls (a backend
    property: zero on ONNX CPU, small float differences on a GPU -- report 12). Not a gate. Each
    batched box is compared with its NEAREST single-call box."""
    acc["tiles"] += 1
    if len(batched) != len(single):
        acc["tiles_with_different_box_count"] += 1
    for a in batched:
        if not single:
            break
        b = min(single, key=lambda z: max(abs(a[i] - z[i]) for i in range(4)))
        acc["boxes"] += 1
        acc["boxes_not_bit_identical"] += a != b
        acc["max_box_diff_px"] = max(acc["max_box_diff_px"], max(abs(a[i] - b[i]) for i in range(4)))
        acc["max_conf_diff"] = max(acc["max_conf_diff"], abs(a[4] - b[4]))


def sha(obj) -> str:
    b = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(b).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--model", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.onnx")
    ap.add_argument("--modes", default="B0,B1,B2,B3,B4")
    ap.add_argument("--scenes", type=int, default=150)
    ap.add_argument("--per-side", type=int, default=4)
    ap.add_argument("--day-tiles", type=int, default=40_000)
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--p-dark", type=float, default=0.10)
    ap.add_argument("--b4-raan", type=float, default=90.0)
    ap.add_argument("--b4-lam-e", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=TILE,
                    help="SAHI window in px: 768 = the canonical B2; 512 = the paper's representative "
                         "setting. The detector input stays 768 px, so a smaller window is enlarged to it")
    ap.add_argument("--overlap", type=float, default=0.20, help="SAHI overlap between neighbouring windows")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. paper -> "
                    "wp26_b0_b4_paper.json)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    if (args.window, args.overlap) != (TILE, 0.20) and not args.tag:
        ap.error("--window / --overlap other than 768 / 0.20 need --tag (e.g. --tag paper): "
                 "wp26_b0_b4.json is the canonical 768 px run")
    sahi_cfg = PerceptionConfig(mode="sahi", window=args.window, overlap=args.overlap)
    modes = [MODES[m.strip()] for m in args.modes.split(",")]
    res_dir, t_all = ROOT / "results", time.perf_counter()
    img_dir, lbl_dir = args.data / "images" / "test", args.data / "labels" / "test"

    # ================================================================ shared inputs (built once)
    with (res_dir / "wp6_tiles.csv").open(newline="") as f:
        cat = list(csv.DictReader(f))
    pools = {}
    for r in cat:
        pools.setdefault(r["ctx3"], []).append(r["image"])
    onboard = {r["image"]: ONBOARD_CTX.get(r["context"], "ships") for r in cat}

    def gts_of(name):
        p = lbl_dir / (Path(name).stem + ".txt")
        return parse_yolo_lines(p.read_text().splitlines(), TILE, TILE, {0}) if p.exists() else []

    scenes = sample_scenes(pools, onboard, gts_of, args.scenes, args.per_side, seed=args.seed)
    manifest = [[(t.image, t.row, t.col, t.context) for t in s.tiles] for s in scenes]
    det = Counted(load_detector(args.model))
    model_sha = hashlib.sha256(args.model.read_bytes()).hexdigest()[:16]
    lod = LoDConfig(conf_low=args.det_thr)
    pcfg = PriorityConfig(p1_conf=lod.conf_high)
    ecfg = EncoderConfig(det_thr=args.det_thr, priority=pcfg)
    perceptions = sorted({m.perception for m in modes} | {"sahi"})       # sahi: B0's fair ground row
    n_gt = sum(len(s.gts) for s in scenes)
    print(f"=== WP26 B0-B4, same inputs: {len(scenes)} scenes of {args.per_side}x{args.per_side} real "
          f"tiles ({scenes[0].size} px), {n_gt} ships, modes {[m.name for m in modes]} ===\n")

    # ================================================================ static stage, per scene
    dets = {p: [] for p in perceptions}
    grid_dets, plain_dets, calls = [], [], {p: 0 for p in perceptions}
    pixel_sha, memo = [], []
    gate_a_boxes, gate_a_ok, fusion_violations = 0, True, 0
    jitter = {"tiles": 0, "tiles_with_different_box_count": 0, "boxes": 0, "boxes_not_bit_identical": 0,
              "max_box_diff_px": 0.0, "max_conf_diff": 0.0}
    unfused_total = fused_total = 0
    check1 = check4 = None
    b0_bytes_per_scene = raw_packet_bytes(scenes[0].size, scenes[0].size, ecfg.max_packet_data)
    t0 = time.perf_counter()
    for s in scenes:
        img = render(s, lambda n: load_image(img_dir / n)[0])
        pixel_sha.append(sha(img.tobytes()))
        for p in perceptions:
            c0 = det.calls
            dets[p].append(perceive(img, replace(MODES["B2"], perception=p), det, args.window, args.overlap)
                           if p != "none" else [])
            calls[p] += det.calls - c0
        # the per-source-tile grid (what wp18 called B1) -- a labelled reference, + gate (a)
        memo_det = Memo(det)                             # one batched call; reused by the gate below
        grid = detect_image(img, memo_det, PerceptionConfig(mode="sahi", window=TILE, overlap=0.0, fuse=False))
        grid_dets.append(grid)
        # plain tiling at the SAHI window size, no overlap, no fusion (at 768 that is the grid above)
        plain_dets.append(grid if args.window == TILE else
                          detect_image(img, det, replace(sahi_cfg, overlap=0.0, fuse=False)))
        direct = []
        for t in s.tiles:
            x0, y0 = t.col * TILE, t.row * TILE
            crop = np.ascontiguousarray(img[y0:y0 + TILE, x0:x0 + TILE])
            per_tile = memo_det([crop])[0]               # the SAME detections the grid was given
            direct += [(x0 + x, y0 + y, w, h, c) for (x, y, w, h, c) in per_tile]
            batch_vs_single(per_tile, det([crop])[0], jitter)    # backend determinism, measured
        gate_a_boxes += len(direct)
        gate_a_ok &= sorted(direct) == sorted(grid)      # x_global = x0 + x_local, exactly
        # fusion invariant (check 3): no two fused SAHI boxes overlap at >= nms_iou
        fused = dets["sahi"][-1]
        unfused = detect_image(img, det, replace(sahi_cfg, fuse=False))
        unfused_total, fused_total = unfused_total + len(unfused), fused_total + len(fused)
        fusion_violations += sum(1 for i in range(len(fused)) for j in range(i + 1, len(fused))
                                 if _iou(fused[i][:4], fused[j][:4]) >= 0.5)
        # JPEG memo for this scene: every crop any dark pattern can need (tight, wake, coast tile)
        m = {}

        def jfn(box, q, prog, img=img, m=m):
            key = (box, q, prog)
            if key not in m:
                x0, y0, x1, y1 = box
                m[key] = encode_jpeg(np.ascontiguousarray(img[y0:y1, x0:x1]), q, prog)
            return m[key]
        for dark in (False, True):
            downlink(s, img, fused, MODES["B3"], ecfg, dark=[dark] * len(fused), jpeg_fn=jfn)
        memo.append(m)
        # check 1 (first scene): B0's packets decode to the rendered scene, byte for byte
        if check1 is None and any(mm.payload == "raw_image" for mm in modes):
            c0 = det.calls
            b0 = downlink(s, img, perceive(img, MODES["B0"], det), MODES["B0"], ecfg)
            stream = b"".join(b0.products[0].packets)
            back = decode_downlink(stream)["raw_images"][s.scene_id]
            check1 = {"decoded_equals_scene": bool(np.array_equal(back, img)),
                      "packet_bytes": len(stream), "formula_bytes": b0_bytes_per_scene,
                      "pixels_bytes": img.size, "packets": len(b0.products[0].packets),
                      "detector_calls": det.calls - c0}
        # check 4 (first scene with a coast tile and detections): ground decode round trip
        if check4 is None and any(t.context == "coast" for t in s.tiles) and fused:
            dl = downlink(s, img, fused, MODES["B3"], ecfg, jpeg_fn=jfn, ids=Ids(), packetizer=Packetizer())
            g = decode_downlink(b"".join(q for p in dl.products for q in p.packets))
            check4 = {"scene": s.scene_id, "records_sent": sum(1 for lv in dl.levels if lv > 0),
                      "records_decoded": len(g["records"]), "rois_decoded": len(g["rois"]),
                      "contexts_decoded": len(g["contexts"])}
        if (s.scene_id + 1) % 25 == 0:
            print(f"   scenes {s.scene_id + 1}/{len(scenes)}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    static_s = round(time.perf_counter() - t0, 1)

    # ================================================================ per-mode static outputs
    pairs = {p: [match_pairs([d for d in dd if d[4] >= args.det_thr], s.gts, 0.5)
                 for dd, s in zip(dets[p], scenes)] for p in perceptions}
    # match_pairs indexes the thresholded list; map back to indices in the full list
    def kept(dd):
        return [i for i, d in enumerate(dd) if d[4] >= args.det_thr]
    pair_full = {p: [{kept(dd)[a]: b for a, b in pr.items()} for dd, pr in zip(dets[p], pairs[p])]
                 for p in perceptions}
    report = {"label": "detection metrics REAL; bytes REAL (measured serializations); delivered recall "
                       "and latency SIM-over-REAL; B4 links SIM (rates and powers ASSUMPTION)",
              "inputs": {"scenes": len(scenes), "per_side": args.per_side, "scene_px": scenes[0].size,
                         "ships": n_gt, "unique_tiles": len({t.image for s in scenes for t in s.tiles}),
                         "context_tiles": {c: sum(t.drawn_as == c for s in scenes for t in s.tiles)
                                           for c in ("cloud", "ships", "coast", "empty")},
                         "sahi_window_px": args.window, "sahi_overlap": args.overlap,
                         "static_stage_s": static_s}}
    quality, bytes_per_scene, levels_all = {}, {}, {}
    for m in modes:
        if m.perception == "none":
            quality[m.name] = {"onboard_detection": "none (no detector onboard)",
                               "ground_side_if_delivered": "the ground runs the same SAHI + YOLO "
                               "(B2's metrics); a perfect analyst would see every ship (upper bound)"}
            bytes_per_scene[m.name] = b0_bytes_per_scene
            continue
        dd = dets[m.perception]
        if m.payload == "detections":
            sent = [[d for d in x if d[4] >= args.det_thr] for x in dd]
            sizes = [sum(p.size for p in downlink(s, shape_only(s.size), x, m, ecfg).products)
                     for s, x in zip(scenes, dd)]
        else:
            sent, sizes, lvls = [], [], []
            for s, x, mm in zip(scenes, dd, memo):
                dl = downlink(s, shape_only(s.size), x, m, ecfg, jpeg_fn=lambda b, q, pr, mm=mm: mm[(b, q, pr)])
                sent.append([x[i] for i, lv in enumerate(dl.levels) if lv > 0])
                sizes.append(sum(p.size for p in dl.products))
                lvls.append(dl.levels)
            levels_all[m.name] = lvls
        quality[m.name] = evaluate_detections(list(zip(sent, [s.gts for s in scenes])), conf_thr=args.det_thr)
        quality[m.name]["detections_transmitted"] = sum(len(x) for x in sent)
        bytes_per_scene[m.name] = float(np.mean(sizes))
    quality["B1_per_tile_reference"] = evaluate_detections(
        list(zip([[d for d in x if d[4] >= args.det_thr] for x in grid_dets], [s.gts for s in scenes])),
        conf_thr=args.det_thr)
    quality["B1_per_tile_reference"]["note"] = ("one detector call per 768 px source tile = what wp18 "
                                                "called B1; on a scene it is a 16-window tiling (a slicing "
                                                "stage), so it is NOT the paper's B1")
    quality["plain_tiling_no_overlap"] = evaluate_detections(
        list(zip([[d for d in x if d[4] >= args.det_thr] for x in plain_dets], [s.gts for s in scenes])),
        conf_thr=args.det_thr)
    quality["plain_tiling_no_overlap"] |= {
        "window_px": args.window,
        "windows_per_scene": slice_count(scenes[0].size, scenes[0].size, replace(sahi_cfg, overlap=0.0)),
        "note": "abutting windows of the SAHI window size, no overlap, no fusion"
                + (" (= B1_per_tile_reference at 768 px)" if args.window == TILE else "")}
    report["detection_quality_same_scenes"] = quality
    report["mean_bytes_per_scene"] = bytes_per_scene

    # ================================================================ the same day for every mode
    rng = np.random.default_rng(args.seed + 1)
    n_inst = args.day_tiles // args.per_side ** 2
    inst = rng.integers(0, len(scenes), n_inst)
    times = np.sort(rng.uniform(0, 24 * 3600, n_inst))
    ships, inst_ids = [], []
    for k, (sid, t) in enumerate(zip(inst, times)):
        ids = []
        for g in scenes[sid].gts:
            ships.append(Ship(len(ships), float(t), bool(rng.random() < args.p_dark), 0.0, False,
                              float(max(g[2], g[3]))))
            ids.append(len(ships) - 1)
        inst_ids.append(ids)
    wl = Workload(ships, [(float(t), "scene", tuple(ids)) for t, ids in zip(times, inst_ids)],
                  WorkloadConfig(hours=24, tiles_per_day=args.day_tiles), source="wp26 scenes")
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    passes = find_passes(sat, GroundStation(), start, 36, LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    day_fp = sha([inst.tolist(), [round(t, 3) for t in times], [s.dark for s in ships]])
    passes_fp = sha([(p.rise.isoformat(), round(p.capacity_bytes)) for p in passes])
    # B4's contact plan [SIM]: the SAME primary passes, plus the relay's ISL windows and ground passes
    ccfg = CommsConfig(link=LINK_PRESETS[args.link], link_share=args.link_share,
                       relay_raan_deg=args.b4_raan, lam_E=args.b4_lam_e)
    links = build_links(ccfg, start, 36.0)
    assert passes_fp == sha([(p.rise.isoformat(), round(p.capacity_bytes)) for p in links.primary_passes]), \
        "B4 must use the same primary ground passes as B0-B3"

    def run(items):
        res = simulate(items, passes, start, ValueGreedy(), args.storage_gb * 1e9)
        mt = metrics(res, wl, lod)
        keep = ("ship_recall", "dark_recall", "latency_med_h", "latency_p90_h", "MB_sent", "MB_offered",
                "MB_capacity", "dropped_storage")
        return res, {k: (round(float(mt[k]), 4) if isinstance(mt[k], float) else mt[k]) for k in keep}

    day, fingerprints = {}, {}
    sahi_found = [set(pf.values()) for pf in pair_full["sahi"]]
    for m in modes:
        items = []
        if m.payload == "raw_image":
            for k, sid in enumerate(inst):
                items.append(Item(k, float(times[k]), float(b0_bytes_per_scene), 1.0, "RAW",
                                  tuple(inst_ids[k])))
            res, upper = run(items)
            fair = [replace(it, ships=tuple(inst_ids[k][g] for g in sorted(sahi_found[inst[k]])))
                    for k, it in enumerate(items)]
            _r, ground = run(fair)
            day[m.name] = upper | {"ship_recall_ground_runs_B2": ground["ship_recall"],
                                   "note": "ship_recall credits every ship in a delivered image (perfect "
                                           "analyst); ship_recall_ground_runs_B2 credits what the same "
                                           "SAHI + YOLO finds in it on the ground"}
        else:
            cache = {}
            for k, sid in enumerate(inst):
                s, dd, pf = scenes[sid], dets[m.perception][sid], pair_full[m.perception][sid]
                dark = tuple(bool(ships[inst_ids[k][pf[i]]].dark) if i in pf else False
                             for i in range(len(dd))) if m.payload == "semantic" else ()
                key = (int(sid), dark)
                if key not in cache:
                    mm = memo[sid]
                    dl = downlink(s, shape_only(s.size), dd, m, ecfg, dark=list(dark) or None,
                                  jpeg_fn=lambda b, q, pr, mm=mm: mm[(b, q, pr)])
                    cache[key] = [(p.kind, di, p.size, p.value, p.progressive, p.min_fraction)
                                  for p, di in zip(dl.products, dl.det_index)]
                for (kind, di, size, value, prog, mfrac) in cache[key]:
                    items.append(Item(len(items), float(times[k]), float(size), value, kind,
                                      tuple(inst_ids[k][pf[i]] for i in di if i in pf),
                                      progressive=prog, min_fraction=mfrac if prog else 1.0))
            res, day[m.name] = run(items)
            if m.relay:
                day[m.name] = relay_mode(items, res, day[m.name], links, ccfg, start, wl, lod, args)
        fingerprints[m.name] = {"scenes": sha(manifest), "pixels": sha(pixel_sha),
                                "ground_truth": sha([s.gts for s in scenes]), "model": model_sha,
                                "det_thr": args.det_thr, "matching": "threshold-then-match IoU 0.5",
                                "encoder": sha(repr(ecfg)), "day": day_fp, "passes": passes_fp,
                                "scheduler": "ValueGreedy", "storage_gb": args.storage_gb}
    report["day"] = {"tiles": args.day_tiles, "scene_instances": int(n_inst), "ships": len(ships),
                     "passes": len(passes), "link_share": args.link_share, "results": day}

    # ================================================================ the paper's checks
    checks = {}
    if check1:
        checks["1_B0_original_image"] = {"passed": check1["decoded_equals_scene"] and check1["detector_calls"] == 0
                                         and check1["packet_bytes"] == check1["formula_bytes"], **check1}
    n = len(scenes)
    if "whole" in perceptions:
        frame = scenes[0].size
        gain = letterbox(np.zeros((frame, frame, 3), np.uint8), det.size)[1]
        checks["2_B1_no_slicing"] = {"passed": calls["whole"] == n, "detector_calls_per_scene": calls["whole"] / n,
                                     "frame_px": frame, "letterbox_gain": gain}
    checks["3_B2_sahi_fusion"] = {
        "passed": calls["sahi"] == n * slice_count(scenes[0].size, scenes[0].size, sahi_cfg)
                  and fusion_violations == 0 and gate_a_ok,
        "windows_per_scene": calls["sahi"] / n, "fused_pairs_overlapping_iou_ge_0.5": fusion_violations,
        "boxes_before_fusion": unfused_total, "boxes_after_fusion": fused_total,
        "global_coords_exact_vs_per_tile": bool(gate_a_ok), "boxes_compared": gate_a_boxes,
        "detector_batched_vs_single_call": {k: (round(v, 5) if isinstance(v, float) else int(v))
                                            for k, v in jitter.items()} | {
            "note": "a property of the inference backend, not of the pipeline: bit-identical on ONNX CPU; a "
                    "GPU returns slightly different floats for a 16-crop batch than for single calls, and "
                    "occasionally one box more or fewer (report 12). The coordinate gate above is computed "
                    "on identical per-crop detections, so it is exact on any backend"}}
    if "B3" in levels_all:
        bad = sent_p0 = 0
        for s, dd, lv in zip(scenes, dets["sahi"], levels_all["B3"]):
            for d, level in zip(dd, lv):
                t = s.tiles[s.tile_index(d[0], d[1])]
                want = 0 if t.context == "cloud" else classify(
                    Ship(0, 0.0, False, float(d[4]), t.context == "coast", float(max(d[2], d[3]))), t.context, pcfg, lod)
                bad += want != level
                sent_p0 += level > 0 and d[4] < args.det_thr
        hist = {f"P{k}": sum(lv.count(k) for lv in levels_all["B3"]) for k in range(4)}
        checks["4_B3_semantic_policy"] = {
            "passed": bad == 0 and sent_p0 == 0 and bool(check4)
                      and check4["records_decoded"] == check4["records_sent"],
            "levels_disagreeing_with_priority.classify": bad, "below_cut_transmitted": sent_p0,
            "level_histogram": hist, "ground_round_trip": check4}
    fps = {json.dumps(v, sort_keys=True) for v in fingerprints.values()}
    checks["5_same_inputs"] = {"passed": len(fps) == 1, "fingerprint": next(iter(fingerprints.values())),
                               "modes": list(fingerprints)}
    report["checks"] = checks

    # ================================================================ print
    print(f"\n[static] same {n} scenes ({static_s} s): detection quality of what each mode transmits "
          f"(P/R/F1 @ {args.det_thr}, IoU 0.5)")
    for name in [m.name for m in modes] + ["B1_per_tile_reference", "plain_tiling_no_overlap"]:
        q = quality[name]
        if "precision" in q:
            print(f"   {name:<22} P {q['precision']:.4f}  R {q['recall']:.4f}  F1 {q['F1']:.4f}  "
                  f"AP50 {q['AP50']:.4f}  sent {q['detections_transmitted'] if 'detections_transmitted' in q else '-':>5}"
                  f"  bytes/scene {bytes_per_scene.get(name, float('nan')):12,.0f}")
        else:
            print(f"   {name:<22} no onboard detection; bytes/scene {bytes_per_scene[name]:12,.0f}")
    print(f"\n[day] {n_inst} scene instances = {args.day_tiles} tiles, {len(ships)} ships, {len(passes)} "
          f"passes, capacity {day[modes[0].name]['MB_capacity']} MB, value-greedy, same for every mode")
    for name, r in day.items():
        extra = (f"  (ground runs B2: {r['ship_recall_ground_runs_B2']})" if "ship_recall_ground_runs_B2" in r
                 else f"  [SIM] relay carried {100 * r['totals']['relay_fraction_items']:.0f}% of items, "
                      f"tx energy {r['totals']['energy_J'] / 1e3:.2f} kJ vs {r['direct_only_energy_kJ']} direct-only"
                 if "by_path" in r else "")
        print(f"   {name}  sent {r['MB_sent']:9.2f} / offered {r['MB_offered']:10.2f} MB  recall "
              f"{r['ship_recall']:.4f}  latency med {r['latency_med_h']:.2f} h  dropped {r['dropped_storage']}{extra}")
    print("\n[checks]")
    for k, v in checks.items():
        print(f"   {k:<26} {'PASS' if v['passed'] else 'FAIL'}")
    report["elapsed_s"] = round(time.perf_counter() - t_all, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"wp26_b0_b4{'_' + args.tag if args.tag else ''}.json"
    out_file.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nSaved {out_file} ({report['elapsed_s']} s)")
    return 0 if all(v["passed"] for v in checks.values()) else 1


def relay_mode(items, b3_res, b3_metrics, links, ccfg, start, wl, lod, args) -> dict:
    """B4 [SIM] = the same semantic items as B3, sent over the two-path link simulator (sat7.comms):
    the primary's own ground passes, plus ISL hand-offs to a relay that forwards them on ITS ground
    passes. Capacity, transmit time and energy are computed per stage; the relay never detects."""
    storage = args.storage_gb * 1e9
    off = simulate_comms(items, links, start, ValueGreedy(), replace(ccfg, relay_enabled=False), storage)
    gate = (len(off.result.sent) == len(b3_res.sent) and all(
        a.item is b.item and a.delivered_s == b.delivered_s for a, b in zip(off.result.sent, b3_res.sent)))
    cr = simulate_comms(items, links, start, ValueGreedy(), ccfg, storage)
    mt = metrics(cr.result, wl, lod)
    out = {k: (round(float(mt[k]), 4) if isinstance(mt[k], float) else mt[k]) for k in b3_metrics}
    return out | {"label": f"{LABEL}: simulated links; rates and powers ASSUMPTION (reports 17, 19)",
                  "gate_relay_disabled_equals_B3": bool(gate), "totals": cr.totals(),
                  "by_path": cr.by_path(), "link_use": cr.link_use,
                  "direct_only_energy_kJ": round(off.totals()["energy_J"] / 1e3, 3),
                  "status": "implemented as a SIMULATION: orbits, windows, rates and powers are modeled; "
                            "no relay hardware, no real link"}


if __name__ == "__main__":
    sys.exit(main())
