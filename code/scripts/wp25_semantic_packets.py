"""WP25 -- the P0-P3 semantic downlink with REAL serialized packets (audit C2, C3, D2).

Every downlink size in WP5-WP23 was a model (40 B per report, the WP4 power law per ROI, a 32.4 kB
median per coastal tile). This run builds the actual bytes with sat7.semantic -- binary semantic
records, real JPEG ROI / context crops cut from the real test tiles, CCSDS space packets with
headers and CRC -- and feeds their `len()` to the existing scheduler. Then:

  1. canon     the modeled P0-P3 run exactly as wp23 does it (now without the duplicated dark wake
               crop, priority.py fix) -- the reference
  2. same day  the same 40k-tile day (same sampled tiles, same dark flags, same orbit passes), but
               encoded onboard-style from the detector's real boxes (wp1_predictions.csv), matched
               to ground truth AFTER the 0.25 cut (threshold-then-match, wp24), in three sizings:
               modeled sizes | measured packets (progressive JPEG) | measured, baseline JPEG |
               measured, metadata aggregated per tile
  3. policies  value-greedy vs FIFO vs no-buffer on the measured packets, at nominal load (40k
               tiles/day) and congested (160k), the baselines the repo already supports
  4. sweep     the configurable thresholds (onboard cut, P1/P2 boundary, coastal escalation) ->
               measured MB/day and the risk of missed objects, at nominal load
  5. timing    T_enc (onboard encode) and T_dec (ground decode) per tile, this laptop

Labels: bytes REAL (measured encodings of real tiles); recall / latency SIM-over-REAL (scheduler sim
over real detections); thresholds, values, qualities ASSUMPTION (swept in 4). The orbit capacity
keeps its link-overhead discount (LinkConfig.efficiency 0.8); packet bytes are what the scheduler
sees, so link framing is not counted twice (sat7.semantic, TMFraming).

    .venv/Scripts/python.exe scripts/wp25_semantic_packets.py        # ~3-6 min, CPU, needs the split
Output: results/wp25_semantic_packets.json  (sizes and metrics only -- no pixels are written)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from sat7.datasets import parse_yolo_lines
from sat7.evaluation import match_pairs
from sat7.imagery import load_image
from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite
from sat7.priority import PriorityConfig, encode_priority
from sat7.scheduler import encode_lod
from sat7.real_workload import RealWorkloadConfig, load_catalogue, load_size_model, workload_from_catalogue
from sat7.scheduler import FIFO, Item, LoDConfig, NoBuffer, Ship, ValueGreedy, _l1, _l2, metrics, simulate
from sat7.semantic import (_IMG_HDR, EncoderConfig, Ids, Packetizer, TMFraming, decode_downlink,
                           encode_image, encode_jpeg, pack_record)

TILE = 768
ONBOARD_CTX = {"cloud": "cloud", "land_coast": "coast"}      # pre-filter verdict; else "ships"
SHAPE = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (TILE, TILE, 3))   # shape only (memo path)


class Corpus:
    """The 5,320 real test tiles: detections, ground truth, onboard context, lazy pixels, JPEG memo."""

    def __init__(self, data: Path, tiles, preds_csv: Path):
        self.img_dir, lbl = data / "images" / "test", data / "labels" / "test"
        self.ctx = {r.image: ONBOARD_CTX.get(r.context, "ships") for r in tiles.itertuples()}
        self.preds = {n: [] for n in self.ctx}
        import csv
        with preds_csv.open(newline="") as f:
            for r in csv.DictReader(f):
                self.preds[r["image"]].append(tuple(float(r[k]) for k in ("cx", "cy", "w", "h", "conf")))
        self.gts = {}
        for n in self.ctx:
            p = lbl / (Path(n).stem + ".txt")
            self.gts[n] = parse_yolo_lines(p.read_text().splitlines(), TILE, TILE, {0}) if p.exists() else []
        self.jpeg, self._last = {}, (None, None)
        self.pairs = {}

    def pixels(self, name):
        if self._last[0] != name:
            self._last = (name, load_image(self.img_dir / name)[0])
        return self._last[1]

    def jpeg_fn(self, name):
        def f(box, q, prog):
            key = (name, box, q, prog)
            b = self.jpeg.get(key)
            if b is None:
                x0, y0, x1, y1 = box
                b = encode_jpeg(np.ascontiguousarray(self.pixels(name)[y0:y1, x0:x1]), q, prog)
                self.jpeg[key] = b
            return b
        return f

    def matched(self, name, det_thr):
        """{det index: GT index} at the onboard cut (threshold first, then match -- wp24)."""
        key = (name, det_thr)
        if key not in self.pairs:
            d = self.preds[name]
            keep = [i for i, x in enumerate(d) if x[4] >= det_thr]
            m = match_pairs([d[i] for i in keep], self.gts[name], 0.5)
            self.pairs[key] = {keep[a]: b for a, b in m.items()}
        return self.pairs[key]


def summarise(enc, dets, cfg: EncoderConfig, lod: LoDConfig):
    """Per product: (kind, sub-kind, dets, measured bytes, modeled bytes, value, progressive,
    min_fraction, n_packets, jpeg bytes, jpeg header bytes, packet header bytes). The modeled size
    is what WP5-WP23 would have charged for the same product (40 B / SizeModel / 32.4 kB)."""
    out = []
    for p in enc.products:
        if p.kind == "P1":
            sub, modeled = "P1", lod.l0_bytes * len(p.dets)
        else:
            _pid, _x0, _y0, _w, _h, q, _codec = _IMG_HDR.unpack_from(p.payload, 0)
            s = Ship(0, 0.0, False, 1.0, False, float(max(dets[p.dets[0]][2], dets[p.dets[0]][3])))
            if p.kind == "P3" and q == cfg.coast_tile_quality and q != cfg.ctx_quality:
                sub, modeled = "P3_coast_tile", lod.coast_tile_bytes
            elif p.kind == "P3":
                sub, modeled = "P3_crop", _l2(s, lod)
            elif q == cfg.ctx_quality and q != cfg.roi_quality:
                sub, modeled = "P2_wake", _l2(s, lod)
            else:
                sub, modeled = "P2", _l1(s, lod)
        out.append((p.kind, sub, p.dets, p.size, float(modeled), p.value, p.progressive, p.min_fraction,
                    len(p.packets), p.jpeg_bytes, p.jpeg_header, p.header_bytes))
    return out


def build_day(wl, corpus: Corpus, cfg: EncoderConfig, lod: LoDConfig, cache: dict, sizing: str,
              timing: list | None = None):
    """Scheduler Items for a day of real tiles. sizing: "measured" | "modeled"."""
    ckey = repr(cfg)
    items, n = [], 0
    for idx, (t, _ctx, ids) in enumerate(wl.tiles):
        name = wl.sources[idx]
        dets = corpus.preds[name]
        m = corpus.matched(name, cfg.det_thr)
        dark = tuple(bool(wl.ships[ids[m[k]]].dark) if k in m else False for k in range(len(dets)))
        key = (name, dark, ckey)
        if key not in cache:
            if timing is not None:
                corpus.pixels(name)          # the frame is in memory onboard: keep file I/O out of T_enc
            t0 = time.perf_counter()
            enc = encode_image(SHAPE, 0, 0.0, dets, corpus.ctx[name], cfg, dark=dark,
                               jpeg_fn=corpus.jpeg_fn(name))
            if timing is not None:
                timing.append((corpus.ctx[name], 1e3 * (time.perf_counter() - t0), len(enc.products)))
            cache[key] = summarise(enc, dets, cfg, lod)
        for (kind, _sub, pd, meas, mod, value, prog, mfrac, *_rest) in cache[key]:
            n += 1
            ships = tuple(ids[m[k]] for k in pd if k in m)
            if sizing == "measured":
                items.append(Item(n, t, float(meas), value, kind, ships, progressive=prog,
                                  min_fraction=mfrac if prog else 1.0))
            else:
                items.append(Item(n, t, mod, value, kind, ships, progressive=kind != "P1"))
    return items


def product_stats(wl, cache, cfg, corpus):
    """Measured vs modeled bytes by product kind, over the day (offered)."""
    ckey = repr(cfg)
    rows = {"P1": [], "P2": [], "P2_wake": [], "P3_coast_tile": [], "P3_crop": []}
    hdr = jpg_hdr = jpg = total = mod_total = 0
    for idx, (t, _c, ids) in enumerate(wl.tiles):
        name = wl.sources[idx]
        m = corpus.matched(name, cfg.det_thr)
        dark = tuple(bool(wl.ships[ids[m[k]]].dark) if k in m else False for k in range(len(corpus.preds[name])))
        for (_kind, k, pd, meas, mod, _v, _p, mfrac, npk, jb, jh, hb) in cache[(name, dark, ckey)]:
            rows[k].append((meas, mod, mfrac, npk))
            hdr, jpg_hdr, jpg, total, mod_total = hdr + hb, jpg_hdr + jh, jpg + jb, total + meas, mod_total + mod
    out = {}
    for k, r in rows.items():
        if not r:
            continue
        meas = [x[0] for x in r]
        out[k] = {"count": len(r), "MB": round(sum(meas) / 1e6, 3),
                  "median_B": statistics.median(meas), "p90_B": float(np.percentile(meas, 90)),
                  "modeled_MB": round(sum(x[1] for x in r) / 1e6, 3),
                  "measured_over_modeled": round(sum(meas) / max(1e-9, sum(x[1] for x in r)), 3),
                  "median_min_fraction": round(statistics.median(x[2] for x in r), 3),
                  "packets": int(sum(x[3] for x in r))}
    tm = TMFraming()
    out["totals"] = {"MB_measured": round(total / 1e6, 3), "MB_modeled": round(mod_total / 1e6, 3),
                     "packet_header_crc_share": round(hdr / total, 4) if total else None,
                     "jpeg_header_share": round(jpg_hdr / total, 4) if total else None,
                     "jpeg_payload_share": round(jpg / total, 4) if total else None,
                     "on_air_MB_ccsds_tm": round(tm.on_air_bytes(total) / 1e6, 3),
                     "tm_efficiency": round(tm.efficiency, 4),
                     "capacity_efficiency_assumed": 0.8}
    return out


def run(items, wl, lod, passes, start, policy, storage_gb, enc, with_value: bool = False):
    """value_frac only for GT-catalogue encodings: onboard, a false alarm carries a report's value
    (the satellite cannot tell), so value / ground-truth value is not meaningful for the
    detection-based items."""
    res = simulate(items, passes, start, policy, storage_gb * 1e9, encoder=enc)
    m = metrics(res, wl, lod)
    keep = ("ship_recall", "dark_recall", "latency_med_h", "latency_p90_h", "MB_sent", "MB_offered",
            "MB_capacity") + (("value_frac",) if with_value else ())
    return {k: (round(float(m[k]), 4) if isinstance(m[k], float) else m[k]) for k in keep} | {
        "policy": policy.name, "n_items": len(items)}


def lod_with_measured_coast(wl, lod, corpus: Corpus, quality: int = 40):
    """The LoD ladder (the 557x headline encoder), item for item as scheduler.encode_lod builds it,
    with each coastal tile's size replaced by its MEASURED encoding (baseline JPEG q40 -- what WP7 /
    WP13 measured -- + an 18 B image header + packet headers). Everything else keeps its model."""
    from sat7.scheduler import _Ids, encode_tile
    pk_over = lambda n: n + 8 * math.ceil(n / (4096 - 2))       # CCSDS headers + CRC per segment
    nid, rng, items, swaps = _Ids(), np.random.default_rng(wl.cfg.seed + 1), [], []
    for idx in range(len(wl.tiles)):
        for it in encode_tile(wl, idx, lod, nid, rng):
            if it.kind == "tile":
                name = wl.sources[idx]
                n = len(corpus.jpeg_fn(name)((0, 0, TILE, TILE), quality, False)) + _IMG_HDR.size
                swaps.append((it.size, pk_over(n)))
                it.size = float(pk_over(n))
            items.append(it)
    return items, swaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--day-tiles", type=int, default=40_000)
    ap.add_argument("--congested-tiles", type=int, default=160_000)
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--link", default="cubesat_sband")
    ap.add_argument("--link-share", type=float, default=0.25)
    ap.add_argument("--storage-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    res_dir = ROOT / "results"
    t_all = time.perf_counter()

    tiles, ships = load_catalogue(res_dir)
    corpus = Corpus(args.data, tiles, res_dir / "wp1_predictions.csv")
    lod = LoDConfig(conf_low=args.det_thr, size_model=load_size_model(res_dir / "wp6_size_model.json"))
    pcfg = PriorityConfig(p1_conf=lod.conf_high)
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    passes = find_passes(sat, GroundStation(), start, 36, LINK_PRESETS[args.link])
    passes = [replace(p, capacity_bytes=p.capacity_bytes * args.link_share) for p in passes]
    wl, stats = workload_from_catalogue(tiles, ships, RealWorkloadConfig(
        tiles_per_day=args.day_tiles, det_thr=args.det_thr, seed=args.seed))
    print(f"=== WP25 semantic downlink with real packets: {stats.tiles} tiles, {stats.ships} ships, "
          f"{len(passes)} passes, link_share {args.link_share} ===\n")
    report = {"label": "bytes REAL (measured encodings of real Airbus test tiles); recall/latency "
                       "SIM-over-REAL; thresholds/values/qualities ASSUMPTION",
              "day": {"tiles": stats.tiles, "ships": stats.ships, "passes": len(passes),
                      "link_share": args.link_share, "MB_capacity": round(sum(p.capacity_bytes for p in passes) / 1e6, 1)}}

    # ---- 1. canon: the modeled P0-P3 run, as wp23
    canon = run(encode_priority(wl, lod, pcfg), wl, lod, passes, start, ValueGreedy(), args.storage_gb,
                "P0-P3 modeled (wp23 method)", with_value=True)
    w23 = json.loads((res_dir / "wp23_semantic_compare.json").read_text())["semantic_schemes"]["priority"]
    canon["wp23_committed"] = {"MB_sent": w23["MB_sent"], "ship_recall": w23["ship_recall"],
                               "note": "committed before the dark-wake de-duplication (priority.py)"}
    report["canon_modeled"] = canon
    print(f"[1] canon modeled P0-P3: {canon['MB_sent']} MB, recall {canon['ship_recall']}  "
          f"(wp23 committed {w23['MB_sent']} MB / {w23['ship_recall']}; the difference is the "
          f"de-duplicated dark wake crop)")

    # ---- 2. the same day, onboard-style from real boxes: modeled vs measured sizes
    base = EncoderConfig(det_thr=args.det_thr, priority=pcfg)
    variants = {"measured_progressive": base,
                "measured_baseline_jpeg": replace(base, progressive=False),
                "measured_tile_metadata": replace(base, metadata="tile")}
    cache, timing, same_day = {}, [], {}
    items_meas = build_day(wl, corpus, base, lod, cache, "measured", timing)
    same_day["modeled_sizes"] = run(build_day(wl, corpus, base, lod, cache, "modeled"), wl, lod, passes,
                                    start, ValueGreedy(), args.storage_gb, "modeled sizes")
    same_day["measured_progressive"] = run(items_meas, wl, lod, passes, start, ValueGreedy(),
                                           args.storage_gb, "measured")
    for name in ("measured_baseline_jpeg", "measured_tile_metadata"):
        same_day[name] = run(build_day(wl, corpus, variants[name], lod, cache, "measured"), wl, lod,
                             passes, start, ValueGreedy(), args.storage_gb, name)
    report["same_day_detection_based"] = same_day
    report["product_sizes"] = {name: product_stats(wl, cache, cfg, corpus) for name, cfg in variants.items()}
    print("[2] same day, from real boxes (value-greedy):")
    for name, r in same_day.items():
        print(f"     {name:<24} {r['MB_sent']:8.2f} MB sent  recall {r['ship_recall']:.4f}  "
              f"dark {r['dark_recall']:.4f}  latency med {r['latency_med_h']:.2f} h")
    ps = report["product_sizes"]["measured_progressive"]
    for k in ("P1", "P2", "P2_wake", "P3_coast_tile", "P3_crop"):
        if k in ps:
            v = ps[k]
            print(f"     {k:<14} n {v['count']:6d}  {v['MB']:7.2f} MB  median {v['median_B']:8.0f} B  "
                  f"p90 {v['p90_B']:8.0f} B  measured/modeled {v['measured_over_modeled']:.2f}  "
                  f"min decodable {v['median_min_fraction']}")
    tt = ps["totals"]
    print(f"     headers+CRC {100 * tt['packet_header_crc_share']:.2f}% | JPEG headers "
          f"{100 * tt['jpeg_header_share']:.1f}% | on-air (CCSDS TM, eff {tt['tm_efficiency']}) "
          f"{tt['on_air_MB_ccsds_tm']} MB")

    # ---- 2b. the coastal context tile: measured on EVERY real coastal tile, and its effect on the
    #          LoD ladder (the 557x headline: coastal tiles were 62% of its bytes at 32,420 B each)
    coast_names = sorted(n for n, c in corpus.ctx.items() if c == "coast")
    sizes = {}
    for prog in (False, True):
        b = [len(corpus.jpeg_fn(n)((0, 0, TILE, TILE), 40, prog)) for n in coast_names]
        sizes["progressive" if prog else "baseline"] = {
            "tiles": len(b), "median_B": statistics.median(b), "mean_B": round(statistics.mean(b), 1),
            "p10_B": float(np.percentile(b, 10)), "p90_B": float(np.percentile(b, 90))}
    lod_canon = run(encode_lod(wl, lod), wl, lod, passes, start, ValueGreedy(), args.storage_gb,
                    "LoD modeled (wp6/wp23)", with_value=True)
    lod_items, swaps = lod_with_measured_coast(wl, lod, corpus)
    lod_meas = run(lod_items, wl, lod, passes, start, ValueGreedy(), args.storage_gb,
                   "LoD, measured coastal tiles", with_value=True)
    report["coastal_context_tile_q40"] = {"measured_all_coastal_test_tiles": sizes,
                                          "modeled_LoDConfig_coast_tile_bytes": LoDConfig().coast_tile_bytes,
                                          "why_they_differ": "LoDConfig took WP7's 'whole tile' ladder, "
                                          "which sampled 400 random tiles WITH SHIPS (mostly open sea); "
                                          "WP13 measured real coastal tiles at 65.5 kB (q40)"}
    report["LoD_headline_with_measured_coast"] = {
        "modeled": lod_canon, "measured_coast": lod_meas,
        "coastal_tiles_swapped": len(swaps),
        "coastal_MB_modeled": round(sum(a for a, _ in swaps) / 1e6, 2),
        "coastal_MB_measured": round(sum(b for _, b in swaps) / 1e6, 2)}
    print(f"[2b] coastal context tile, q40, all {len(coast_names)} real coastal tiles: baseline median "
          f"{sizes['baseline']['median_B']:.0f} B, progressive {sizes['progressive']['median_B']:.0f} B "
          f"(modeled {LoDConfig().coast_tile_bytes:.0f} B)")
    print(f"     LoD ladder (headline encoder): modeled {lod_canon['MB_offered']} MB offered / "
          f"{lod_canon['MB_sent']} sent, recall {lod_canon['ship_recall']}  ->  measured coastal tiles "
          f"{lod_meas['MB_offered']} MB offered / {lod_meas['MB_sent']} sent, recall {lod_meas['ship_recall']} "
          f"(capacity {lod_meas['MB_capacity']} MB)")

    # ---- 3. policies, nominal and congested, on measured packets
    pol = {"nominal_40k": {}, "congested": {}}
    for policy in (ValueGreedy(), FIFO(), NoBuffer()):
        pol["nominal_40k"][policy.name] = run(items_meas, wl, lod, passes, start, policy, args.storage_gb,
                                              "measured")
    wlc, sc = workload_from_catalogue(tiles, ships, RealWorkloadConfig(
        tiles_per_day=args.congested_tiles, det_thr=args.det_thr, seed=args.seed))
    items_c = build_day(wlc, corpus, base, lod, cache, "measured")
    for policy in (ValueGreedy(), FIFO(), NoBuffer()):
        pol["congested"][policy.name] = run(items_c, wlc, lod, passes, start, policy, args.storage_gb,
                                            "measured")
    pol["congested"]["tiles"] = sc.tiles
    pol["congested"]["canon_modeled_value_greedy"] = run(encode_priority(wlc, lod, pcfg), wlc, lod, passes,
                                                         start, ValueGreedy(), args.storage_gb, "modeled")
    report["policies_measured_packets"] = pol
    print(f"[3] policies on measured packets:")
    for load in ("nominal_40k", "congested"):
        for name, r in pol[load].items():
            if isinstance(r, dict):
                print(f"     {load:<12} {name:<28} {r['MB_sent']:8.2f} / {r['MB_offered']:8.2f} MB  "
                      f"recall {r['ship_recall']:.4f}  "
                      f"latency {r['latency_med_h']:.2f} h")

    # ---- 4. threshold sweep at nominal load (offered == sent there): volume vs missed objects
    if not args.no_sweep:
        sweep = []
        cloud_ships = {i for t, c, ids in wl.tiles if c == "cloud" for i in ids}
        n_ships = len(wl.ships)
        for det_thr in (0.15, 0.25, 0.35, 0.50):
            for p1 in (0.50, 0.67, 0.80, 0.90):
                for esc in ("all", "uncertain"):
                    cfg = replace(base, det_thr=det_thr,
                                  priority=PriorityConfig(p1_conf=p1, coast_escalation=esc))
                    its = build_day(wl, corpus, cfg, lod, cache, "measured")
                    seen = {s for it in its for s in it.ships}
                    imaged = {s for it in its if it.kind != "P1" for s in it.ships}
                    p1_items = [it for it in its if it.kind == "P1"]
                    sweep.append({"det_thr": det_thr, "p1_conf": p1, "coast_escalation": esc,
                                  "MB_per_day": round(sum(it.size for it in its) / 1e6, 2),
                                  "records": len(p1_items),
                                  "false_alarm_records": sum(1 for it in p1_items if not it.ships),
                                  "rois": sum(1 for it in its if it.kind == "P2"),
                                  "contexts": sum(1 for it in its if it.kind == "P3"),
                                  "recall_ceiling": round(len(seen) / n_ships, 4),
                                  "recall_non_cloud": round(len(seen) / (n_ships - len(cloud_ships)), 4),
                                  "ships_with_image_evidence": round(len(imaged) / max(1, len(seen)), 4)})
        report["threshold_sweep_nominal"] = sweep
        print("[4] threshold sweep (nominal load; offered = sent):")
        print(f"     {'cut':>5} {'p1':>5} {'coast':>9} {'MB/day':>7} {'recall':>7} {'non-cloud':>9} "
              f"{'FA rec':>7} {'img ev':>7}")
        for r in sweep:
            if r["coast_escalation"] == "all" or r["p1_conf"] == 0.67:
                print(f"     {r['det_thr']:5.2f} {r['p1_conf']:5.2f} {r['coast_escalation']:>9} "
                      f"{r['MB_per_day']:7.2f} {r['recall_ceiling']:7.4f} {r['recall_non_cloud']:9.4f} "
                      f"{r['false_alarm_records']:7d} {r['ships_with_image_evidence']:7.3f}")

    # ---- 5. timing: onboard encode (cold JPEG memo) and ground decode, per tile
    enc_ms = {}
    for ctx, ms, n_products in timing:
        if n_products:                                  # tiles that actually send something
            enc_ms.setdefault(ctx, []).append(ms)
    names = sorted(corpus.ctx)[:300]
    dec_ms, pk, ids = [], Packetizer(), Ids()
    for n in names:
        if corpus.ctx[n] == "cloud":
            continue
        enc = encode_image(corpus.pixels(n), 1, 0.0, corpus.preds[n], corpus.ctx[n], base, ids, pk)
        stream = b"".join(q for p in enc.products for q in p.packets)
        t0 = time.perf_counter()
        g = decode_downlink(stream)
        for d in list(g["rois"].values()) + list(g["contexts"].values()):
            cv2.imdecode(np.frombuffer(d["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        dec_ms.append(1e3 * (time.perf_counter() - t0))
    report["timing_ms_per_tile"] = {
        "T_enc_onboard_median": {c: round(statistics.median(v), 2) for c, v in enc_ms.items()},
        "T_enc_onboard_p90": {c: round(float(np.percentile(v, 90)), 2) for c, v in enc_ms.items()},
        "T_enc_tiles_timed": {c: len(v) for c, v in enc_ms.items()},
        "T_dec_ground_median": round(statistics.median(dec_ms), 2),
        "T_dec_ground_p90": round(float(np.percentile(dec_ms, 90)), 2),
        "note": "this laptop's CPU (not flight hardware); encode includes the JPEG crops"}
    print(f"[5] T_enc median per tile {report['timing_ms_per_tile']['T_enc_onboard_median']} ms; "
          f"T_dec median {report['timing_ms_per_tile']['T_dec_ground_median']} ms")

    # ---- sample record packets (metadata only -- no pixels leave this script)
    sample = next(n for n in names if corpus.ctx[n] == "ships" and len(corpus.preds[n]) >= 3)
    enc = encode_image(corpus.pixels(sample), 42, 3600.5, corpus.preds[sample], "ships", base, Ids(), Packetizer())
    report["sample_record_packets"] = {"image": sample, "packets_hex": [
        p.packets[0].hex() for p in enc.products if p.kind == "P1"][:3],
        "record_bytes": [len(pack_record(r)) for r in enc.records]}
    report["elapsed_s"] = round(time.perf_counter() - t_all, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wp25_semantic_packets.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {args.out / 'wp25_semantic_packets.json'} ({report['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
