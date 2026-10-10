"""IASTAM P7 -- QUICKSTART: the onboard chain, live, on a laptop CPU, in well under a minute.

    python demo/quickstart/run_demo.py                     # bundled synthetic stand-in tiles
    python demo/quickstart/run_demo.py --data code/data/yolo_ships   # the REAL tiles, if you have them

No GPU, no torch, no dataset download. Needs: numpy, opencv-python-headless, onnxruntime, sgp4,
skyfield (see requirements.txt; the last two only because sat7.scheduler imports sat7.orbit).

What runs LIVE, in order, on a 3x3 swath of 768 px tiles (2304 x 2304 px):
  1. classic pre-filter per tile (sat7.prefilter)                -> cloud / coast / sea context
  2. B1: trained YOLOv8n per tile, ONNX FP32 on CPU                -> whole-tile detections
  3. B2: SAHI over the swath + global-coordinate fusion (sat7.perception, the same code as wp16/wp18)
  4. paper Table I P0-P3 decision per detection (sat7.priority.classify)
  5. the semantic packet (sat7.priority.encode_priority) vs raw bytes (B0 definition)
  6. the value-greedy downlink order over that packet (sat7.scheduler.ValueGreedy)
Then, read from committed results (nothing recomputed): the same 9 tile IDs through the same
packet builder using the REAL detections in code/results/wp1_predictions.csv, and the measured
headline (data reduction, B0-B4) from code/results.

HONESTY. The bundled tiles are SYNTHETIC stand-ins (Airbus rules forbid redistributing the real
ones; see README.md). Every number computed on them is labelled SYNTH: it proves the chain
executes end to end, it is not a measurement. Every other number is read from a committed
results file and keeps the label it has there (REAL / SIM / SIM-over-REAL / ASSUMPTION). With
--data pointing at the real Airbus test split, the live numbers become REAL-sample numbers and
the live boxes are cross-checked against the committed wp1 predictions (decision flips).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CODE = REPO / "code"
RESULTS = CODE / "results"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE.parent))

import host_power  # noqa: E402  (demo/host_power.py: keeps Windows from throttling this process)
from sat7.b2_sahi_fusion import _iou, plan_slices  # noqa: E402  (_iou: the rule wp12/wp16 validated)
from sat7.perception import (OnnxDetector, PerceptionConfig, decode_yolo,  # noqa: E402
                             detect_image, nms_xyxy, slice_count)
from sat7.prefilter import CLOUD, LAND, run_prefilter  # noqa: E402
from sat7.priority import LEVEL_NAME, P0, PriorityConfig, classify, encode_priority  # noqa: E402
from sat7.semantic import (EncoderConfig, Ids, Packetizer, TMFraming, decode_downlink,  # noqa: E402
                           encode_image, pack_record, to_items)
from sat7.scheduler import (COAST_TILE_BYTES_MEASURED, RAW_TILE_BYTES, LoDConfig, Ship,  # noqa: E402
                            SizeModel,
                            ValueGreedy, Workload, WorkloadConfig, encode_lod)

TILE = 768
DET_THR = 0.25     # onboard operating threshold -- every wp script overrides LoDConfig.conf_low to it
RAW_CONF = 0.05    # raw dump threshold, the one wp1_predictions.csv was written at
NMS_IOU = 0.70     # ultralytics predict default, i.e. what produced wp1_predictions.csv
RULE, THIN = "=" * 92, "-" * 92


# ================================================================================ detector
# The ONNX detector adapter (onnxruntime on CPU, ultralytics-equivalent post-processing) lives in
# sat7.perception, where the rest of the pipeline can load it too; the names below are kept so this
# script and code/tests/test_quickstart.py read as before.
OnnxYolo = OnnxDetector
decode = decode_yolo


# ================================================================================ tiles

@dataclass
class Tile:
    slot: int
    row: int
    col: int
    file: str
    ref_id: str                    # the REAL test tile this one is / stands in for
    synthetic: bool
    img: np.ndarray
    ref_context: str = ""          # pre-filter context of the REAL tile (wp6_tiles.csv)
    live_context: str = ""         # pre-filter context computed live on THIS image
    ctx: str = ""                  # encoder vocabulary: cloud | coast | ships | empty
    b1: list = field(default_factory=list)
    dets: list = field(default_factory=list)   # B2 fused detections centred in this tile, global px


def load_tiles(data: Path | None) -> tuple[list[Tile], dict]:
    manifest = json.loads((HERE / "tiles" / "manifest.json").read_text())
    tiles = []
    for e in manifest["tiles"]:
        if data is None:
            path = HERE / "tiles" / e["file"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != e["sha256"]:
                print(f"    (!) {e['file']} differs from manifest.json -- regenerated or edited?")
            synthetic = True
        else:
            cands = [data / "images" / "test" / e["stands_in_for"], data / e["stands_in_for"]]
            path = next((p for p in cands if p.exists()), None)
            if path is None:
                sys.exit(f"--data: {e['stands_in_for']} not found under {data} "
                         f"(expected images/test/<id>.jpg, as built by wp0_build_dataset.py)")
            raw = path.read_bytes()
            synthetic = False
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (TILE, TILE):
            sys.exit(f"{path}: expected a {TILE}x{TILE} image")
        tiles.append(Tile(e["slot"], e["row"], e["col"], path.name, e["stands_in_for"],
                          synthetic, img, ref_context=e["reference_context"]))
    return tiles, manifest


def context_of(prefilter_context: str, n_dets: int) -> str:
    """Pre-filter context -> the encoder's tile vocabulary (the wp11 mapping, but onboard: the
    satellite has no ground truth, so 'ships' means 'the detector reported something here')."""
    if prefilter_context == CLOUD:
        return "cloud"
    if prefilter_context == LAND:
        return "coast"
    return "ships" if n_dets else "empty"


# ================================================================================ encoding

def load_size_model() -> SizeModel:
    """The WP4-measured power law bytes = a * px^b (same fields as sat7.real_workload.load_size_model,
    read without pandas so the quickstart stays light)."""
    m = json.loads((RESULTS / "wp6_size_model.json").read_text())
    return SizeModel(l1_a=m["L1"]["a"], l1_b=m["L1"]["b"], l2_a=m["L2"]["a"], l2_b=m["L2"]["b"])


def build_workload(contexts: list[str], dets_per_tile: list[list]) -> tuple[Workload, list[int]]:
    """One sat7 Workload whose 'ships' are the onboard detections.

    Onboard there is no ground truth, so every detection is reported on its merits; a detector false
    alarm is costed exactly like a real ship at the same confidence (the repo's catalogue runs cost
    it as a zero-value 40 B report instead -- the difference is noted in README.md). Ship length
    is the box's long side, the input the WP4 size model was fitted on. No AIS feed -> dark=False.
    """
    ships, rows, tile_of = [], [], []
    for k, (ctx, dets) in enumerate(zip(contexts, dets_per_tile)):
        ids = []
        for (_cx, _cy, w, h, conf) in dets:
            ships.append(Ship(len(ships), 0.0, dark=False, confidence=float(conf),
                              coastal=ctx == "coast", size_px=float(max(w, h))))
            ids.append(len(ships) - 1)
            tile_of.append(k)
        rows.append((0.0, ctx, tuple(ids)))
    wl = Workload(ships, rows, WorkloadConfig(), false_alarms=[0] * len(rows),
                  source="quickstart detections")
    return wl, tile_of


def level_of(ship: Ship, ctx: str, pcfg: PriorityConfig, lod: LoDConfig) -> int:
    return P0 if ctx == "cloud" else classify(ship, ctx, pcfg, lod)   # cloud: dropped by context


def packet_bytes(items) -> dict:
    by = {"P1": [0, 0.0], "P2": [0, 0.0], "P3": [0, 0.0]}
    for it in items:
        by[it.kind][0] += 1
        by[it.kind][1] += it.size
    return by


# ================================================================================ reference data

def _csv(name: str) -> list[dict]:
    p = RESULTS / name
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def _json(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def committed_predictions(ids: set[str]) -> dict[str, list[tuple]]:
    out = {i: [] for i in ids}
    for r in _csv("wp1_predictions.csv"):
        if r["image"] in ids:
            out[r["image"]].append(tuple(float(r[k]) for k in ("cx", "cy", "w", "h", "conf")))
    return out


def cross_check(live: list[tuple], ref: list[tuple], thresholds: dict) -> dict:
    """Pair live boxes with committed ones (greedy, IoU >= 0.5) and count decision flips."""
    used, pairs = set(), []
    for d in sorted(live, key=lambda z: -z[4]):
        best, bj = 0.0, -1
        for j, c in enumerate(ref):
            if j not in used and (v := _iou(d[:4], c[:4])) > best:
                best, bj = v, j
        if best >= 0.5:
            used.add(bj)
            pairs.append((d, ref[bj]))
    flips = {}
    for name, thr in thresholds.items():
        f = sum((a[4] >= thr) != (b[4] >= thr) for a, b in pairs)
        f += sum(d[4] >= thr for d in live if all(d is not a for a, _ in pairs))
        f += sum(r[4] >= thr for j, r in enumerate(ref) if j not in used)
        flips[name] = int(f)
    return {"live": len(live), "committed": len(ref), "paired": len(pairs),
            "max_dconf": max((abs(a[4] - b[4]) for a, b in pairs), default=0.0), "flips": flips}


# ================================================================================ figure

LEVEL_BGR = {0: (150, 150, 150), 1: (80, 220, 80), 2: (40, 215, 255), 3: (40, 110, 255)}


def _text(img, s, org, scale, colour, thick=2) -> None:
    """Outlined text, readable on sea, sand and cloud alike."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def draw_swath(swath, tiles, windows, records, synthetic: bool, path: Path) -> None:
    img = swath.copy()
    for (x, y, w, h) in windows:                                   # SAHI windows
        cv2.rectangle(img, (x + 2, y + 2), (x + w - 3, y + h - 3), (200, 200, 120), 1)
    for k in range(1, 3):                                          # tile seams
        cv2.line(img, (k * TILE, 0), (k * TILE, 3 * TILE), (255, 255, 255), 2)
        cv2.line(img, (0, k * TILE), (3 * TILE, k * TILE), (255, 255, 255), 2)
    for r in records:
        cx, cy, w, h = r["box"]
        c = LEVEL_BGR[r["level"]]
        p0, p1 = (int(cx - w / 2) - 3, int(cy - h / 2) - 3), (int(cx + w / 2) + 3, int(cy + h / 2) + 3)
        cv2.rectangle(img, p0, p1, c, 3 if r["level"] else 1)
        if r["level"]:
            _text(img, f"P{r['level']} {r['conf']:.2f}", (p0[0], max(14, p0[1] - 6)), 0.7, c)
    for t in tiles:
        _text(img, f"{t.slot + 1}: {t.ctx}", (t.col * TILE + 12, t.row * TILE + 34), 1.0, (255, 255, 255))
    x = 20
    for lv, name in ((1, "P1 metadata"), (2, "P2 +ROI"), (3, "P3 +context"), (0, "P0 discard")):
        _text(img, name, (x, 3 * TILE - 70), 0.9, LEVEL_BGR[lv])
        x += 230
    if synthetic:
        _text(img, "SYNTHETIC STAND-IN TILES - NOT AIRBUS IMAGERY - NOT A MEASUREMENT",
              (20, 3 * TILE - 24), 1.25, (255, 255, 255), 3)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ================================================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None,
                    help="real Airbus split root (e.g. code/data/yolo_ships); default = bundled synthetic tiles")
    ap.add_argument("--model", type=Path, default=HERE / "model" / "best.onnx")
    ap.add_argument("--window", type=int, default=TILE, help="SAHI window px (768 = detector-native)")
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--out", type=Path, default=HERE / "_out")
    args = ap.parse_args()
    t_start = time.perf_counter()
    host_power.keep_full_speed()        # a terminal behind another window is throttled on battery

    if not args.model.exists():
        sys.exit(f"model not found: {args.model}\n  It is checked in at demo/quickstart/model/best.onnx;"
                 f" if you cloned without it, re-export from code/runs/ships/weights/best.pt (wp1).")

    tiles, manifest = load_tiles(args.data)
    synthetic = tiles[0].synthetic
    live_lbl = "SYNTH" if synthetic else "REAL-sample"
    lod = LoDConfig(conf_low=DET_THR, size_model=load_size_model())        # wp18/wp23 settings
    pcfg = PriorityConfig(p1_conf=lod.conf_high)                            # share the LoD point

    print("\n" + RULE)
    print("  IASTAM 6.0 - Problem 7 - QUICKSTART: onboard semantic downlink, live on CPU")
    src = "9 SYNTHETIC stand-in tiles (no Airbus pixels)" if synthetic else "9 REAL Airbus test tiles"
    print(f"  {src} -> pre-filter -> YOLO+SAHI -> P0-P3 packet -> scheduler")
    print(RULE)
    print("  LABELS  REAL / SIM / SIM-over-REAL / ASSUMPTION exactly as in the repo README, plus")
    if synthetic:
        print("          SYNTH = computed live on the synthetic stand-ins. It proves the chain runs end")
        print("          to end; it is NOT a measurement and is never compared with the headline.")
    else:
        print("          REAL-sample = computed live on 9 real tiles: real, but a 9-tile sample, not")
        print("          the headline (which is a 40k-tile day, section [7]).")

    # ------------------------------------------------------------------ [1] input + pre-filter
    t0 = time.perf_counter()
    for t in tiles:
        t.live_context = run_prefilter(t.img).context
    pre_ms = 1e3 * (time.perf_counter() - t0)
    swath = np.vstack([np.hstack([t.img for t in tiles[r * 3:r * 3 + 3]]) for r in range(3)])
    print(f"\n[1] INPUT  3x3 swath of {TILE} px tiles = {swath.shape[1]}x{swath.shape[0]} px, "
          f"stitched abutting (wp15 convention)")
    print(f"    {'#':<3}{'file':<30}{'stands in for':<16}{'pre-filter LIVE':<17}{'catalogue [REAL]':<17}")
    for t in tiles:
        mark = "" if t.live_context == t.ref_context else "  <- differs"
        print(f"    {t.slot + 1:<3}{t.file:<30}{t.ref_id:<16}{t.live_context:<17}{t.ref_context:<17}{mark}")

    # ------------------------------------------------------------------ [2] perception
    t0 = time.perf_counter()
    det = OnnxYolo(args.model)
    load_s = time.perf_counter() - t0
    for t in tiles:                                    # B1: one call on each whole tile
        t.b1 = det([t.img])[0]
    b1_ms, b1_calls = det.ms, det.calls
    pcfg_sahi = PerceptionConfig(mode="sahi", window=args.window, overlap=args.overlap)
    fused = detect_image(swath, det, pcfg_sahi)        # B2: sat7.perception -> b2_sahi_fusion
    windows = plan_slices(swath.shape[1], swath.shape[0], pcfg_sahi.to_sahi())
    b2_ms, b2_calls = det.ms - b1_ms, det.calls - b1_calls
    for d in fused:                                    # each fused box belongs to the tile its centre is in
        r, c = min(2, int(d[1] // TILE)), min(2, int(d[0] // TILE))
        tiles[r * 3 + c].dets.append(d)
    n_b1 = sum(sum(d[4] >= DET_THR for d in t.b1) for t in tiles)
    n_b2 = sum(d[4] >= DET_THR for d in fused)
    n_sahi = slice_count(swath.shape[1], swath.shape[0], pcfg_sahi)
    print(f"\n[2] PERCEPTION  trained YOLOv8n, ONNX FP32, CPU (onnxruntime {det.ort_version}); "
          f"raw dump >= {RAW_CONF}, onboard cut {DET_THR}")
    print(f"    B1 whole tile : {b1_calls} detector calls -> {n_b1} boxes >= {DET_THR}   [{live_lbl}]")
    print(f"    B2 SAHI+fusion: {n_sahi} windows ({args.window} px, {args.overlap:.0%} overlap) -> "
          f"{len(fused)} fused boxes, {n_b2} >= {DET_THR}   [{live_lbl}]")
    print(f"    compute B2/B1 : {n_sahi}/{b1_calls} detector calls on this 3x3 swath (geometry; the "
          f"measured 4x4 figure is in [7])")
    print(f"    time          : model load {load_s:.2f} s, pre-filter {pre_ms / 1e3:.2f} s, "
          f"B1 {b1_ms / 1e3:.2f} s, B2 {b2_ms / 1e3:.2f} s  "
          f"({det.ms / det.calls:.0f} ms/call)   [LIVE, this machine]")

    # ------------------------------------------------------------------ [3] P0-P3 per detection
    for t in tiles:
        t.ctx = context_of(t.live_context, len(t.dets))
    wl, tile_of = build_workload([t.ctx for t in tiles], [t.dets for t in tiles])
    flat = [d for t in tiles for d in t.dets]          # ship i <-> flat[i], same order as build_workload
    # the REAL packets (sat7.semantic): records, JPEG crops cut from these pixels, CCSDS packets
    ecfg = EncoderConfig(det_thr=DET_THR, priority=pcfg)
    pk, ids = Packetizer(ecfg.max_packet_data), Ids()
    encs = []
    for t in tiles:
        x0, y0 = t.col * TILE, t.row * TILE
        local = [(cx - x0, cy - y0, w, h, c) for (cx, cy, w, h, c) in t.dets]   # global -> tile pixels
        encs.append((t, encode_image(t.img, t.slot + 1, 0.0, local, t.ctx, ecfg, ids, pk, origin=(x0, y0))))
    offs, n = [], 0
    for t in tiles:
        offs.append(n)
        n += len(t.dets)
    items = []
    for (t, enc), off in zip(encs, offs):
        items += to_items(enc, 0.0, ships_of=lambda k, off=off: (off + k,), next_id=lambda: len(items) + 1)
    products = [(t, p_) for t, enc in encs for p_ in enc.products]
    own = {i: 0.0 for i in range(len(wl.ships))}
    shared = {k: 0.0 for k in range(len(tiles))}
    for (t, p_), off in ((tp, offs[tp[0].slot]) for tp in products):
        if p_.kind == "P3" and t.ctx == "coast":
            shared[t.slot] += p_.size                    # one context tile shared by the coast's ships
        else:
            for k in p_.dets:
                own[off + k] += p_.size
    records = []
    print(f"\n[3] DETECTIONS -> P0-P3  (paper Table I, sat7.priority.classify; thresholds "
          f"{DET_THR} / {pcfg.p1_conf:.3f} [ASSUMPTION, swept in report 11])")
    print(f"    {'tile':<5}{'ctx':<7}{'cx':>6}{'cy':>6}{'w':>5}{'h':>5}{'conf':>7}{'len px':>8}"
          f"  {'level':<12}{'own bytes':>10}   why")
    for i, s in enumerate(wl.ships):
        t = tiles[tile_of[i]]
        cx, cy, w, h, conf = flat[i]
        lv = level_of(s, t.ctx, pcfg, lod)
        why = ("cloud tile: dropped by the pre-filter" if t.ctx == "cloud" else
               f"below onboard cut {DET_THR}" if lv == P0 else
               "on a coast -> +ROI +shared context tile" if lv == 3 else
               f"uncertain (< {pcfg.p1_conf:.3f}) -> +ROI crop" if lv == 2 else
               f"confident (>= {pcfg.p1_conf:.3f}) -> record only")
        records.append({"tile": t.slot + 1, "stands_in_for": t.ref_id, "ctx": t.ctx,
                        "box": [round(cx, 1), round(cy, 1), round(w, 1), round(h, 1)],
                        "conf": round(conf, 4), "length_px": round(s.size_px, 1),
                        "level": lv, "level_name": LEVEL_NAME[lv], "own_bytes": round(own[i])})
        print(f"    {t.slot + 1:<5}{t.ctx:<7}{cx:6.0f}{cy:6.0f}{w:5.0f}{h:5.0f}{conf:7.3f}{s.size_px:8.1f}"
              f"  {LEVEL_NAME[lv]:<12}{own[i]:10,.0f}   {why}")
    for k, b in shared.items():
        if b:
            print(f"    tile {k + 1}: + one coastal context tile {b:,.0f} B shared by its P3 detections")
    hist = {name: sum(r["level_name"] == name for r in records) for name in LEVEL_NAME.values()}
    print("    levels: " + "  ".join(f"{k} {v}" for k, v in hist.items()) + f"   [{live_lbl}]")

    # ------------------------------------------------------------------ [4] packet vs raw
    stream = b"".join(q for _t, p_ in products for q in p_.packets)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "downlink.bin").write_bytes(stream)
    ground = decode_downlink(stream)                    # the ground segment reads it back
    sem = len(stream)
    modeled = sum(it.size for it in encode_priority(wl, lod, pcfg))
    n_raw_tiles = sum(t.ctx != "cloud" for t in tiles)
    raw = n_raw_tiles * RAW_TILE_BYTES
    lod_bytes = sum(it.size for it in encode_lod(wl, lod))
    by = {k: [0, 0, 0, 0] for k in ("P1", "P2", "P3")}     # products, packets, bytes, of which headers+CRC
    jpeg = jpeg_hdr = 0
    for _t, p_ in products:
        b = by[p_.kind]
        b[0], b[1], b[2], b[3] = b[0] + 1, b[1] + len(p_.packets), b[2] + p_.size, b[3] + p_.header_bytes
        jpeg, jpeg_hdr = jpeg + p_.jpeg_bytes, jpeg_hdr + p_.jpeg_header
    print(f"\n[4] SEMANTIC PACKET  real bytes (sat7.semantic): binary records + JPEG crops of these pixels "
          f"+ CCSDS space packets")
    print(f"    {'':<22}{'products':>9}{'packets':>9}{'bytes':>11}{'of which hdr+CRC':>18}")
    for k, label in (("P1", "P1 records"), ("P2", "P2 ROI crops"), ("P3", "P3 context")):
        b = by[k]
        print(f"    {label:<22}{b[0]:9d}{b[1]:9d}{b[2]:11,}{b[3]:18,}")
    print(f"    SEMANTIC TOTAL (downlink.bin)          {sem:11,} B   [{live_lbl}, measured]")
    print(f"      record payloads {sum(len(pack_record(r)) for _t, e in encs for r in e.records):,} B | JPEG "
          f"{jpeg:,} B (of which JPEG headers {jpeg_hdr:,} B) | image headers + packet headers + CRC the rest")
    print(f"      ground decode: {len(ground['records'])} records, {len(ground['rois'])} ROIs, "
          f"{len(ground['contexts'])} contexts, every CRC checked")
    print(f"      on air with CCSDS TM framing + RS(255,223): {TMFraming().on_air_bytes(sem):,} B "
          f"(reported only: the link capacity already carries this overhead)")
    print(f"    same packet, MODELED sizes (wp23: 40 B record, WP4 power law; coastal tile at the "
          f"measured mean {COAST_TILE_BYTES_MEASURED / 1e3:.1f} kB): {modeled:,.0f} B")
    print(f"    RAW (B0 definition)                    {raw:11,} B   = {n_raw_tiles} non-cloud tiles x "
          f"{RAW_TILE_BYTES:,} B (768x768x3, uncompressed)")
    red = 1 - sem / raw if raw else float("nan")
    factor = f"{raw / sem:,.0f}x" if sem else "no bytes sent"
    print(f"    REDUCTION  1 - D_tx/D_raw =            {100 * red:11.3f} %   ({factor})   "
          f"[{live_lbl}; 9 ship-rich tiles -- NOT the headline]")
    print(f"    same detections, LoD ladder (the scheme behind the headline; crops and thumbnails "
          f"modeled): {lod_bytes:,.0f} B")

    # ------------------------------------------------------------------ [5] scheduler order
    order = ValueGreedy().order(items, 0.0)
    print(f"\n[5] SCHEDULER  value-greedy downlink order over this packet (sat7.scheduler.ValueGreedy):"
          f" highest value/byte first")
    print(f"    {'order':<9}{'kind':<6}{'items':>6}{'bytes':>10}   value per kB")
    runs, n = [], 0
    for it in order:                                   # consecutive items of one kind -> one row
        if runs and runs[-1][0] == it.kind:
            runs[-1][1].append(it)
        else:
            runs.append((it.kind, [it]))
    for kind, its in runs:
        d = [1e3 * it.value / it.size for it in its]
        span = f"{n + 1}" if len(its) == 1 else f"{n + 1}-{n + len(its)}"
        dens = f"{d[0]:.2f}" if max(d) - min(d) < 1e-9 else f"{max(d):.2f} .. {min(d):.2f}"
        print(f"    {span:<9}{kind:<6}{len(its):6d}{sum(it.size for it in its):10,.0f}   {dens}")
        n += len(its)
    print("    -> every metadata report goes first; ROI crops next; whole coastal context last. A short")
    print("       pass truncates from the bottom (progressive items cut, value ~ sqrt(fraction sent)).")

    # ------------------------------------------------------------------ [6] REAL reference
    ref = committed_predictions({t.ref_id for t in tiles})
    ref_ctx = [context_of(t.ref_context, len(ref[t.ref_id])) for t in tiles]
    rwl, rtile = build_workload(ref_ctx, [ref[t.ref_id] for t in tiles])
    ritems = encode_priority(rwl, lod, pcfg)
    rsem = sum(it.size for it in ritems)
    rraw = sum(c != "cloud" for c in ref_ctx) * RAW_TILE_BYTES
    rhist = {name: 0 for name in LEVEL_NAME.values()}
    for i, s in enumerate(rwl.ships):
        rhist[LEVEL_NAME[level_of(s, ref_ctx[rtile[i]], pcfg, lod)]] += 1
    print("\n[6] REFERENCE  the same 9 tile IDs, REAL detections committed in code/results/wp1_predictions.csv")
    print("    (B1 whole-tile, measured on the real tiles), through the identical P0-P3 rules; sizes MODELED")
    print("    here -- the bundle carries no Airbus pixels to encode (run with --data for real packets)")
    print(f"    {'#':<3}{'tile':<16}{'context':<9}{'boxes':>6}{'>=0.25':>8}")
    for k, t in enumerate(tiles):
        boxes = ref[t.ref_id]
        print(f"    {t.slot + 1:<3}{t.ref_id:<16}{ref_ctx[k]:<9}{len(boxes):6d}"
              f"{sum(b[4] >= DET_THR for b in boxes):8d}")
    print("    levels: " + "  ".join(f"{k} {v}" for k, v in rhist.items()) + "   [SIM-over-REAL]")
    print(f"    semantic {rsem:,.0f} B vs raw {rraw:,.0f} B -> reduction {100 * (1 - rsem / rraw):.3f} % "
          f"({rraw / rsem:,.0f}x)   [SIM-over-REAL, 9-tile sample]" if rsem and rraw else
          f"    semantic {rsem:,.0f} B vs raw {rraw:,.0f} B")
    check = None
    if not synthetic:
        check = {t.ref_id: cross_check(t.b1, ref[t.ref_id], {"det_thr": DET_THR, "conf_high": lod.conf_high})
                 for t in tiles}
        flips = {k: sum(c["flips"][k] for c in check.values()) for k in ("det_thr", "conf_high")}
        paired = sum(c["paired"] for c in check.values())
        print(f"    REPRODUCIBILITY live ONNX-CPU B1 vs committed (PyTorch-GPU): "
              f"{sum(c['live'] for c in check.values())} vs {sum(c['committed'] for c in check.values())} "
              f"boxes, {paired} paired, max |dconf| {max(c['max_dconf'] for c in check.values()):.1e}")
        print(f"    decision flips @{DET_THR}: {flips['det_thr']}   @{lod.conf_high:.3f}: {flips['conf_high']}"
              f"   {'OK' if not any(flips.values()) else '(!) CHECK'}")
    else:
        print("    (run with --data <Airbus split> to execute on these real tiles and cross-check the")
        print("     live boxes against these committed ones -- the reproducibility check)")

    # ------------------------------------------------------------------ [7] canon
    table = {r["strategy"]: r for r in _csv("wp6_real_table.csv")}
    ours = table.get("Ours: LoD + value-greedy", {})
    camp = _json("wp18_campaign.json").get("configs", {})
    w23 = _json("wp23_semantic_compare.json").get("semantic_schemes", {})
    w1 = _json("wp1_metrics.json").get("test", {})
    w1x = _json("wp1_export.json")
    print("\n[7] CANON  the measured results, read from code/results (never from this sample)")
    if w1:
        print(f"    detector  mAP50 {float(w1['mAP50']):.3f}, P {float(w1['precision']):.3f}, "
              f"R {float(w1['recall']):.3f} on 5,320 test tiles   [REAL, wp1_metrics.json]")
    if w1x:
        acc = w1x.get("accuracy", {})
        same = acc.get("onnx_fp32", {}).get("recall") == acc.get("pytorch_fp32", {}).get("recall")
        print(f"    ONNX FP32 (this model) recall {'== PyTorch' if same else '!= PyTorch'}; "
              f"{w1x['latency_ms_per_tile']['onnx_fp32_cpu']} ms/tile CPU   [REAL, laptop, wp1_export.json]")
    if "B2" in camp:
        b2 = camp["B2"]
        b1 = camp.get("B1", {})
        print(f"    B1 one call per scene: recall {float(b1['recall']['overall']):.3f} -> B2 SAHI+fusion "
              f"{float(b2['recall']['overall']):.3f} at {float(b2['compute_vs_regular_tiling_x']):.2f}x "
              f"compute, same {b2.get('scenes', '?')} scenes   [{b2.get('label', 'REAL')}, wp26_b0_b4.json]")
    bent = table.get("Bent pipe (raw, FIFO)", {})
    if ours and bent:
        print(f"    B3 LoD + value-greedy: {float(bent['MB_offered']):,.0f} MB raw -> "
              f"{float(ours['MB_sent']):.1f} MB/day = {float(ours['data_reduction_x']):,.0f}x, ship recall "
              f"{float(ours['ship_recall']):.3f} @ assumed 15% cloud (band 0.40-0.82)   "
              f"[SIM-over-REAL; coastal tiles measured, thumbnails/crops modeled; wp6_real_table.csv]")
    if "priority" in w23 and "lod" in w23:
        p, q = w23["priority"], w23["lod"]
        print(f"    P0-P3 scheme, same day: {float(p['MB_sent']):.1f} MB vs LoD {float(q['MB_sent']):.1f} MB, ship "
              f"recall {float(p['ship_recall']):.3f} vs {float(q['ship_recall']):.3f}: leaner, "
              f"{100 * (float(p['ship_recall']) - float(q['ship_recall'])):+.1f} pts   "
              f"[SIM-over-REAL, wp23_semantic_compare.json]")

    # ------------------------------------------------------------------ outputs
    draw_swath(swath, tiles, windows, records, synthetic, args.out / "swath_annotated.jpg")
    packet = {"label": live_lbl, "synthetic_tiles": synthetic, "source": manifest["label"] if synthetic else str(args.data),
              "settings": {"det_thr": DET_THR, "p1_conf": pcfg.p1_conf, "sahi_window": args.window,
                           "sahi_overlap": args.overlap, "fusion_nms_iou": pcfg_sahi.nms_iou},
              "raw_bytes": raw, "semantic_bytes_measured": sem, "semantic_bytes_modeled": modeled,
              "reduction": red, "packets": sum(len(p_.packets) for _t, p_ in products),
              "detections": records,
              "items_in_downlink_order": [{"kind": it.kind, "tile": tile_of[it.ships[0]] + 1,
                                           "bytes": round(it.size), "value": it.value} for it in order],
              "reference_real": {"levels": rhist, "semantic_bytes": rsem, "raw_bytes": rraw},
              "reproducibility": check}
    (args.out / "packet.json").write_text(json.dumps(packet, indent=2))
    total = time.perf_counter() - t_start
    print("\n" + RULE)
    print(f"  done in {total:.1f} s.  wrote {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}"
          f"/swath_annotated.jpg, downlink.bin and packet.json")
    print("  full story: python demo/summary.py  |  runbook: demo/DEMO.md  |  this demo: demo/quickstart/README.md")
    print(RULE + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
