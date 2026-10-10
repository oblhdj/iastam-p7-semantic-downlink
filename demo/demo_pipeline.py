"""The live half of the dashboard: one image through B0-B4, on CPU, with no Streamlit in sight.

    bundle, artifacts = run_all(img_bgr, info, detector, Settings())

Nothing is reimplemented. Every stage is the sat7 component the committed results were produced
with, called through sat7.campaign exactly as scripts/wp26_b0_b4.py calls it:

    input        sat7.imagery.decode_image            validate, decode, raw = H x W x 3
    context      sat7.prefilter.run_prefilter         per 768 px tile: cloud / coast / sea
    perception   sat7.campaign.perceive               none | one whole-image call | SAHI + fusion
    payload      sat7.campaign.downlink               raw image | records | P0-P3 semantic packet
    ground       sat7.semantic.decode_downlink        CRC-checked decode of the real byte stream
    link         sat7.comms.simulate_comms            direct / relay, SGP4 windows            [SIM]

A "bundle" is a JSON-serialisable dict; "artifacts" are the binary files beside it (annotated
images, the downlink stream). demo/make_fallback_assets.py writes exactly these to
demo/fallback_assets/, and the dashboard renders a stored bundle and a live one the same way.

What is measured and what is not, for the numbers this module produces:
  bytes       MEASURED: the length of the serialized packets of THIS image.
  times       MEASURED on this machine (wall clock): not flight hardware.
  link        SIM: propagated orbits, geometric windows, link-budget rates.
  energy      ESTIMATE: an assumed power x a measured or simulated time. No power was measured.
  accuracy    only when a ground-truth label file is supplied; otherwise counts and confidences.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sat7.accounting import raw_image_bytes  # noqa: E402
from sat7.b2_sahi_fusion import plan_slices  # noqa: E402
from sat7.campaign import MODES as BMODES, ONBOARD_CTX, TILE, downlink, perceive  # noqa: E402
from sat7.comms import CommsConfig, Links, build_links, pass_rate_bps, simulate_comms  # noqa: E402
from sat7.datasets import parse_yolo_lines  # noqa: E402
from sat7.energy import DEFAULT_POWERS_W  # noqa: E402
from sat7.evaluation import describe_detections, evaluate_detections  # noqa: E402
from sat7.imagery import ImageInfo, decode_image  # noqa: E402
from sat7.perception import PerceptionConfig, load_detector, slice_count  # noqa: E402
from sat7.prefilter import run_prefilter  # noqa: E402
from sat7.priority import LEVEL_NAME  # noqa: E402
from sat7.scheduler import Item, ValueGreedy  # noqa: E402
from sat7.semantic import (MISSION_EPOCH, PKT_CRC, PKT_HDR, EncoderConfig, TMFraming,  # noqa: E402
                           _IMG_HDR, decode_downlink, raw_packet_bytes)

import demo_data  # noqa: E402

MAX_SIDE = 8192                 # demo guard: a larger frame is minutes of CPU, not a live demo
RAW_BUILD_LIMIT = 80_000_000    # above this B0's packets are sized by formula instead of built
LINK_HOURS = 36.0               # the contact-plan horizon every wp27 / wp26 run uses
LEVEL_BGR = {0: (150, 150, 150), 1: (80, 220, 80), 2: (40, 215, 255), 3: (40, 110, 255)}


@dataclass(frozen=True)
class Settings:
    conf: float = 0.25                  # the onboard operating cut every experiment uses
    window: int = 768                   # SAHI window (detector-native)
    overlap: float = 0.20
    context: str = "prefilter"          # prefilter | ships | coast  (manual override, labelled)
    capture_h: float = 2.0              # hours after the simulated mission epoch
    sweep: bool = True                  # latency vs capture time (24 extra link simulations)


# ====================================================================================== loading
def load_model(path: Path | None = None):
    """The detector, warmed up so the first timed call is not the session start-up."""
    path = Path(path) if path else demo_data.find_model()
    if path is None or not path.exists():
        raise FileNotFoundError(
            "detector weights not found. Expected demo/quickstart/model/best.onnx (tracked in git: "
            "`git checkout -- demo/quickstart/model/best.onnx`) or code/runs/ships/weights/best.onnx.")
    det = load_detector(path)
    det([np.zeros((TILE, TILE, 3), np.uint8)])
    return det, path


def model_fingerprint(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def read_image(data: bytes, name: str):
    """Validate + decode. Raises ValueError with a message meant for the screen."""
    img, info = decode_image(data, name)
    if max(info.width, info.height) > MAX_SIDE:
        raise ValueError(f"{info.name}: {info.width} x {info.height} px. This demo accepts up to "
                         f"{MAX_SIDE} px a side (a larger frame is minutes of CPU inference); "
                         f"crop it, or run code/scripts/detect_image.py on it offline.")
    return img, info


def bundled_swath() -> tuple[np.ndarray, ImageInfo, dict]:
    """The 3x3 swath of bundled synthetic tiles, stitched abutting as the quickstart does."""
    d = demo_data.QUICKSTART / "tiles"
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    tiles = sorted(manifest["tiles"], key=lambda e: e["slot"])
    imgs, nbytes = [], 0
    for e in tiles:
        p = d / e["file"]
        if not p.exists():
            raise FileNotFoundError(f"bundled tile missing: demo/quickstart/tiles/{e['file']} "
                                    f"(regenerate with demo/quickstart/make_synthetic_tiles.py)")
        raw = p.read_bytes()
        nbytes += len(raw)
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (TILE, TILE):
            raise ValueError(f"{p.name}: expected a {TILE} x {TILE} image")
        imgs.append(im)
    n = manifest["swath"]["cols"]
    swath = np.vstack([np.hstack(imgs[r * n:(r + 1) * n]) for r in range(manifest["swath"]["rows"])])
    h, w = swath.shape[:2]
    info = ImageInfo("synthetic_swath_3x3", "JPEG", w, h, 3, "uint8", nbytes, raw_image_bytes(h, w),
                     ["9 bundled tiles stitched abutting (wp15 convention); file size = the 9 JPEGs"])
    return swath, info, manifest


def parse_labels(text: str, info: ImageInfo) -> list[tuple]:
    """YOLO `class cx cy w h` (normalised) -> pixel boxes; ValueError if nothing parses."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hint = "expected YOLO format, one box per line: `0 cx cy w h`, all normalised to 0-1"
    try:
        gts = parse_yolo_lines(lines, info.width, info.height, classes={0})
    except ValueError as e:
        raise ValueError(f"the label file is not in YOLO format ({e}); {hint}") from None
    if lines and not gts:
        raise ValueError(f"the label file has lines but none is a class-0 box in YOLO format; {hint}")
    return gts


def try_labels(text: str, info: ImageInfo) -> tuple[list[tuple] | None, str | None]:
    """(boxes, None), or (None, why) for a label file that cannot be used. A bad label file is not a
    reason to reject a good image: the image is then shown without accuracy figures."""
    try:
        return parse_labels(text, info), None
    except ValueError as e:
        return None, str(e)


# ====================================================================================== grid
@dataclass
class GridTile:
    row: int
    col: int
    box: tuple[int, int, int, int]      # x0, y0, x1, y1 in image pixels
    prefilter: str                      # the pre-filter's own verdict
    context: str                        # policy context: cloud | coast | ships


class Grid:
    """The captured image as 768 px tiles. Stands where sat7.campaign.Scene stands in
    campaign.downlink (same four members), for an image of any size: a remainder narrower than a
    quarter tile is merged into its neighbour instead of becoming a sliver."""
    scene_id = 1

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.xs, self.ys = self._cuts(width), self._cuts(height)
        self.cols, self.rows = len(self.xs) - 1, len(self.ys) - 1
        self.per_side = max(self.cols, self.rows)
        self.tiles: list[GridTile] = []

    @staticmethod
    def _cuts(extent: int) -> list[int]:
        cuts = list(range(0, extent, TILE))
        if len(cuts) > 1 and extent - cuts[-1] < TILE // 4:
            cuts.pop()
        return cuts + [extent]

    def tile_box(self, k: int) -> tuple[int, int, int, int]:
        return self.tiles[k].box

    def tile_index(self, cx: float, cy: float) -> int:
        c = max(0, min(self.cols - 1, int(np.searchsorted(self.xs, cx, side="right")) - 1))
        r = max(0, min(self.rows - 1, int(np.searchsorted(self.ys, cy, side="right")) - 1))
        return r * self.cols + c


def build_grid(img: np.ndarray, override: str = "prefilter") -> Grid:
    g = Grid(img.shape[1], img.shape[0])
    for r in range(g.rows):
        for c in range(g.cols):
            box = (g.xs[c], g.ys[r], g.xs[c + 1], g.ys[r + 1])
            verdict = run_prefilter(np.ascontiguousarray(img[box[1]:box[3], box[0]:box[2]])).context
            if override not in demo_data.CONTEXTS:
                raise ValueError(f"context must be one of {list(demo_data.CONTEXTS)}, got {override!r}")
            ctx = ONBOARD_CTX.get(verdict, "ships") if override == "prefilter" else override
            g.tiles.append(GridTile(r, c, box, verdict, ctx))
    return g


# ====================================================================================== drawing
def _text(img, s, org, scale, colour, thick):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _jpg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("could not encode the visualisation")
    return buf.tobytes()


def draw_detections(img, dets, conf, windows=None, banner: str = "") -> bytes:
    """Boxes at or above the cut in green with their confidence, below it in grey."""
    out = img.copy()
    k = max(1.0, max(img.shape[:2]) / 900)
    for (x, y, w, h) in windows or []:
        cv2.rectangle(out, (x + 2, y + 2), (x + w - 3, y + h - 3), (200, 200, 120), max(1, int(k)))
    for (cx, cy, w, h, c) in sorted(dets, key=lambda d: d[4]):
        p0, p1 = (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2))
        if c >= conf:
            cv2.rectangle(out, p0, p1, (60, 220, 60), max(2, int(2 * k)))
            _text(out, f"{c:.2f}", (p0[0], max(int(14 * k), p0[1] - int(4 * k))), 0.5 * k, (60, 220, 60),
                  max(1, int(k)))
        else:
            cv2.rectangle(out, p0, p1, (150, 150, 150), max(1, int(k)))
    if banner:
        _text(out, banner, (int(10 * k), img.shape[0] - int(12 * k)), 0.6 * k, (255, 255, 255), max(1, int(2 * k)))
    return _jpg(out)


def draw_semantic(img, grid: Grid, dets, levels, banner: str = "") -> bytes:
    """The semantic view: tile grid with each tile's context, boxes coloured by P-level."""
    out = img.copy()
    k = max(1.0, max(img.shape[:2]) / 900)
    for t in grid.tiles:
        x0, y0, x1, y1 = t.box
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), max(1, int(k)))
        _text(out, t.context, (x0 + int(8 * k), y0 + int(22 * k)), 0.6 * k, (255, 255, 255), max(1, int(2 * k)))
    for (cx, cy, w, h, c), lv in sorted(zip(dets, levels), key=lambda z: z[1]):
        p0, p1 = (int(cx - w / 2) - 2, int(cy - h / 2) - 2), (int(cx + w / 2) + 2, int(cy + h / 2) + 2)
        cv2.rectangle(out, p0, p1, LEVEL_BGR[lv], max(2, int(2 * k)) if lv else max(1, int(k)))
        if lv:
            _text(out, f"P{lv} {c:.2f}", (p0[0], max(int(14 * k), p0[1] - int(4 * k))), 0.5 * k,
                  LEVEL_BGR[lv], max(1, int(k)))
    if banner:
        _text(out, banner, (int(10 * k), img.shape[0] - int(12 * k)), 0.6 * k, (255, 255, 255), max(1, int(2 * k)))
    return _jpg(out)


# ====================================================================================== payload
def _byte_breakdown(products) -> dict:
    """Total packet bytes split into what carries information and what frames it."""
    total = sum(p.size for p in products)
    pkt = sum(p.header_bytes for p in products)
    images = [p for p in products if p.kind in ("P2", "P3", "RAW")]
    img_hdr = _IMG_HDR.size * len(images)
    jpeg_hdr = sum(p.jpeg_header for p in products)
    records = sum(len(p.payload) for p in products if p.kind == "P1")
    jpeg_data = sum(p.jpeg_bytes - p.jpeg_header for p in products)
    raw_px = sum(len(p.payload) - _IMG_HDR.size for p in products if p.kind == "RAW")
    payload = records + jpeg_data + raw_px
    overhead = pkt + img_hdr + jpeg_hdr
    assert payload + overhead == total, (payload, overhead, total)
    by_kind = {}
    for p in products:
        b = by_kind.setdefault(p.kind, {"products": 0, "packets": 0, "bytes": 0})
        b["products"] += 1
        b["packets"] += len(p.packets)
        b["bytes"] += p.size
    return {"total": total, "payload": payload, "overhead": overhead,
            "payload_parts": {"semantic records": records, "JPEG image data": jpeg_data,
                              "raw pixels": raw_px},
            "overhead_parts": {"CCSDS packet headers + CRC (8 B per packet)": pkt,
                               "image headers (18 B per image)": img_hdr,
                               "JPEG headers (tables, frame, scan)": jpeg_hdr},
            "by_kind": by_kind, "packets": sum(len(p.packets) for p in products),
            "on_air_with_TM_framing": TMFraming().on_air_bytes(total)}


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def run_mode(name: str, img, grid: Grid, dets: list, perception: dict, cfg: EncoderConfig,
             t_prefilter: float, gts: list | None, capture_s: float) -> tuple[dict, list[Item], bytes]:
    """One of B0-B4 on this image -> (bundle block, scheduler items, downlink byte stream)."""
    mode = BMODES[name]
    raw_bytes = raw_image_bytes(*img.shape[:2])
    built = True
    t0 = time.perf_counter()
    if mode.payload == "raw_image" and raw_bytes > RAW_BUILD_LIMIT:
        built = False                                   # exact size without holding the packets
        total = raw_packet_bytes(*img.shape[:2])
        n_pkt = math.ceil((_IMG_HDR.size + raw_bytes) / (cfg.max_packet_data - PKT_CRC))
        pkt = n_pkt * (PKT_HDR + PKT_CRC)
        bytes_ = {"total": total, "payload": raw_bytes, "overhead": pkt + _IMG_HDR.size,
                  "payload_parts": {"semantic records": 0, "JPEG image data": 0, "raw pixels": raw_bytes},
                  "overhead_parts": {"CCSDS packet headers + CRC (8 B per packet)": pkt,
                                     "image headers (18 B per image)": _IMG_HDR.size,
                                     "JPEG headers (tables, frame, scan)": 0},
                  "by_kind": {"RAW": {"products": 1, "packets": n_pkt, "bytes": total}}, "packets": n_pkt,
                  "on_air_with_TM_framing": TMFraming().on_air_bytes(total)}
        products, det_index, levels = [], [()], None
        items = [Item(1, capture_s, float(total), 1.0, "raw")]
        stream = b""
    else:
        dl = downlink(grid, img, dets, mode, cfg, t_capture_s=capture_s)
        products, det_index, levels = dl.products, dl.det_index, dl.levels
        bytes_ = _byte_breakdown(products)
        items = [Item(i + 1, capture_s, float(p.size), p.value, "raw" if p.kind == "RAW" else p.kind,
                      tuple(det_index[i]), progressive=p.progressive,
                      min_fraction=p.min_fraction if p.progressive else 1.0)
                 for i, p in enumerate(products)]
        stream = b"".join(q for p in products for q in p.packets)
    t_enc = time.perf_counter() - t0

    # ---- the ground reads the stream back (B0: only when small; reassembling a large raw image
    #      from thousands of segments is a ground-side cost this demo does not time)
    records, rois, contexts, t_dec = [], [], [], None
    if mode.payload != "raw_image" or (built and raw_bytes <= 2_000_000):
        t0 = time.perf_counter()
        ground = decode_downlink(stream)
        t_dec = time.perf_counter() - t0
        for r in ground["records"]:
            records.append({"image_id": r.image_id, "class": "ship", "confidence": round(r.confidence, 4),
                            "cx": r.cx, "cy": r.cy, "w": r.w, "h": r.h, "level": f"P{r.level}",
                            "coastal": r.coastal, "roi_id": r.roi_id, "context_id": r.ctx_id})
        for store, src in ((rois, ground["rois"]), (contexts, ground["contexts"])):
            for pid, v in src.items():
                store.append({"id": pid, "x0": v["x0"], "y0": v["y0"], "w": v["w"], "h": v["h"],
                              "quality": v["quality"], "jpeg_bytes": len(v["jpeg"]),
                              "jpeg_b64": _b64(v["jpeg"])})

    levels = list(levels) if levels is not None else None
    tx = [d for d, lv in zip(dets, levels) if lv > 0] if levels is not None else []
    hist = None
    if levels is not None and mode.payload == "semantic":
        hist = {LEVEL_NAME[k]: sum(lv == k for lv in levels) for k in sorted(LEVEL_NAME)}
    quality = None
    if gts is not None and mode.perception != "none":
        quality = evaluate_detections([(tx, gts)], conf_thr=cfg.det_thr)
    t_proc = {"none": 0.0, "whole": perception["whole"]["seconds"],
              "sahi": perception["sahi"]["seconds"]}[mode.perception]
    stages = {"detector": t_proc}
    if mode.payload == "semantic":                      # the policy needs each tile's context
        stages["pre-filter (tile context)"] = t_prefilter
        t_proc += t_prefilter
    calls = {"none": 0, "whole": perception["whole"]["calls"], "sahi": perception["sahi"]["calls"]}[mode.perception]
    n_onboard = sum(d[4] >= cfg.det_thr for d in dets)
    block = {
        "mode": name, "paper": mode.paper, "perception": mode.perception, "payload": mode.payload,
        "relay": mode.relay, "detector_calls": calls,
        "n_detections_onboard": int(n_onboard), "n_transmitted": len(tx),
        "transmitted_share_of_onboard": (len(tx) / n_onboard) if n_onboard else None,
        "level_histogram": hist, "levels": levels,
        "bytes": bytes_, "n_items": len(items),
        "items": [{"id": it.id, "kind": it.kind, "bytes": int(it.size), "value": it.value,
                   "progressive": it.progressive, "min_fraction": round(it.min_fraction, 4)} for it in items],
        "records": records, "rois": rois, "contexts": contexts,
        "ground_decode": None if t_dec is None else {"records": len(records), "rois": len(rois),
                                                    "contexts": len(contexts), "crc": "every packet checked"},
        "timing_s": {"processing": t_proc, "processing_stages": stages, "encoding": t_enc,
                     "decoding": t_dec,
                     "decoding_note": None if t_dec is not None else
                     "not timed: reassembling a multi-megabyte raw image is a ground-side cost outside this demo",
                     "encoding_note": None if built else "sized by formula (image too large to build B0 packets live)"},
        "quality": quality,
    }
    return block, items, stream


# ====================================================================================== link
@lru_cache(maxsize=2)
def contact_plan(hours: float = LINK_HOURS) -> Links:
    """The propagated contact plan [SIM]: same config, epoch and horizon as wp26 / wp27."""
    return build_links(CommsConfig(), MISSION_EPOCH, hours)


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%d %b %H:%M")


def describe_links(links: Links, cfg: CommsConfig) -> dict:
    def passes(ps, share):
        return [{"rise_utc": _iso(p.rise), "rise_h": round((p.rise - MISSION_EPOCH).total_seconds() / 3600, 2),
                 "duration_s": round(p.duration_s), "capacity_MB": round(p.capacity_bytes / 1e6, 2),
                 "payload_rate_Mbps": round(pass_rate_bps(p) / 1e6, 3),
                 "physical_rate_Mbps": round(pass_rate_bps(p, share) / 1e6, 3),
                 "max_elevation_deg": round(p.max_elevation_deg, 1)} for p in ps]
    return {"label": "SIM", "epoch_utc": MISSION_EPOCH.isoformat(), "horizon_h": LINK_HOURS,
            "primary_passes": passes(links.primary_passes, links.primary_share),
            "relay_passes": passes(links.relay_passes, links.relay_share),
            "isl_windows": [{"start_h": round((w.start - MISSION_EPOCH).total_seconds() / 3600, 2),
                             "duration_s": round(w.duration_s)} for w in links.isl],
            "assumptions": {
                "ground link model": f"{cfg.link.name}: {cfg.link.bandwidth_hz / 1e6:g} MHz, SNR "
                                     f"{cfg.link.snr_zenith_db:g} dB at zenith, modem cap "
                                     f"{cfg.link.max_rate_bps / 1e6:g} Mbps, coding efficiency {cfg.link.efficiency:g}",
                "link share of each ground pass": cfg.link_share,
                "relay ground-pass share": cfg.relay_share,
                "ground transmit power (W)": cfg.p_tx_W,
                "inter-satellite rate (Mbps, after coding)": cfg.isl_rate_bps * cfg.isl_efficiency / 1e6,
                "inter-satellite transmit power (W)": cfg.p_isl_W,
                "relay orbital plane (RAAN, deg)": cfg.relay_raan_deg,
                "routing objective": f"J = lam_E x E + lam_T x T, lam_T = 1/{1 / cfg.lam_T:.0f} per s, lam_E = {cfg.lam_E:g}",
                "ground station": "Sfax (ENIS), minimum elevation 5 deg",
                "orbit": "500 km sun-synchronous, SGP4-propagated",
            },
            "assumption_label": "ASSUMPTION: every rate and power (reports 17, 19); windows are geometry"}


def simulate_route(items: list[Item], route: str, links: Links | None = None,
                   cfg: CommsConfig | None = None) -> dict:
    """One set of items down one route [SIM].

    direct  the relay is disabled: sat7.scheduler behaviour exactly.
    policy  each item takes the path with the smaller J (sat7.comms; the route is policy-chosen).
    relay   manual: the primary's own ground passes are withheld, so an item can only go via the
            relay. Raw imagery is not an eligible relay cargo and stays undelivered.
    """
    if route not in demo_data.ROUTES:
        raise ValueError(f"route must be one of {list(demo_data.ROUTES)}, got {route!r}")
    cfg = cfg or CommsConfig()
    links = links or contact_plan()
    if route == "direct":
        cfg = replace(cfg, relay_enabled=False)
    elif route == "relay":
        links = replace(links, primary_passes=[])
    res = simulate_comms(list(items), links, MISSION_EPOCH, ValueGreedy(), cfg)
    lat = [d.latency_s for d in res.deliveries]
    whole = {}
    for d in res.deliveries:
        whole[d.item.id] = whole.get(d.item.id, 0.0) + d.fraction
    delivered = sum(1 for v in whole.values() if v >= 1.0 - 1e-9)
    by = {"ground": 0.0, "isl": 0.0, "relay_ground": 0.0}
    en = dict(by)
    for d in res.deliveries:
        for k, v in d.tx_s.items():
            by[k] += v
        for k, v in d.energy_J.items():
            en[k] += v
    n_relay = sum(d.path == "relay" for d in res.deliveries)
    return {
        "label": "SIM", "route": route,
        "chosen_by": {"direct": "fixed: direct only", "policy": "policy (min J per item)",
                      "relay": "manual: relay forced"}[route],
        "items": len(items), "delivered_items": delivered, "undelivered_items": len(items) - delivered,
        "relay_items": n_relay, "direct_items": len(res.deliveries) - n_relay,
        "bytes_offered": int(sum(i.size for i in items)),
        "bytes_delivered": int(round(sum(d.item.size * d.fraction for d in res.deliveries))),
        "latency_first_s": min(lat) if lat else None, "latency_last_s": max(lat) if lat else None,
        "latency_median_s": float(np.median(lat)) if lat else None,
        "tx_s": by, "tx_s_total": sum(by.values()), "energy_J": en, "energy_J_total": sum(en.values()),
        "link_use": res.link_use,
        "deliveries": [{"item": d.item.id, "kind": d.item.kind, "bytes": int(d.item.size),
                        "fraction": round(d.fraction, 4), "path": d.path,
                        "delivered_h": round(d.delivered_s / 3600, 3),
                        "latency_h": round(d.latency_s / 3600, 3),
                        "tx_s": round(sum(d.tx_s.values()), 5),
                        "energy_J": round(sum(d.energy_J.values()), 5)} for d in res.deliveries[:400]],
    }


def simulate_all(items_by_mode: dict[str, list[Item]], capture_h: float, sweep: bool = True) -> dict:
    """Every mode down every route it has, plus (optionally) B3/B4's latency against capture time."""
    cfg, links = CommsConfig(), contact_plan()
    stamp = lambda items, h: [replace(it, created_s=h * 3600.0) for it in items]
    runs = {}
    for mode, items in items_by_mode.items():
        its = stamp(items, capture_h)
        runs[mode] = {"direct": simulate_route(its, "direct", links, cfg)}
        if demo_data.MODE_INFO[mode]["relay"]:
            runs[mode]["policy"] = simulate_route(its, "policy", links, cfg)
            runs[mode]["relay"] = simulate_route(its, "relay", links, cfg)
    out = {"label": "SIM", "capture_h": capture_h,
           "capture_utc": _iso(MISSION_EPOCH + timedelta(hours=capture_h)),
           "plan": describe_links(links, cfg), "runs": runs, "sweep": None}
    if sweep and items_by_mode.get("B4"):
        rows = []
        for h in range(24):
            its = stamp(items_by_mode["B4"], float(h))
            d, p = simulate_route(its, "direct", links, cfg), simulate_route(its, "policy", links, cfg)
            ok = lambda r: r["delivered_items"] == r["items"]
            rows.append({"capture_h": h,
                         "direct_h": d["latency_last_s"] / 3600 if ok(d) else None,
                         "policy_h": p["latency_last_s"] / 3600 if ok(p) else None,
                         "policy_relay_items": p["relay_items"], "items": p["items"]})
        out["sweep"] = {"what": "hours from capture until the LAST item of the semantic packet is on "
                                "the ground, for an image captured at each hour of the simulated day",
                        "rows": rows}
    return out


# ====================================================================================== everything
def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_all(img: np.ndarray, info: ImageInfo, detector, settings: Settings | None = None,
            gts: list | None = None, source: dict | None = None,
            model_path: Path | None = None) -> tuple[dict, dict[str, bytes]]:
    """One image through B0-B4 and the simulated links -> (bundle, artifacts)."""
    s = settings or Settings()
    source = source or {"kind": "upload", "label": "UPLOAD", "synthetic": False,
                        "note": "an image supplied at the demo; not part of any measured result"}
    t_all = time.perf_counter()
    capture_s = s.capture_h * 3600.0

    t0 = time.perf_counter()
    grid = build_grid(img, s.context)
    t_prefilter = time.perf_counter() - t0

    # ---- perception: the two configurations, each timed (wall clock, this machine)
    perception, dets = {}, {}
    for key, mode in (("whole", BMODES["B1"]), ("sahi", BMODES["B2"])):
        t0 = time.perf_counter()
        dets[key] = perceive(img, mode, detector, window=s.window, overlap=s.overlap)
        sec = time.perf_counter() - t0
        pc = PerceptionConfig(mode=key, window=s.window, overlap=s.overlap)
        kept = sorted((d for d in dets[key] if d[4] >= s.conf), key=lambda d: -d[4])
        block = {"seconds": sec, "calls": slice_count(info.width, info.height, pc),
                 "n_raw": len(dets[key]), "n_at_or_above_cut": len(kept),
                 "detections": [{"cx": round(d[0], 1), "cy": round(d[1], 1), "w": round(d[2], 1),
                                 "h": round(d[3], 1), "confidence": round(d[4], 4),
                                 "class": "ship", "at_or_above_cut": bool(d[4] >= s.conf)}
                                for d in sorted(dets[key], key=lambda d: -d[4])],
                 "describe": describe_detections(dets[key], s.conf),
                 "quality": evaluate_detections([(dets[key], gts)], conf_thr=s.conf) if gts is not None else None}
        perception[key] = block
    one_window = max(info.width, info.height) <= s.window
    scale = min(1.0, getattr(detector, "size", TILE) / max(info.width, info.height))
    perception["note"] = (
        "The image fits one detector window, so plain YOLO and SAHI are the same single detector "
        "call. They can still differ by a box: SAHI's fusion step merges boxes overlapping above "
        "IoU 0.5 that the detector's own NMS (0.7) kept. SAHI only changes what the detector sees "
        "on an image larger than the window." if one_window else
        f"Plain YOLO letterboxes the whole {info.width} x {info.height} px image to the detector's "
        f"{getattr(detector, 'size', TILE)} px input (scale {scale:.2f}), so a small ship shrinks below "
        f"what the detector resolves; SAHI keeps native resolution at "
        f"{perception['sahi']['calls']} calls instead of 1.")
    perception["one_window"] = one_window
    windows = plan_slices(info.width, info.height, PerceptionConfig(
        mode="sahi", window=s.window, overlap=s.overlap).to_sahi())

    # ---- the five modes
    cfg = EncoderConfig(det_thr=s.conf)
    modes, items, streams = {}, {}, {}
    for name, mode in BMODES.items():
        d = [] if mode.perception == "none" else dets[mode.perception]
        modes[name], items[name], streams[name] = run_mode(name, img, grid, d, perception, cfg,
                                                          t_prefilter, gts, capture_s)

    # ---- the simulated links (the page still works without them)
    try:
        comms = simulate_all(items, s.capture_h, s.sweep)
    except Exception as e:                               # SGP4 / skyfield trouble must not sink the run
        comms = {"error": f"{type(e).__name__}: {e}", "runs": {}, "plan": None, "sweep": None,
                 "label": "SIM", "capture_h": s.capture_h}

    banner = "SYNTHETIC STAND-IN - NOT AIRBUS IMAGERY - NOT A MEASUREMENT" if source.get("synthetic") else ""
    artifacts = {
        "original.jpg": _jpg(img),
        "detections_plain_yolo.jpg": draw_detections(img, dets["whole"], s.conf, banner=banner),
        "detections_sahi.jpg": draw_detections(img, dets["sahi"], s.conf, windows, banner=banner),
        "semantic_levels.jpg": draw_semantic(img, grid, dets["sahi"], modes["B3"]["levels"], banner=banner),
        "downlink_B3.bin": streams["B3"],
    }
    cc = CommsConfig()
    bundle = {
        "bundle_version": demo_data.BUNDLE_VERSION,
        "meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "commit": _git_commit(), "python": platform.python_version(),
                 "machine": f"{platform.system()} {platform.machine()}",
                 "backend": f"onnxruntime {getattr(detector, 'ort_version', '?')} CPU",
                 "model": Path(model_path).name if model_path else "?",
                 "model_sha256_16": model_fingerprint(model_path) if model_path else None,
                 "elapsed_s": None},
        "source": source,
        "input": {"name": info.name, "format": info.format, "width": info.width, "height": info.height,
                  "channels_in": info.channels_in, "file_bytes": info.file_bytes,
                  "raw_bytes": info.raw_bytes, "notes": list(info.notes),
                  "ground_truth": None if gts is None else {"n": len(gts)}},
        "settings": asdict(s),
        "grid": {"rows": grid.rows, "cols": grid.cols, "t_prefilter_s": t_prefilter,
                 "context_source": demo_data.CONTEXTS[s.context],
                 "tiles": [{"row": t.row, "col": t.col, "box": list(t.box), "prefilter": t.prefilter,
                            "context": t.context} for t in grid.tiles]},
        "perception": perception,
        "modes": modes,
        "comms": comms,
        "assumptions": {"p_cpu_W": DEFAULT_POWERS_W["cpu"], "p_tx_W": cc.p_tx_W, "p_isl_W": cc.p_isl_W,
                        "label": "ASSUMPTION: no power was measured in this project (report 17)"},
        "labels": {"bytes": "MEASURED on this image: the length of the serialized packets",
                   "times": "MEASURED on this machine (wall clock); a laptop, not flight hardware",
                   "link": "SIM: propagated orbits, geometric windows, link-budget rates",
                   "energy": "ESTIMATE: assumed power x measured or simulated time",
                   "accuracy": "REAL against the supplied labels" if gts is not None else
                               "none: no ground truth, so counts and confidences only"},
    }
    bundle["meta"]["elapsed_s"] = round(time.perf_counter() - t_all, 2)
    return bundle, artifacts
