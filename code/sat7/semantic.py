"""Semantic records and the ACTUAL bytes that carry them to the ground (paper Table I, P0-P3).

Until now every downlink byte in the repo was a size MODEL: a P1 report "costs 40 B" (an
assumption), a ROI "costs SizeModel.l1(length)" (a power law fitted on WP4's JPEG crops) and a
coastal context "costs 32,420 B" (a median). Nothing was ever serialized (audit C2). This module
builds the bytes: a binary semantic record per detection, real JPEG ROI / context crops, CCSDS
space packets with their headers and checksums, and a ground-side decoder that reads them back.
Item sizes handed to the scheduler are then `len()` of real packets.

Layers and what each costs (counted ONCE each):
  1. record      a fixed-width binary semantic record (below). Image id + capture time live in the
                 record (or once per tile with metadata="tile"), never again in a packet header.
  2. image       ROI / context: a small header (id, crop geometry, quality, codec) + JPEG bytes.
                 The JPEG's own header (quantisation + Huffman tables, ~600 B) is counted -- it is
                 measured, not assumed. A ROI packet does NOT repeat the metadata: the record
                 references it by roi_id. A coastal context tile is encoded once per tile and
                 referenced by every record on that coast (no per-ship copies).
  3. packet      CCSDS 133.0-B-2 space packet: 6 B primary header + data + 2 B CRC-16-CCITT
                 (packet error control). Payloads above `max_packet_data` are segmented (first /
                 continuation / last), each segment paying its own 8 B.
  4. link        TM transfer frames, ASM and Reed-Solomon parity. NOT added to item sizes: the
                 orbit capacity already discounts link overhead (orbit.LinkConfig.efficiency = 0.8,
                 ASSUMPTION). `TMFraming` computes the explicit CCSDS figure (0.866 for 1115 B frames,
                 RS(255,223) x5) to show 0.8 is conservative -- adding it to the items as well would
                 count the same overhead twice.

Record layout (little-endian), 26 B base:
  u8  flags        bit0 geo, bit1 roi, bit2 context, bit3 dark, bit4 coastal, bits5-6 level P0-P3
  u8  class id
  u16 confidence   round(conf * 65535)                -> resolution 1.5e-5
  u32 image id
  u32 t_s, u16 t_ms capture time since MISSION_EPOCH
  u32 cx, u32 cy   global box centre, 0.1 px           -> scenes up to 429 M px
  u16 w,  u16 h    box size, 0.1 px                     -> objects up to 6,553 px
  [i32 lat, i32 lon]  1e-7 deg (+8 B)   only if a georeference is supplied
  [u32 roi_id]        (+4 B)            P2 and P3
  [u32 ctx_id]        (+4 B)            P3
With metadata="tile" a packet carries one 11 B tile header (image id, time, count) and 16 B records
without the per-record image id / time.

Assignment of levels is sat7.priority.classify (configurable PriorityConfig: the 0.25 onboard cut,
p1_conf 0.670, coastal escalation, dark bump); products mirror sat7.priority.encode_tile_priority
item for item, so the scheduler sees the same structure -- only the sizes are now real.

Labels: sizes produced here are REAL (measured bytes of real encodings) when the pixels are real;
thresholds, values and the JPEG qualities are the repo's ASSUMPTIONS (WP4 / report 23), swept.
"""
from __future__ import annotations

import binascii
import math
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

from .priority import P0, P2, P3, PriorityConfig, classify
from .scheduler import Item, LoDConfig, Ship

MISSION_EPOCH = datetime(2026, 9, 18, tzinfo=timezone.utc)    # the orbit simulation's start

FLAG_GEO, FLAG_ROI, FLAG_CTX, FLAG_DARK, FLAG_COAST = 1, 2, 4, 8, 16
_REC = struct.Struct("<BBHIIHIIHH")        # 26 B: flags class conf image t_s t_ms cx cy w h
_REC_TILE = struct.Struct("<BBHIIHH")      # 16 B: the same minus image id / time (tile mode)
_TILE_HDR = struct.Struct("<IIHB")         # 11 B: image id, t_s, t_ms, n records
_GEO = struct.Struct("<ii")
_REF = struct.Struct("<I")
_IMG_HDR = struct.Struct("<IIIHHBB")       # 18 B: id, x0, y0, w, h, quality, codec
CODEC_RAW, CODEC_JPEG, CODEC_JPEG_PROGRESSIVE = 0, 1, 2
APID = {"RAW": 0x100, "P1": 0x101, "P2": 0x102, "P3": 0x103, "P1T": 0x104}
PKT_HDR, PKT_CRC = 6, 2


# ================================================================================ record
@dataclass
class SemanticRecord:
    image_id: int
    t_capture_s: float                  # seconds since MISSION_EPOCH
    class_id: int
    confidence: float
    cx: float                           # global box, pixels, centre form (the repo convention)
    cy: float
    w: float
    h: float
    level: int
    coastal: bool = False
    dark: bool = False
    geo: tuple[float, float] | None = None     # (lat, lon) degrees, if georeferenced
    roi_id: int | None = None
    ctx_id: int | None = None

    def flags(self) -> int:
        f = (FLAG_GEO if self.geo else 0) | (FLAG_ROI if self.roi_id is not None else 0)
        f |= (FLAG_CTX if self.ctx_id is not None else 0) | (FLAG_DARK if self.dark else 0)
        f |= (FLAG_COAST if self.coastal else 0) | ((self.level & 3) << 5)
        return f


def _q(v: float, scale: float, hi: int) -> int:
    return int(min(hi, max(0, round(v * scale))))


def pack_record(r: SemanticRecord, with_image: bool = True) -> bytes:
    t_s = int(r.t_capture_s)
    t_ms = int(round((r.t_capture_s - t_s) * 1000)) % 1000
    common = (r.flags(), r.class_id, _q(r.confidence, 65535, 65535))
    box = (_q(r.cx, 10, 2**32 - 1), _q(r.cy, 10, 2**32 - 1), _q(r.w, 10, 65535), _q(r.h, 10, 65535))
    out = (_REC.pack(*common, r.image_id, t_s, t_ms, *box) if with_image
           else _REC_TILE.pack(*common, *box))
    if r.geo:
        out += _GEO.pack(int(round(r.geo[0] * 1e7)), int(round(r.geo[1] * 1e7)))
    if r.roi_id is not None:
        out += _REF.pack(r.roi_id)
    if r.ctx_id is not None:
        out += _REF.pack(r.ctx_id)
    return out


def unpack_record(buf: bytes, off: int = 0, image_id: int | None = None,
                  t_capture_s: float | None = None) -> tuple[SemanticRecord, int]:
    if image_id is None:
        flags, cls, conf, image_id, t_s, t_ms, cx, cy, w, h = _REC.unpack_from(buf, off)
        off += _REC.size
        t_capture_s = t_s + t_ms / 1000
    else:
        flags, cls, conf, cx, cy, w, h = _REC_TILE.unpack_from(buf, off)
        off += _REC_TILE.size
    geo = roi = ctx = None
    if flags & FLAG_GEO:
        la, lo = _GEO.unpack_from(buf, off)
        geo, off = (la / 1e7, lo / 1e7), off + _GEO.size
    if flags & FLAG_ROI:
        (roi,), off = _REF.unpack_from(buf, off), off + _REF.size
    if flags & FLAG_CTX:
        (ctx,), off = _REF.unpack_from(buf, off), off + _REF.size
    rec = SemanticRecord(image_id, t_capture_s, cls, conf / 65535, cx / 10, cy / 10, w / 10, h / 10,
                         (flags >> 5) & 3, bool(flags & FLAG_COAST), bool(flags & FLAG_DARK), geo, roi, ctx)
    return rec, off


# ================================================================================ images
def crop_box(cx, cy, bw, bh, margin: float, width: int, height: int):
    """WP4's crop rule (scripts/wp4_measure_lod.py): a square of side max(w, h) * (1 + margin)
    centred on the box, clipped to the image. Returns (x0, y0, x1, y1)."""
    half = max(bw, bh) * (1 + margin) / 2
    return (int(max(0, cx - half)), int(max(0, cy - half)),
            int(min(width, cx + half)), int(min(height, cy + half)))


def encode_jpeg(img: np.ndarray, quality: int, progressive: bool) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    if progressive:
        params += [cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
    ok, buf = cv2.imencode(".jpg", img, params)
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def _scan_ends(jpg: bytes) -> list[int]:
    """Byte offsets where each scan's entropy-coded data ends (start of the next marker)."""
    ends, i, n = [], 2, len(jpg)
    while i < n - 1:
        if jpg[i] != 0xFF:
            i += 1
            continue
        m = jpg[i + 1]
        if m == 0xD9:                                   # EOI
            break
        seg = struct.unpack(">H", jpg[i + 2:i + 4])[0]
        i += 2 + seg
        if m == 0xDA:                                   # SOS: skip the entropy-coded data
            while i < n - 1 and not (jpg[i] == 0xFF and jpg[i + 1] not in (0x00, *range(0xD0, 0xD8))):
                i += 1
            ends.append(i)
    return ends


def jpeg_header_bytes(jpg: bytes) -> int:
    """Bytes before the first scan's entropy-coded data: SOI + tables + frame + scan header."""
    i = jpg.find(b"\xff\xda")
    return i + 2 + struct.unpack(">H", jpg[i + 2:i + 4])[0] if i >= 0 else len(jpg)


def min_decodable_fraction(jpg: bytes, progressive: bool) -> float:
    """Smallest prefix the ground can still turn into a whole (coarse) image.

    Progressive JPEG: the end of the first scan (the DC pass, a 1/8-resolution image) -- what the
    scheduler may truncate down to. Baseline JPEG has no such prefix: a cut file is a partial
    image, so the whole item is needed (1.0)."""
    if not progressive:
        return 1.0
    ends = _scan_ends(jpg)
    return min(1.0, ends[0] / len(jpg)) if ends else 1.0   # the ground appends the EOI marker


# ================================================================================ packets
def crc16(data: bytes) -> int:
    """CRC-16-CCITT (poly 0x1021, init 0xFFFF): CCSDS packet error control."""
    return binascii.crc_hqx(data, 0xFFFF)


@dataclass
class Packetizer:
    """CCSDS 133.0-B-2 space packets: 6 B primary header + data field (payload + 2 B CRC)."""
    max_packet_data: int = 4096                   # data-field bytes per packet, CRC included
    _seq: dict = field(default_factory=dict)

    def packetize(self, apid: int, payload: bytes) -> list[bytes]:
        room = self.max_packet_data - PKT_CRC
        if room < 1:
            raise ValueError("max_packet_data must exceed the 2 B CRC")
        chunks = [payload[i:i + room] for i in range(0, max(1, len(payload)), room)] or [b""]
        out = []
        for k, chunk in enumerate(chunks):
            flags = 0b11 if len(chunks) == 1 else 0b01 if k == 0 else 0b10 if k == len(chunks) - 1 else 0b00
            seq = self._seq.get(apid, 0)
            self._seq[apid] = (seq + 1) & 0x3FFF
            hdr = struct.pack(">HHH", apid & 0x7FF, (flags << 14) | seq, len(chunk) + PKT_CRC - 1)
            out.append(hdr + chunk + struct.pack(">H", crc16(hdr + chunk)))
        return out


def depacketize(stream: bytes) -> list[tuple[int, bytes]]:
    """Split a packet stream, check every CRC, reassemble segments -> [(apid, payload)]."""
    out, pending, i = [], {}, 0
    while i < len(stream):
        w1, w2, ln = struct.unpack(">HHH", stream[i:i + PKT_HDR])
        end = i + PKT_HDR + ln + 1
        body, crc = stream[i + PKT_HDR:end - PKT_CRC], struct.unpack(">H", stream[end - PKT_CRC:end])[0]
        if crc16(stream[i:end - PKT_CRC]) != crc:
            raise ValueError(f"CRC error in packet at byte {i}")
        apid, flags = w1 & 0x7FF, w2 >> 14
        if flags == 0b11:
            out.append((apid, body))
        elif flags == 0b01:
            pending[apid] = body
        else:
            pending[apid] = pending.get(apid, b"") + body
            if flags == 0b10:
                out.append((apid, pending.pop(apid)))
        i = end
    return out


@dataclass
class TMFraming:
    """CCSDS TM link layer, for reporting on-air bytes (NOT added to item sizes -- see module doc).
    Defaults: 1115 B transfer frame (6 B header + 2 B FECF inside), 4 B ASM, RS(255,223) interleave 5."""
    frame_len: int = 1115
    tf_header: int = 6
    fecf: int = 2
    asm: int = 4
    rs_parity: int = 160

    @property
    def data_per_frame(self) -> int:
        return self.frame_len - self.tf_header - self.fecf

    @property
    def efficiency(self) -> float:
        return self.data_per_frame / (self.frame_len + self.asm + self.rs_parity)

    def on_air_bytes(self, packet_bytes: int) -> int:
        return math.ceil(packet_bytes / self.data_per_frame) * (self.frame_len + self.asm + self.rs_parity)


# ================================================================================ encoder
@dataclass
class EncoderConfig:
    """Everything that decides what is sent and how big it is. Defaults = the repo's (report 23,
    WP4): 0.25 onboard cut, P1/P2 at 0.670, tight ROI q40, wake context q80, coastal tile q40."""
    det_thr: float = 0.25                 # below: P0, never transmitted
    priority: PriorityConfig = field(default_factory=PriorityConfig)
    roi_margin: float = 0.15              # WP4 "tight" crop  (P2 ROI = the L1 chip)
    roi_quality: int = 40
    ctx_margin: float = 1.5               # WP4 "wake" crop   (P3 context away from a coast)
    ctx_quality: int = 80
    coast_tile_quality: int = 40          # whole tile        (P3 context on a coast, WP7 q40)
    progressive: bool = True              # truncatable images (the scheduler's progressive items)
    metadata: str = "record"              # "record": one packet per P1 record (finest scheduling)
                                          # "tile":   one packet per tile, image id / time once
    max_packet_data: int = 4096
    class_id: int = 0
    dark_weight: float = 5.0              # value of a dark vessel's report (LoDConfig.dark_weight)

    def __post_init__(self):
        if self.metadata not in ("record", "tile"):
            raise ValueError(f"metadata must be 'record' or 'tile', got {self.metadata!r}")


@dataclass
class Product:
    """One schedulable downlink unit (becomes one scheduler Item)."""
    kind: str                       # "P1" metadata | "P2" ROI | "P3" context
    image_id: int
    dets: tuple[int, ...]           # detection indices whose information this product carries
    payload: bytes                  # application payload, before packetization
    packets: list[bytes]
    value: float
    progressive: bool = False
    min_fraction: float = 1.0
    jpeg_bytes: int = 0
    jpeg_header: int = 0

    @property
    def size(self) -> int:
        return sum(len(p) for p in self.packets)

    @property
    def header_bytes(self) -> int:
        return len(self.packets) * (PKT_HDR + PKT_CRC)


@dataclass
class Encoded:
    records: list[SemanticRecord]
    products: list[Product]
    levels: list[int]               # per input detection, P0..P3

    @property
    def size(self) -> int:
        return sum(p.size for p in self.products)


class Ids:
    """Mission-unique ROI / context ids (u32)."""
    def __init__(self):
        self.n = 0

    def __call__(self) -> int:
        self.n = (self.n + 1) & 0xFFFFFFFF
        return self.n


def encode_image(img, image_id: int, t_capture_s: float, dets, context: str,
                 cfg: EncoderConfig | None = None, ids: Ids | None = None,
                 packetizer: Packetizer | None = None, dark=None, georef=None,
                 origin: tuple[int, int] = (0, 0), jpeg_fn=None,
                 context_box: tuple[int, int, int, int] | None = None) -> Encoded:
    """Semantic records + real ROI / context crops + space packets for one captured image.

    `dets`: (cx, cy, w, h, conf) in this image's pixels; `context`: "ships" | "coast" | "empty" |
    "cloud" (the pre-filter's verdict; a cloud tile is P0 by context). `dark`: per-detection AIS
    flags (None = no AIS feed). `georef(x, y) -> (lat, lon)` adds the optional geo field.
    `origin` lifts the box into scene-global coordinates (x_global = x0 + x_local).
    `jpeg_fn((x0, y0, x1, y1), quality, progressive) -> bytes` overrides the JPEG encoder: a sweep
    memoises identical crops through it (same bytes), and then `img` only needs a `.shape`.
    `context_box` (x0, y0, x1, y1) is the coastal context image when `img` is a larger scene than
    the captured tile (default: the whole `img`), so ROI crops can still be cut across tile seams.
    """
    cfg = cfg or EncoderConfig()
    ids = ids or Ids()
    pk = packetizer or Packetizer(cfg.max_packet_data)
    pcfg, lod = cfg.priority, LoDConfig(conf_low=cfg.det_thr)
    dark = list(dark) if dark is not None else [False] * len(dets)
    H, W = img.shape[:2]
    levels = [P0] * len(dets)
    if context == "cloud":
        return Encoded([], [], levels)
    codec = CODEC_JPEG_PROGRESSIVE if cfg.progressive else CODEC_JPEG

    def image_product(kind, det_ids, box, quality, value):
        x0, y0, x1, y1 = box
        jpg = (jpeg_fn(box, quality, cfg.progressive) if jpeg_fn
               else encode_jpeg(np.ascontiguousarray(img[y0:y1, x0:x1]), quality, cfg.progressive))
        pid = ids()
        payload = _IMG_HDR.pack(pid, x0 + origin[0], y0 + origin[1], x1 - x0, y1 - y0, quality, codec) + jpg
        frac = (_IMG_HDR.size + min_decodable_fraction(jpg, cfg.progressive) * len(jpg)) / len(payload)
        return pid, Product(kind, image_id, tuple(det_ids), payload, pk.packetize(APID[kind], payload),
                            value, progressive=cfg.progressive, min_fraction=min(1.0, frac),
                            jpeg_bytes=len(jpg), jpeg_header=jpeg_header_bytes(jpg))

    records, products, coast_ctx = [], [], []
    for k, (cx, cy, bw, bh, conf) in enumerate(dets):
        ship = Ship(k, 0.0, bool(dark[k]), float(conf), context == "coast", float(max(bw, bh)))
        lv = classify(ship, context, pcfg, lod)
        levels[k] = lv
        if lv == P0:
            continue
        rec = SemanticRecord(image_id, t_capture_s, cfg.class_id, float(conf), cx + origin[0],
                             cy + origin[1], bw, bh, lv, context == "coast", bool(dark[k]),
                             georef(cx + origin[0], cy + origin[1]) if georef else None)
        coast_tile = context == "coast" and pcfg.coast_context_tile
        if lv >= P2:
            # the ROI: tight chip, or the wake crop when a dark vessel was escalated (priority.py)
            wake = dark[k] and lv >= P3
            box = crop_box(cx, cy, bw, bh, cfg.ctx_margin if wake else cfg.roi_margin, W, H)
            value = pcfg.roi_value
            if wake and lv >= P3 and not coast_tile:
                # off a coast the P3 context IS the wake crop: sending it again as a separate product
                # would ship identical pixels twice. One product carries both; it earns both values.
                value += pcfg.context_value
            rec.roi_id, prod = image_product("P2", (k,), box, cfg.ctx_quality if wake else cfg.roi_quality,
                                             value)
            products.append(prod)
            if wake and lv >= P3 and not coast_tile:
                rec.ctx_id = rec.roi_id
        if lv >= P3 and rec.ctx_id is None:
            if coast_tile:
                coast_ctx.append(k)                      # one shared tile context, below
            else:
                box = crop_box(cx, cy, bw, bh, cfg.ctx_margin, W, H)
                rec.ctx_id, prod = image_product("P3", (k,), box, cfg.ctx_quality, pcfg.context_value)
                products.append(prod)
        records.append((k, rec))
    if coast_ctx:                                       # the whole tile, encoded ONCE
        ctx_id, prod = image_product("P3", tuple(coast_ctx), context_box or (0, 0, W, H), cfg.coast_tile_quality,
                                     pcfg.context_value * len(coast_ctx))
        for k, r in records:
            if k in coast_ctx:
                r.ctx_id = ctx_id
        products.append(prod)

    # P1: the metadata. Packed last so every roi_id / ctx_id reference is already known.
    weight = [cfg.dark_weight if dark[k] else 1.0 for k in range(len(dets))]
    if cfg.metadata == "record":
        meta = [Product("P1", image_id, (k,), pack_record(r), pk.packetize(APID["P1"], pack_record(r)),
                        weight[k]) for k, r in records]
    elif records:
        t_s = int(t_capture_s)
        payload = _TILE_HDR.pack(image_id, t_s, int(round((t_capture_s - t_s) * 1000)) % 1000, len(records))
        payload += b"".join(pack_record(r, with_image=False) for _, r in records)
        meta = [Product("P1", image_id, tuple(k for k, _ in records), payload,
                        pk.packetize(APID["P1T"], payload), sum(weight[k] for k, _ in records))]
    else:
        meta = []
    return Encoded([r for _, r in records], meta + products, levels)


def encode_raw_image(img: np.ndarray, image_id: int, packetizer: Packetizer | None = None,
                     value: float = 1.0) -> Product:
    """B0's product: the ORIGINAL image as captured -- H x W x 3 uint8, uncompressed -- behind an
    18 B image header, in CCSDS packets. No detection, no selection, nothing dropped."""
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("raw image must be H x W x 3 uint8")
    h, w = img.shape[:2]
    if max(h, w) > 65535:
        raise ValueError("image side exceeds the 16-bit header field")
    payload = _IMG_HDR.pack(image_id, 0, 0, w, h, 0, CODEC_RAW) + np.ascontiguousarray(img).tobytes()
    pk = packetizer or Packetizer()
    return Product("RAW", image_id, (), payload, pk.packetize(APID["RAW"], payload), value)


def raw_packet_bytes(h: int, w: int, max_packet_data: int = 4096) -> int:
    """Exact packetized size of encode_raw_image without building it (header + pixels + 8 B/segment)."""
    n = _IMG_HDR.size + h * w * 3
    return n + (PKT_HDR + PKT_CRC) * math.ceil(n / (max_packet_data - PKT_CRC))


def to_items(enc: Encoded, created_s: float, ships_of=None, next_id=None) -> list[Item]:
    """Scheduler Items whose size is the real packet byte count. `ships_of(det) -> ground-truth ship
    ids` (for recall bookkeeping; empty for a false alarm) defaults to the detection index."""
    ships_of = ships_of or (lambda k: (k,))
    out = []
    for p in enc.products:
        sh = tuple(s for k in p.dets for s in ships_of(k))
        out.append(Item(next_id() if next_id else len(out), created_s, float(p.size), p.value, p.kind, sh,
                        progressive=p.progressive, min_fraction=p.min_fraction if p.progressive else 1.0))
    return out


# ================================================================================ ground side
def decode_downlink(stream: bytes) -> dict:
    """Ground decoder: packets -> records, ROI images, context images (CRC-checked, reassembled)."""
    records, rois, ctxs, raws = [], {}, {}, {}
    for apid, payload in depacketize(stream):
        if apid == APID["RAW"]:
            pid, x0, y0, w, h, _q, _codec = _IMG_HDR.unpack_from(payload, 0)
            raws[pid] = np.frombuffer(payload[_IMG_HDR.size:], np.uint8).reshape(h, w, 3)
        elif apid == APID["P1"]:
            records.append(unpack_record(payload)[0])
        elif apid == APID["P1T"]:
            image_id, t_s, t_ms, n = _TILE_HDR.unpack_from(payload, 0)
            off = _TILE_HDR.size
            for _ in range(n):
                r, off = unpack_record(payload, off, image_id, t_s + t_ms / 1000)
                records.append(r)
        elif apid in (APID["P2"], APID["P3"]):
            pid, x0, y0, w, h, q, codec = _IMG_HDR.unpack_from(payload, 0)
            jpg = payload[_IMG_HDR.size:]
            (rois if apid == APID["P2"] else ctxs)[pid] = {"x0": x0, "y0": y0, "w": w, "h": h,
                                                          "quality": q, "codec": codec, "jpeg": jpg}
    return {"records": records, "rois": rois, "contexts": ctxs, "raw_images": raws}


def decode_jpeg_prefix(jpg_prefix: bytes):
    """What the ground can show from a truncated progressive JPEG (EOI appended if missing)."""
    data = jpg_prefix if jpg_prefix.endswith(b"\xff\xd9") else jpg_prefix + b"\xff\xd9"
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
