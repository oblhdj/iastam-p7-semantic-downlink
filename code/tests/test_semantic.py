"""sat7.semantic: real serialized packets -- sizes, round trips, no double counting."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sat7.orbit import Pass
from sat7.priority import P1, P2, P3, PriorityConfig, classify, encode_priority
from sat7.scheduler import Item, LoDConfig, Ship, ValueGreedy, Workload, WorkloadConfig, simulate
from sat7.semantic import (APID, EncoderConfig, Ids, Packetizer, SemanticRecord, TMFraming,
                           decode_downlink, decode_jpeg_prefix, depacketize, encode_image, encode_jpeg,
                           min_decodable_fraction, pack_record, to_items, unpack_record)


def _rec(**kw):
    base = dict(image_id=7, t_capture_s=3600.25, class_id=0, confidence=0.8123, cx=1234.5, cy=987.6,
                w=40.2, h=12.3, level=1)
    return SemanticRecord(**(base | kw))


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(20, 60, (768, 768, 3), np.uint8)
    img[100:140, 200:260] = 220                                # something worth cropping
    return img


# ------------------------------------------------------------------ record
@pytest.mark.parametrize("kw,size", [({}, 26), ({"roi_id": 5}, 30), ({"roi_id": 5, "ctx_id": 9}, 34),
                                     ({"geo": (34.74, 10.76)}, 34),
                                     ({"geo": (34.74, 10.76), "roi_id": 5, "ctx_id": 9}, 42)])
def test_record_sizes_are_exact_and_round_trip(kw, size):
    r = _rec(**kw)
    b = pack_record(r)
    assert len(b) == size
    back, off = unpack_record(b)
    assert off == size and back.image_id == 7 and back.level == 1
    assert back.t_capture_s == pytest.approx(3600.25, abs=1e-3)
    assert back.confidence == pytest.approx(0.8123, abs=1 / 65535)
    assert (back.cx, back.cy, back.w, back.h) == pytest.approx((1234.5, 987.6, 40.2, 12.3), abs=0.05)
    assert back.roi_id == kw.get("roi_id") and back.ctx_id == kw.get("ctx_id")
    if "geo" in kw:
        assert back.geo == pytest.approx(kw["geo"], abs=1e-7)


def test_quantisation_keeps_threshold_decisions():
    # u16 confidence: a value just above a threshold stays above it after the round trip
    for thr in (0.25, 0.670):
        back, _ = unpack_record(pack_record(_rec(confidence=thr + 2e-5)))
        assert back.confidence >= thr


# ------------------------------------------------------------------ packets
def test_packet_is_header_plus_payload_plus_crc_and_crc_catches_corruption():
    pk = Packetizer()
    (p,) = pk.packetize(APID["P1"], pack_record(_rec()))
    assert len(p) == 6 + 26 + 2
    assert depacketize(p)[0] == (APID["P1"], pack_record(_rec()))
    bad = bytearray(p)
    bad[10] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        depacketize(bytes(bad))


def test_segmentation_round_trips_and_pays_a_header_per_segment():
    pk = Packetizer(max_packet_data=1000)
    payload = bytes(range(256)) * 20                            # 5,120 B -> 6 segments of 998
    pkts = pk.packetize(APID["P3"], payload)
    assert len(pkts) == 6 and sum(len(p) for p in pkts) == len(payload) + 6 * 8
    assert depacketize(b"".join(pkts)) == [(APID["P3"], payload)]


def test_sequence_count_wraps_at_14_bits():
    pk = Packetizer()
    pk._seq[APID["P1"]] = 0x3FFF
    a, b = pk.packetize(APID["P1"], b"x")[0], pk.packetize(APID["P1"], b"y")[0]
    assert (int.from_bytes(a[2:4], "big") & 0x3FFF, int.from_bytes(b[2:4], "big") & 0x3FFF) == (0x3FFF, 0)


def test_tm_framing_is_reported_not_double_counted():
    tm = TMFraming()
    assert tm.efficiency == pytest.approx(1107 / 1279)
    assert tm.efficiency > 0.8               # the capacity model's 0.8 already covers TM framing
    assert tm.on_air_bytes(1107) == 1279 and tm.on_air_bytes(1108) == 2 * 1279


# ------------------------------------------------------------------ encoder
def test_levels_follow_priority_classify_and_products_mirror_them():
    img = _scene()
    dets = [(230, 120, 60, 40, 0.9), (400, 400, 30, 20, 0.5), (600, 600, 20, 20, 0.1)]
    enc = encode_image(img, 1, 10.0, dets, "ships")
    assert enc.levels == [P1, P2, 0]
    kinds = sorted(p.kind for p in enc.products)
    assert kinds == ["P1", "P1", "P2"]                          # no product for the P0 box
    p2 = next(p for p in enc.products if p.kind == "P2")
    assert p2.payload[18:20] == b"\xff\xd8"                     # header + a real JPEG, no record inside


def test_coastal_context_tile_is_encoded_once_and_shared():
    img = _scene()
    dets = [(230, 120, 60, 40, 0.9), (400, 400, 30, 20, 0.5), (500, 300, 30, 20, 0.6)]
    enc = encode_image(img, 1, 0.0, dets, "coast")
    p3 = [p for p in enc.products if p.kind == "P3"]
    assert len(p3) == 1 and set(p3[0].dets) == {0, 1, 2}       # one tile, every coastal ship
    assert {r.ctx_id for r in enc.records} == {int.from_bytes(p3[0].payload[:4], "little")}


def test_dark_wake_is_not_sent_twice_off_coast():
    img = _scene()
    enc = encode_image(img, 1, 0.0, [(230, 120, 60, 40, 0.5)], "ships", dark=[True])
    assert enc.levels == [P3]
    images = [p for p in enc.products if p.kind != "P1"]
    assert len(images) == 1                                     # the wake crop serves as ROI AND context
    assert enc.records[0].roi_id == enc.records[0].ctx_id
    assert images[0].value == pytest.approx(PriorityConfig().roi_value + PriorityConfig().context_value)


def test_cloud_tile_sends_nothing():
    assert encode_image(_scene(), 1, 0.0, [(230, 120, 60, 40, 0.9)], "cloud").products == []


def test_tile_metadata_mode_hoists_image_id_and_time():
    img, dets = _scene(), [(100 + 60 * i, 120, 30, 20, 0.9) for i in range(5)]
    per_rec = encode_image(img, 1, 0.0, dets, "ships", EncoderConfig(metadata="record"))
    per_tile = encode_image(img, 1, 0.0, dets, "ships", EncoderConfig(metadata="tile"))
    assert sum(p.size for p in per_rec.products) == 5 * (26 + 8)
    assert sum(p.size for p in per_tile.products) == 11 + 5 * 16 + 8


def test_ground_decoder_reads_back_everything():
    img = _scene()
    dets = [(230, 120, 60, 40, 0.9), (400, 400, 30, 20, 0.5), (500, 300, 30, 20, 0.6)]
    enc = encode_image(img, 99, 7.5, dets, "coast", ids=Ids(), packetizer=Packetizer())
    g = decode_downlink(b"".join(q for p in enc.products for q in p.packets))
    # coast + "all" escalation: every record is P3 = metadata + ROI + (shared) context
    assert len(g["records"]) == 3 and len(g["rois"]) == 3 and len(g["contexts"]) == 1
    assert {r.image_id for r in g["records"]} == {99}
    ctx = next(iter(g["contexts"].values()))
    assert (ctx["w"], ctx["h"]) == (768, 768)


def test_progressive_prefix_at_the_measured_floor_decodes():
    img = _scene()
    jpg = encode_jpeg(img, 40, progressive=True)
    f = min_decodable_fraction(jpg, progressive=True)
    assert 0 < f < 0.5
    part = decode_jpeg_prefix(jpg[:int(np.ceil(f * len(jpg)))])
    assert part is not None and part.shape == img.shape
    assert min_decodable_fraction(encode_jpeg(img, 40, progressive=False), progressive=False) == 1.0


def test_items_carry_real_sizes_and_truncation_floors():
    img = _scene()
    enc = encode_image(img, 1, 0.0, [(230, 120, 60, 40, 0.5)], "ships")
    items = to_items(enc, 5.0)
    assert [it.size for it in items] == [float(p.size) for p in enc.products]
    roi = next(it for it in items if it.kind == "P2")
    assert roi.progressive and 0 < roi.min_fraction < 1


# ------------------------------------------------------------------ scheduler + priority changes
def test_scheduler_respects_an_items_truncation_floor():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    p = Pass(start + timedelta(hours=1), start + timedelta(hours=1, minutes=2),
             start + timedelta(hours=1, minutes=4), 30.0, 300.0)
    big = Item(1, 0.0, 1000.0, 1.0, "P3", (0,), progressive=True, min_fraction=0.5)
    sent = simulate([big], [p], start, ValueGreedy()).sent
    assert sent == []                      # 30% of the bytes fit, but 50% is the decodable floor
    big.min_fraction = 0.1
    assert simulate([big], [p], start, ValueGreedy()).sent[0].fraction == pytest.approx(0.3)


def test_coast_escalation_rule_is_configurable():
    lod = LoDConfig(conf_low=0.25)
    confident = Ship(0, 0.0, False, 0.9, True)
    assert classify(confident, "coast", PriorityConfig(), lod) == P3                 # "all" (default)
    assert classify(confident, "coast", PriorityConfig(coast_escalation="uncertain"), lod) == P1
    unsure = Ship(1, 0.0, False, 0.4, True)
    assert classify(unsure, "coast", PriorityConfig(coast_escalation="uncertain"), lod) == P3


def test_modeled_priority_encoder_no_longer_sends_the_wake_twice():
    dark_unsure = Ship(0, 0.0, True, 0.4, False, size_px=60.0)
    wl = Workload([dark_unsure], [(0.0, "ships", (0,))], WorkloadConfig(), false_alarms=[0])
    items = encode_priority(wl, LoDConfig(conf_low=0.25), PriorityConfig())
    assert sorted(i.kind for i in items) == ["P1", "P2"]        # was P1 + P2(wake) + P3(wake)
