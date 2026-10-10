"""Units and bookkeeping, end to end: semantic encoding, coordinate conversion, duplicate removal,
data sizes, energy, communication routing -- and that the documents quote what the results say.

Each block states the definition it pins (sat7/accounting.py) and then checks that the module
that carries its own copy of the arithmetic agrees with it."""
import csv
import importlib.util
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from sat7.accounting import (MB, LatencyBudget, data_reduction_percent, measured_stage_times_s,
                             raw_image_bytes, reduction_factor, stage_energy_J, tx_energy_J, tx_time_s)
from sat7.b2_sahi_fusion import SahiConfig, plan_slices
from sat7.comms import CommsConfig, Links, simulate_comms
from sat7.energy import EnergyModel
from sat7.orbit import Pass
from sat7.perception import PerceptionConfig, detect_image
from sat7.relay import ISLWindow, LinkParams, route_item
from sat7.scheduler import (RAW_TILE_BYTES, FIFO, Item, Ship, ValueGreedy, Workload, WorkloadConfig,
                            encode_raw, simulate)
from sat7.semantic import Ids, Packetizer, decode_downlink, encode_image, encode_raw_image, raw_packet_bytes

CODE = Path(__file__).resolve().parents[1]
RES = CODE / "results"
START = datetime(2026, 9, 18, tzinfo=timezone.utc)


def _load_script(name):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(CODE / "scripts"))
    spec = importlib.util.spec_from_file_location(name, CODE / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _results(name):
    p = RES / name
    if not p.exists():
        pytest.skip(f"results/{name} not generated")
    return json.loads(p.read_text(encoding="utf-8"))


# ====================================================================== units
def test_a_megabyte_is_decimal_and_a_rate_is_in_bits():
    assert MB == 1_000_000
    assert tx_time_s(1 * MB, 8e6) == 1.0                        # 1 MB at 8 Mbit/s takes one second
    assert tx_time_s(125_000, 1e6) == 1.0                       # 125 kB = 1 Mbit
    assert tx_energy_J(15.0, 1 * MB, 2.53e6) == pytest.approx(15.0 * 8e6 / 2.53e6)   # 47.4 J
    assert stage_energy_J(28.0, 38.6) == pytest.approx(1.0808)  # ms -> s before watts x seconds
    with pytest.raises(ValueError):
        tx_time_s(1.0, 0.0)


def test_forgetting_bytes_to_bits_would_be_exactly_eight_times_off():
    em = EnergyModel()
    wrong = em.powers_W["tx"] * 1 * MB / em.r_tx_bps           # bytes / (bit/s): the bug fixed on 10 Oct
    assert em.e_comm_direct_J(1 * MB) == pytest.approx(8 * wrong)


# ====================================================================== data-size calculations
def test_raw_baseline_is_every_pixel_as_uncompressed_rgb():
    assert RAW_TILE_BYTES == raw_image_bytes(768, 768) == 768 * 768 * 3 == 1_769_472
    assert raw_image_bytes(3072, 3072) == 16 * RAW_TILE_BYTES   # a 4x4 scene is 16 tiles, no more


def test_data_reduction_percent_and_factor_are_the_same_statement():
    assert data_reduction_percent(1.0, 100.0) == pytest.approx(99.0)
    assert data_reduction_percent(100.0, 100.0) == 0.0
    f = reduction_factor(176.266, 60124.299)
    assert data_reduction_percent(176.266, 60124.299) == pytest.approx(100 * (1 - 1 / f))
    assert data_reduction_percent(2.0, 8.0) == data_reduction_percent(2.0 * MB, 8.0 * MB)   # unit-free ratio
    with pytest.raises(ValueError):
        data_reduction_percent(1.0, 0.0)
    with pytest.raises(ValueError):
        reduction_factor(0.0, 1.0)


def test_bent_pipe_offers_whole_non_cloud_tiles():
    ships = [Ship(0, 10.0, False, 0.9, False, 40.0)]
    wl = Workload(ships, [(10.0, "ships", (0,)), (20.0, "cloud", ()), (30.0, "empty", ()), (40.0, "coast", ())],
                  WorkloadConfig(hours=1.0, tiles_per_day=4))
    raw = encode_raw(wl)
    assert [i.size for i in raw] == [RAW_TILE_BYTES] * 3        # the cloud tile is not in the baseline
    assert sum(i.size for i in raw) == 3 * raw_image_bytes(768, 768)


@pytest.mark.parametrize("payload", [1, 26, 4093, 4094, 4095, 8188, 8189, 66_527 + 18])
def test_packet_overhead_is_eight_bytes_per_segment(payload):
    wire = sum(len(p) for p in Packetizer().packetize(0x103, bytes(payload)))
    assert wire == payload + 8 * math.ceil(payload / 4094)      # 6 B header + 2 B CRC, 4094 B of data each


def test_raw_image_packets_are_pixels_plus_headers_and_nothing_else():
    img = np.zeros((96, 128, 3), np.uint8)
    prod = encode_raw_image(img, 1, 0.0)
    stream = b"".join(prod.packets)
    assert len(stream) == prod.size == raw_packet_bytes(96, 128)
    assert raw_packet_bytes(96, 128) > raw_image_bytes(96, 128)               # headers cost bytes ...
    assert raw_packet_bytes(96, 128) < 1.01 * raw_image_bytes(96, 128)        # ... well under 1%


def test_headline_table_reduction_is_raw_offered_over_bytes_sent():
    p = RES / "wp6_real_table.csv"
    if not p.exists():
        pytest.skip("results/wp6_real_table.csv not generated")
    with p.open(newline="", encoding="utf-8") as f:
        t = {r["strategy"]: r for r in csv.DictReader(f)}
    ours, bent = t["Ours: LoD + value-greedy"], t["Bent pipe (raw, FIFO)"]
    d_tx, d_raw = float(ours["MB_sent"]), float(bent["MB_offered"])
    assert float(ours["data_reduction_x"]) == pytest.approx(reduction_factor(d_tx, d_raw), rel=1e-12)
    assert float(ours["MB_sent"]) <= float(ours["MB_offered"])                # cannot send more than exists
    n_tiles = d_raw * MB / RAW_TILE_BYTES                                     # the baseline is whole tiles
    assert 0.80 * 40_000 < n_tiles < 0.90 * 40_000                            # 85% of a 40k-tile day is cloud-free


def test_measured_coastal_model_products_add_up():
    w = _results("wp28_coast_tile_model.json")
    for m in w["models"].values():
        assert sum(p["MB"] for p in m["products"].values()) == pytest.approx(m["D_tx_offered_MB"], abs=0.02)
        assert sum(p["share"] for p in m["products"].values()) == pytest.approx(1.0, abs=1e-3)
        assert m["reduction_x"] == pytest.approx(w["days"]["D_raw_MB_mean"] / m["D_tx_offered_MB"], rel=1e-3)
        assert m["DR_percent"] == pytest.approx(
            data_reduction_percent(m["D_tx_offered_MB"], w["days"]["D_raw_MB_mean"]), abs=2e-3)


# ====================================================================== semantic encoding
def _scene(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(20, 60, (768, 768, 3), np.uint8)
    img[100:140, 200:260] = 220
    return img


def test_semantic_stream_length_is_the_sum_of_its_products_and_decodes():
    dets = [(230, 120, 60, 40, 0.9), (400, 400, 30, 20, 0.5), (600, 600, 20, 20, 0.1)]
    enc = encode_image(_scene(), 5, 12.0, dets, "ships", ids=Ids(), packetizer=Packetizer())
    stream = b"".join(q for p in enc.products for q in p.packets)
    assert len(stream) == sum(p.size for p in enc.products)                   # size IS the bytes on the wire
    assert all(p.size == sum(len(q) for q in p.packets) for p in enc.products)
    g = decode_downlink(stream)
    assert len(g["records"]) == 2 and len(g["rois"]) == 1                     # the 0.1 box is below the cut
    assert {round(r.confidence, 2) for r in g["records"]} == {0.9, 0.5}
    dr = data_reduction_percent(len(stream), raw_image_bytes(768, 768))
    assert 99.0 < dr < 100.0                                                  # a few kB against 1.77 MB


def test_semantic_bytes_grow_with_what_is_sent_never_with_what_is_dropped():
    img = _scene()
    one = encode_image(img, 1, 0.0, [(230, 120, 60, 40, 0.9)], "ships")
    two = encode_image(img, 1, 0.0, [(230, 120, 60, 40, 0.9), (400, 400, 30, 20, 0.5)], "ships")
    plus_p0 = encode_image(img, 1, 0.0, [(230, 120, 60, 40, 0.9), (600, 600, 20, 20, 0.1)], "ships")
    size = lambda e: sum(p.size for p in e.products)
    assert size(two) > size(one) == size(plus_p0)                             # a discarded box costs nothing
    assert size(encode_image(img, 1, 0.0, [(230, 120, 60, 40, 0.9)], "cloud")) == 0


# ====================================================================== coordinates + duplicates
W_IMG, H_IMG = 2000, 1500
SHIPS = [(300, 300), (690, 300), (690, 690), (1500, 1200), (1900, 80)]        # global centres (x, y)


def _blob_scene():
    img = np.zeros((H_IMG, W_IMG, 3), np.uint8)
    for x, y in SHIPS:
        img[y - 6:y + 6, x - 10:x + 10] = 255                                 # a 20 x 12 px "ship"
    return img


def _blob_detector(conf_of_crop=lambda k: 0.9):
    """Finds the bright blobs of each crop and reports them in THAT CROP's coordinates."""
    import cv2

    def detect(crops):
        out = []
        for k, crop in enumerate(crops):
            n, _lab, stats, cent = cv2.connectedComponentsWithStats((crop[..., 0] > 128).astype(np.uint8))
            out.append([(float(cent[i][0]) + 0.5, float(cent[i][1]) + 0.5, float(stats[i][2]), float(stats[i][3]),
                         conf_of_crop(k)) for i in range(1, n) if stats[i][2] == 20 and stats[i][3] == 12])
        return out
    return detect


def _windows_holding(x, y, slices):
    return [k for k, (x0, y0, w, h) in enumerate(slices)
            if x0 <= x - 10 and x + 10 <= x0 + w and y0 <= y - 6 and y + 6 <= y0 + h]


def test_every_window_reports_a_ship_at_the_same_global_position():
    cfg = PerceptionConfig(mode="sahi", window=768, overlap=0.2, fuse=False)
    slices = plan_slices(W_IMG, H_IMG, SahiConfig(window=768, overlap=0.2))
    dets = detect_image(_blob_scene(), _blob_detector(), cfg)
    assert len(dets) == sum(len(_windows_holding(x, y, slices)) for x, y in SHIPS)   # one per window
    for cx, cy, w, h, _c in dets:                                # x_global = x0 + x_local, y likewise
        assert min(abs(cx - x) + abs(cy - y) for x, y in SHIPS) < 1e-6
        assert (w, h) == (20.0, 12.0)                            # a shift never rescales a box
    assert max(len(_windows_holding(x, y, slices)) for x, y in SHIPS) >= 4   # the corner ship is seen 4 times


def test_fusion_keeps_one_box_per_ship_and_the_most_confident_duplicate():
    slices = plan_slices(W_IMG, H_IMG, SahiConfig(window=768, overlap=0.2))
    conf = lambda k: 0.50 + 0.03 * k                             # later windows are more confident
    fused = detect_image(_blob_scene(), _blob_detector(conf), PerceptionConfig(mode="sahi", window=768, overlap=0.2))
    assert len(fused) == len(SHIPS)                              # duplicates gone, no ship lost
    for x, y in SHIPS:
        box = min(fused, key=lambda d: abs(d[0] - x) + abs(d[1] - y))
        assert abs(box[0] - x) + abs(box[1] - y) < 1e-6
        assert box[4] == pytest.approx(conf(max(_windows_holding(x, y, slices))))


def test_fusion_does_not_merge_two_different_ships_that_sit_close():
    img = np.zeros((768, 768, 3), np.uint8)
    for x in (300, 326):                                         # 26 px apart: boxes 20 px wide, IoU 0
        img[294:306, x - 10:x + 10] = 255
    fused = detect_image(np.pad(img, ((0, 400), (0, 400), (0, 0))), _blob_detector(),
                         PerceptionConfig(mode="sahi", window=768, overlap=0.2))
    assert sorted(round(d[0]) for d in fused) == [300, 326]


def test_whole_image_mode_needs_no_conversion_and_no_duplicate_removal():
    dets = detect_image(_blob_scene(), _blob_detector(), PerceptionConfig(mode="whole"))
    assert sorted((round(d[0]), round(d[1])) for d in dets) == sorted(SHIPS)


# ====================================================================== energy
def test_library_energy_equals_power_times_transmit_time():
    em = EnergyModel()
    d = 3.2 * MB
    assert em.e_comm_direct_J(d) == pytest.approx(tx_energy_J(em.powers_W["tx"], d, em.r_tx_bps))
    r = em.e_comm_relay_J(d, p_isl_W=12.0, r_isl_bps=1.0e6)
    assert r["E_ISL_J"] == pytest.approx(tx_energy_J(12.0, d, 1.0e6))
    assert r["E_GS_J"] == pytest.approx(tx_energy_J(em.powers_W["tx"], d, em.r_tx_bps))
    assert r["E_relay_J"] == pytest.approx(r["E_ISL_J"] + r["E_GS_J"]) and r["E_relay_J"] > em.e_comm_direct_J(d)
    f = em.relay_energy_fn()(d, 15.0, 12.0, 2.0e6, 1.0e6)
    assert f["isl_time_s"] == pytest.approx(tx_time_s(d, 1.0e6)) and f["gs_time_s"] == pytest.approx(tx_time_s(d, 2.0e6))


def test_processing_energy_is_the_sum_of_power_times_stage_time():
    em = EnergyModel()
    proc = em.e_proc_per_tile_J("cpu_onnx")
    t, p = em.stage_times_ms, em.powers_W["cpu"]
    expect = sum(stage_energy_J(p, t[k]) for k in ("preprocess_decode_cpu", "gate_cpu", "detect_cpu_onnx",
                                                   "fuse_cpu", "semantic_cpu"))
    assert proc["E_prop"] == pytest.approx(expect)
    assert proc["E_base"] - proc["E_prop"] == pytest.approx(stage_energy_J(p, t["classic_prefilter_cpu"] - t["gate_cpu"]))
    assert EnergyModel.energy_saving_ratio(100.0, 25.0) == 0.75


def test_the_relay_energy_script_uses_the_same_arithmetic():
    wp19 = _load_script("wp19_relay_energy")
    d = 7.5 * MB
    assert wp19.direct_energy(d, 15.0, 2.5e6)["E_comm_J"] == pytest.approx(tx_energy_J(15.0, d, 2.5e6))
    r = wp19.relay_energy(d, 15.0, 12.0, 2.5e6, 1.5e6)
    assert r["E_relay_J"] == pytest.approx(tx_energy_J(12.0, d, 1.5e6) + tx_energy_J(15.0, d, 2.5e6))
    assert r["isl_time_s"] == pytest.approx(tx_time_s(d, 1.5e6))


def test_committed_energy_results_reproduce_from_their_own_inputs():
    w17, w19 = _results("wp17_energy_model.json"), _results("wp19_relay_energy.json")
    P, day, dl = w17["powers_W_assumption"], w17["E_per_day_kJ"], w17["downlink"]
    e_comm = tx_energy_J(P["tx"], dl["D_sent_MB_day"] * MB, dl["R_eff_Mbps"] * 1e6) / 1e3
    assert e_comm == pytest.approx(day["E_comm_proposed_semantic"], rel=2e-3)          # kJ, rounded in the file
    assert tx_energy_J(P["tx"], dl["D_raw_MB_day"] * MB, dl["R_eff_Mbps"] * 1e6) / 1e3 == pytest.approx(
        day["E_comm_bent_pipe_raw"], rel=2e-3)
    assert day["E_total_proposed"] == pytest.approx(day["E_proc_proposed"] + day["E_comm_proposed_semantic"], abs=0.02)
    assert day["proc_vs_comm_ratio_proposed"] == pytest.approx(day["E_proc_proposed"] / day["E_comm_proposed_semantic"], rel=2e-3)
    assert day["ES_total_vs_bent_pipe"] == pytest.approx(1 - day["E_total_proposed"] / day["E_comm_bent_pipe_raw"], abs=1e-4)
    pp = w19["per_path_energy_daily_kJ"]
    assert pp["E_relay"] == pytest.approx(pp["E_ISL"] + pp["E_GS"], abs=2e-3)
    assert pp["E_direct"] == pytest.approx(e_comm, rel=2e-3)                           # same D, P_tx and rate as wp17
    assert "ASSUMPTION" in w17["label_note"] and all("ASSUMPTION" in k for k in w19["powers_W"])


def test_sahi_energy_variant_is_the_default_plus_extra_detector_calls_and_nothing_else():
    w17 = _results("wp17_energy_model.json")
    if "sahi_variant" not in w17:
        pytest.skip("wp17 generated without wp26_b0_b4.json")
    sv, dflt = w17["sahi_variant"], w17["E_proc_per_tile_J"]["cpu_onnx_DEFAULT"]
    P, T, calls = w17["powers_W_assumption"], w17["stage_times_ms"], w17["sahi_variant"]["detector_calls_per_tile"]
    assert calls == 25 / 16                                        # 25 windows on a 4x4-tile scene
    detect = stage_energy_J(P["cpu"], T["detect_cpu_onnx"])
    for which, key in (("proposed", "E_prop"), ("baseline", "E_base")):
        stages = sv["E_proc_per_tile_J"][which]
        assert stages["detect"] == pytest.approx(calls * detect)
        assert {k: v for k, v in stages.items() if k != "detect"} == \
            {k: v for k, v in dflt[which].items() if k != "detect"}           # gate, prefilter, the rest: untouched
        assert sv["E_proc_per_tile_J"][key] == pytest.approx(dflt[key] + (calls - 1) * detect, abs=1e-3)
    assert sv["gate_calls_per_tile"] == 1.0 and sv["fuse_cpu_ms"] == 0.0 and "ASSUMPTION" in sv["fuse_note"]
    d = sv["E_per_day_kJ"]
    assert d["E_proc_proposed"] == pytest.approx(sv["E_proc_per_tile_J"]["E_prop"] * d["tiles_per_day"] / 1e3, abs=0.03)
    assert d["E_comm_proposed_semantic"] == w17["E_per_day_kJ"]["E_comm_proposed_semantic"]   # the radio does not change
    assert sv["E_proc_per_tile_J"]["ES_proc"] < dflt["ES_proc"]            # detection is a bigger share, so the gate saves less
    assert "VARIANT" in sv["label"] and "ESTIMATE" in sv["label"]


# ====================================================================== latency
def test_total_latency_is_the_sum_of_four_sequential_stages():
    b = LatencyBudget(processing_s=0.040, encoding_s=0.009, communication_s=4.5 * 3600, decoding_s=0.00004)
    assert b.total_s == pytest.approx(0.040 + 0.009 + 16200 + 0.00004)
    assert b.non_communication_s == pytest.approx(0.04904)
    assert b.as_dict()["non_communication_share"] < 1e-5         # hours of waiting dwarf the compute


def test_simulated_latency_is_the_wait_for_a_pass_plus_the_transmission():
    p = Pass(START + timedelta(seconds=1000), START + timedelta(seconds=1000), START + timedelta(seconds=1400), 45.0, 1e6)
    it = Item(1, 100.0, 250_000.0, 1.0, "P2", (0,), progressive=True)        # captured at t = 100 s
    sent = simulate([it], [p], START, ValueGreedy(), 8e9).sent[0]
    wait, in_pass = 1000.0 - 100.0, 400.0 * 250_000.0 / 1e6                   # a quarter of the pass
    assert sent.delivered_s - it.created_s == pytest.approx(wait + in_pass)


def test_measured_stage_times_are_seconds_and_small_against_the_tile_interval():
    if not (RES / "wp25_semantic_packets.json").exists() or not (RES / "wp17_energy_model.json").exists():
        pytest.skip("results not generated")
    st, sahi = measured_stage_times_s(RES, "cpu_onnx"), measured_stage_times_s(RES, "cpu_onnx", calls_per_tile=1.5625)
    assert 0.01 < st["processing_s"] < 0.2 and sahi["processing_s"] > st["processing_s"]   # tens of ms, not s or us
    assert 0 < st["decoding_s"] < 0.01 and all(0 < v < 0.1 for v in st["encoding_s"].values())
    onboard = sahi["processing_s"] + max(sahi["encoding_s"].values())
    assert onboard < 86400 / 160_000                               # done before the next tile, even at 160k/day


# ====================================================================== communication routing
def _gp(h, cap=1e6, dur=400):
    t = START + timedelta(hours=h)
    return Pass(t, t, t + timedelta(seconds=dur), 45.0, cap)


def _isl(h, minutes=10):
    t = START + timedelta(hours=h)
    return ISLWindow(t, t + timedelta(minutes=minutes), 2000.0, 2500.0)


def test_a_relayed_item_pays_and_waits_on_both_links():
    links = Links([_gp(10)], [_isl(0.5)], [_gp(2)])               # ISL at 0.5 h, relay pass at 2 h, own pass at 10 h
    cfg = CommsConfig()
    it = Item(1, 60.0, 20_000.0, 1.0, "P2", (0,), progressive=True)
    d = simulate_comms([it], links, START, ValueGreedy(), cfg).deliveries[0]
    assert d.path == "relay"
    r_isl = cfg.isl_rate_bps * cfg.isl_efficiency
    r_gs = links.relay_passes[0].capacity_bytes * 8 / links.relay_passes[0].duration_s
    assert d.tx_s == pytest.approx({"isl": tx_time_s(it.size, r_isl), "relay_ground": tx_time_s(it.size, r_gs)})
    assert d.energy_J["isl"] == pytest.approx(cfg.p_isl_W * d.tx_s["isl"])            # each leg: its power x its time
    assert d.energy_J["relay_ground"] == pytest.approx(cfg.p_tx_W * d.tx_s["relay_ground"])
    assert d.delivered_s >= 2 * 3600 > 0.5 * 3600 + d.tx_s["isl"]                      # ISL first, then the relay's pass
    assert d.latency_s == pytest.approx(d.delivered_s - 60.0)


def test_direct_and_relay_are_costed_with_the_same_ground_link():
    links = Links([_gp(1)], [_isl(0.5)], [_gp(2)])                # the direct pass comes first: direct wins
    cfg = CommsConfig()
    it = Item(1, 60.0, 20_000.0, 1.0, "P2", (0,), progressive=True)
    direct = simulate_comms([it], links, START, ValueGreedy(), cfg).deliveries[0]
    assert direct.path == "direct" and set(direct.tx_s) == {"ground"}
    later = Links([_gp(10)], [_isl(0.5)], [_gp(2)])
    relay = simulate_comms([it], later, START, ValueGreedy(), cfg).deliveries[0]
    assert relay.tx_s["relay_ground"] == pytest.approx(direct.tx_s["ground"])          # same pass model, same rate
    assert sum(relay.energy_J.values()) > sum(direct.energy_J.values())                # the ISL leg is extra


def test_window_level_router_charges_both_legs_and_waits_for_both_windows():
    lp, em = LinkParams(), EnergyModel()
    joules = dict(direct_energy=em.direct_energy_fn(), relay_energy=em.relay_energy_fn())   # as wp18 injects wp19's
    t0 = START + timedelta(minutes=5)
    out = route_item(50_000.0, t0, [_gp(10)], [_isl(0.5)], [_gp(0.2), _gp(2)], lam_E=0.0, lam_T=1.0, link=lp, **joules)
    assert out["path"] == "relay"
    assert out["T_s"] == pytest.approx((START + timedelta(hours=2) - t0).total_seconds())   # the relay pass AFTER the ISL
    e_isl, e_gs = tx_energy_J(lp.p_isl_W, 50_000.0, lp.r_isl_bps), tx_energy_J(lp.p_tx_W, 50_000.0, lp.r_gs_bps)
    assert out["E"] == pytest.approx(e_isl + e_gs)
    alone = route_item(50_000.0, t0, [_gp(10)], [_isl(0.5)], [_gp(2)], lam_E=0.0, lam_T=1.0, link=lp,
                       isl_enabled=False, **joules)
    assert alone["path"] == "direct" and alone["E"] == pytest.approx(e_gs)
    # without injected callables the router's energies are unit-less placeholders, and say so
    from sat7.relay import default_direct_energy, default_relay_energy
    assert default_direct_energy(1.0, 1.0, 1.0)["label"] == default_relay_energy(1.0, 1.0, 1.0, 1.0, 1.0)["label"] \
        == "TARGET-placeholder"


def test_fifo_and_value_greedy_see_the_same_link_and_the_same_items():
    passes = [_gp(1, cap=60_000.0), _gp(3, cap=60_000.0)]
    items = [Item(i, 10.0 * i, 20_000.0, float(1 + i % 3), "P2", (i,), progressive=True) for i in range(9)]
    a, b = simulate(items, passes, START, ValueGreedy(), 8e9), simulate(items, passes, START, FIFO(), 8e9)
    assert a.capacity == b.capacity == 120_000.0 and a.bytes_offered == b.bytes_offered == 180_000.0
    assert a.bytes_sent <= a.capacity and b.bytes_sent <= b.capacity          # nobody gets a bigger link


# ====================================================================== documents vs results
def test_every_headline_number_in_the_documents_matches_its_results_file():
    wp29 = _load_script("wp29_validation")
    if not (RES / "wp6_real_table_modeledcoast.csv").exists():
        pytest.skip("results not generated")
    out = wp29.check_claims(RES, CODE.parent)
    assert out["claims"] >= 30 and out["claim_document_pairs"] >= 150
    assert out["missing"] == [], out["missing"]


def test_committed_validation_has_no_failed_check_and_names_its_gaps():
    v = _results("wp29_validation.json")
    for num, item in v["checklist"].items():
        assert all(c["ok"] for c in item["checks"]), (num, item["failed_checks"])
        assert item["status"] == ("PARTIAL" if any(x.startswith("GAP: ") for x in item["limits"]) else "PASS")
    assert {n for n, i in v["checklist"].items() if i["status"] == "PASS"} == {"7", "8", "9"}
    run = v["steady_state"]["runs"]
    assert run["share_0.25_Value-greedy (ours)"]["offered_over_sustained"] > 1.0   # one day overfills the share ...
    assert run["share_0.35_Value-greedy (ours)"]["offered_over_sustained"] < 1.0   # ... a 35% share would hold it
    vg = [d["ship_recall"] for d in run["share_0.25_Value-greedy (ours)"]["per_day"]]
    assert max(vg) - min(vg) < 0.02                                                # value-greedy holds recall
    fifo = [d["latency_med_h"] for d in run["share_0.25_FIFO"]["per_day"]]
    assert fifo == sorted(fifo) and fifo[-1] > 4 * fifo[0]                         # FIFO's backlog only grows
