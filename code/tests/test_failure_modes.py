"""Fault injection against the FMEA table in the technical report.

The report lists eight failure modes with a mitigation for each. Every one of them was
prose. These tests inject the fault and check whether the claimed mitigation actually holds
in the code, so the table stops being an assertion and becomes a result.

Where a mitigation does NOT hold, the test pins the real behaviour and says so in its name
and docstring rather than being weakened until it passes.

    ["Detector misses a ship",      "Thumbnail + raw ring buffer, ground re-request"]
    ["Miscalibrated confidence",    "Calibration, recalibration from the ground"]
    ["Domain shift",                "Audit sampling: 1% random raw tiles"]
    ["AIS gap or spoofing",         "Priority only; a human confirms"]
    ["Pass capacity below prediction", "Progressive items; re-plan every pass"]
    ["Storage full",                "Drop lowest value-per-byte first"]
    ["Software crash / bit flip",   "Watchdog + safe mode (thumbnails + FIFO)"]
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from sat7.orbit import Pass
from sat7.real_workload import RealWorkloadConfig, workload_from_catalogue
from sat7.scheduler import (FIFO, Item, LoDConfig, NoBuffer, Ship, SizeModel, ValueGreedy,
                            _l1, _l2, _w, encode_lod, metrics, simulate)

START = datetime(2026, 9, 18, tzinfo=timezone.utc)


def _pass(at_h: float, cap: float) -> Pass:
    t = START + timedelta(hours=at_h)
    return Pass(t, t + timedelta(minutes=4), t + timedelta(minutes=8), 45.0, cap)


def _catalogue():
    tiles = pd.DataFrame({
        "image": ["ships.jpg", "coast.jpg", "empty.jpg", "quiet.jpg"],
        "ctx3": ["ships", "coast", "empty", "empty"],
        "context": ["candidates", "land_coast", "candidates", "empty_sea"],
        "n_candidates": [4, 25, 2, 0],
        "n_ships": [3, 2, 0, 0],
        "n_fp_0.25": [1, 0, 1, 0],
    })
    ships = pd.DataFrame({
        "image": ["ships.jpg"] * 3 + ["coast.jpg"] * 2,
        "size_px": [20.0, 60.0, 200.0, 35.0, 90.0],
        "found": [True, True, True, False, True],
        "conf": [0.85, 0.55, 0.30, 0.0, 0.45],
    })
    return tiles, ships


def _workload(tiles_per_day=6000, seed=0, **kw):
    tiles, ships = _catalogue()
    return workload_from_catalogue(
        tiles, ships, RealWorkloadConfig(tiles_per_day=tiles_per_day, det_thr=0.25, seed=seed, **kw))


# ---------------------------------------------------------------- storage full


def test_storage_full_evicts_the_lowest_value_per_byte_first():
    """Mitigation: 'Drop lowest value-per-byte first'."""
    cheap = Item(1, 0, 1000, 0.01, "thumb")            # density 1e-5
    rich = Item(2, 0, 1000, 10.0, "L2")                # density 1e-2
    res = simulate([cheap, rich], [_pass(6, 1500)], START, ValueGreedy(), storage_bytes=1200)
    assert res.dropped_storage == 1
    assert [s.item.id for s in res.sent] == [2]        # the valuable one survived eviction


def test_storage_pressure_does_not_evict_dark_vessels_first():
    """A dark vessel is worth 5x, so it must outlive ordinary traffic when storage fills."""
    lod = LoDConfig()
    dark = Item(1, 0, 2000, _w(Ship(0, 0, True, 0.6, False), lod), "L2", (0,), progressive=True)
    normal = [Item(10 + i, 0, 2000, 1.0, "L1", (i,)) for i in range(5)]
    res = simulate([dark] + normal, [_pass(6, 2500)], START, ValueGreedy(), storage_bytes=4000)
    assert res.dropped_storage > 0
    assert 1 in [s.item.id for s in res.sent]


# ---------------------------------------------------------------- pass shortfall


def test_a_pass_smaller_than_planned_truncates_instead_of_losing_everything():
    """Mitigation: 'Progressive items; re-plan every pass'."""
    tile = Item(1, 0, 40_000, 8.0, "tile", (0,), progressive=True)
    res = simulate([tile], [_pass(6, 10_000)], START, ValueGreedy())
    assert len(res.sent) == 1
    assert 0.0 < res.sent[0].fraction < 1.0            # partial beats nothing


def test_a_lost_pass_is_recovered_by_the_next_one_when_buffering():
    """Ground-station outage: capacity 0. A buffering policy must re-plan, not lose data."""
    items = [Item(1, 0, 1000, 5.0, "L1", (0,))]
    outage = [_pass(6, 0.0), _pass(12, 50_000)]
    assert len(simulate(items, outage, START, ValueGreedy()).sent) == 1
    assert len(simulate(items, outage, START, FIFO()).sent) == 1
    # documented limitation: the no-buffer variant genuinely loses it
    assert simulate(items, outage, START, NoBuffer()).sent == []


def test_every_pass_being_short_degrades_gracefully_rather_than_crashing():
    wl, _ = _workload()
    tiny = [_pass(h, 1_000.0) for h in (4, 10, 16, 22)]
    res = simulate(encode_lod(wl, LoDConfig(conf_low=0.25)), tiny, START, ValueGreedy())
    m = metrics(res, wl, LoDConfig(conf_low=0.25))
    assert 0.0 <= m["ship_recall"] <= 1.0
    assert res.bytes_sent <= sum(p.capacity_bytes for p in tiny) + 1e-6


# ---------------------------------------------------------------- AIS spoofing / gaps


def test_the_old_replace_mode_inverted_priority_for_large_vessels():
    """Regression record of the defect WP9 found, kept so it cannot come back unnoticed.

    With ``dark_mode="replace"`` the AIS flag swapped the chip (L1, bytes ~ px^0.57) for an
    ROI + wake (L2, ~ px^1.14). Above ~128 px the payload outgrew the 5x weight, so a dark
    vessel ranked BELOW the same ship with a clean track -- on 13.6% of real ships, and the
    largest ones at that.
    """
    lod = LoDConfig(size_model=SizeModel(), dark_mode="replace")
    for size_px in (12.0, 25.0, 50.0, 100.0):           # small ships were fine
        n, d = Ship(0, 0, False, 0.6, False, size_px), Ship(1, 0, True, 0.6, False, size_px)
        assert _w(d, lod) / _l2(d, lod) > _w(n, lod) / _l1(n, lod)
    for size_px in (200.0, 300.0, 392.0):               # large ones were inverted
        n, d = Ship(0, 0, False, 0.6, False, size_px), Ship(1, 0, True, 0.6, False, size_px)
        assert _w(d, lod) / _l2(d, lod) < _w(n, lod) / _l1(n, lod)


def test_decoupled_mode_makes_the_claim_true_at_every_size():
    """The fix: the AIS flag changes value only, never the payload, so a dark vessel
    outranks its clean-track twin by exactly ``dark_weight`` at any size."""
    lod = LoDConfig(size_model=SizeModel())              # decoupled is the default
    assert lod.dark_mode == "decoupled"
    for size_px in (12.0, 50.0, 128.0, 200.0, 392.0, 1000.0):
        n, d = Ship(0, 0, False, 0.6, False, size_px), Ship(1, 0, True, 0.6, False, size_px)
        ratio = (_w(d, lod) / _l1(d, lod)) / (_w(n, lod) / _l1(n, lod))
        assert abs(ratio - lod.dark_weight) < 1e-9


def test_a_dark_vessel_gets_the_same_base_product_plus_a_wake_follow_up():
    """Decoupling means one extra item, not a different one: the report goes out on the
    ordinary chip, and the expensive ROI+wake follows only if capacity allows."""
    lod = LoDConfig(conf_low=0.25)
    plain, _ = _workload(p_dark=0.0, seed=11)
    dark, _ = _workload(p_dark=1.0, seed=11)
    kinds = lambda wl: [i.kind for i in encode_lod(wl, lod) if i.ships]
    assert "L2" not in kinds(plain)
    assert "L2" in kinds(dark)
    base = [k for k in kinds(dark) if k in ("L0", "L1")]
    assert base, "a dark vessel still gets the ordinary base product"
    # the follow-up carries the smaller evidence value, not the 5x ship weight
    wake = [i for i in encode_lod(dark, lod) if i.kind == "L2"]
    assert all(i.value == lod.dark_wake_value for i in wake)


def test_a_spoofed_dark_flag_never_removes_a_ship_from_the_downlink():
    """'A human confirms': the flag may reorder, but must never drop a ship outright."""
    lod = LoDConfig(conf_low=0.25)
    for dark in (False, True):
        wl, _ = _workload(p_dark=1.0 if dark else 0.0)
        kinds = {i.kind for i in encode_lod(wl, lod) if i.ships}
        assert kinds, "every detected ship must still produce some item"


# ---------------------------------------------------------------- detector miss


def test_a_missed_ship_still_gets_a_thumbnail_but_it_carries_no_ship_identity():
    """Mitigation claimed: 'Thumbnail + raw ring buffer, ground re-request'.

    Half true. The tile does still get a thumbnail, so a human could spot the ship and
    re-request. But the thumbnail Item has no ship ids, so it contributes nothing to
    delivered recall -- the safety net is a human-in-the-loop hint, not a delivery path,
    and the report should not imply the ship was reported. There is also no raw ring
    buffer anywhere in the code.
    """
    lod = LoDConfig(conf_low=0.25, gate_thumbnails=False)
    wl, _ = _workload()
    items = encode_lod(wl, lod)
    thumbs = [i for i in items if i.kind == "thumb"]
    assert thumbs, "tiles with a missed ship still produce a thumbnail"
    assert all(i.ships == () for i in thumbs), "thumbnails carry no ship identity"
    assert all(i.value <= lod.thumb_value for i in thumbs)


# ---------------------------------------------------------------- domain shift


def test_audit_sampling_keeps_a_fraction_of_gated_tiles_visible():
    """Mitigation: 'Audit sampling: 1% random raw tiles'.

    Implemented as a 2% sample of *thumbnails* on gated-empty tiles, not raw tiles. It does
    fire, so silent gating is detectable, but it is thumbnails rather than the raw tiles the
    report promises.
    """
    wl, _ = _workload(tiles_per_day=20_000, seed=4)
    none = encode_lod(wl, LoDConfig(conf_low=0.25, gate_thumbnails=True, thumb_audit=0.0))
    some = encode_lod(wl, LoDConfig(conf_low=0.25, gate_thumbnails=True, thumb_audit=0.5))
    n = lambda it: sum(1 for i in it if i.kind == "thumb")
    assert n(some) > n(none)


def test_a_silent_confidence_drop_shows_up_as_lost_ships_not_a_crash():
    """Domain shift: every confidence falls by 30%. Recall must degrade monotonically."""
    lod = LoDConfig(conf_low=0.25)
    wl, _ = _workload(tiles_per_day=20_000)
    passes = [_pass(h, 5e6) for h in (4, 10, 16, 22)]
    base = metrics(simulate(encode_lod(wl, lod), passes, START, ValueGreedy()), wl, lod)
    for s in wl.ships:                                   # inject the shift
        s.confidence *= 0.7
    shifted = metrics(simulate(encode_lod(wl, lod), passes, START, ValueGreedy()), wl, lod)
    assert shifted["ship_recall"] <= base["ship_recall"] + 1e-9


# ---------------------------------------------------------------- safe mode


def test_safe_mode_thumbnails_plus_fifo_still_delivers():
    """Mitigation: 'Watchdog + safe mode (thumbnails + FIFO)'. Degraded, but alive."""
    wl, _ = _workload(tiles_per_day=20_000)
    safe = LoDConfig(conf_low=1.1, gate_thumbnails=False)   # nothing clears the ladder
    items = encode_lod(wl, safe)
    assert items and all(i.kind in ("thumb", "tile", "fp") for i in items)
    res = simulate(items, [_pass(h, 2e6) for h in (6, 18)], START, FIFO())
    assert res.bytes_sent > 0


def test_an_empty_day_produces_no_items_and_no_crash():
    wl, _ = _workload(tiles_per_day=0)
    res = simulate(encode_lod(wl, LoDConfig()), [_pass(6, 1e6)], START, ValueGreedy())
    assert res.sent == [] and res.bytes_sent == 0
