from datetime import datetime, timedelta, timezone

import pandas as pd

from sat7.orbit import Pass
from sat7.real_workload import RealWorkloadConfig, workload_from_catalogue
from sat7.scheduler import (FIFO, LoDConfig, NoBuffer, ValueGreedy, encode_lod, escalation_for_pressure,
                            metrics, simulate, simulate_online)

START = datetime(2026, 9, 18, tzinfo=timezone.utc)


def _pass(at_h: float, cap: float) -> Pass:
    t = START + timedelta(hours=at_h)
    return Pass(t, t + timedelta(minutes=4), t + timedelta(minutes=8), 45.0, cap)


def _workload(tiles_per_day: int = 4000, seed: int = 0):
    tiles = pd.DataFrame({
        "image": ["coast_a.jpg", "coast_b.jpg", "ships.jpg", "empty.jpg"],
        "ctx3": ["coast", "coast", "ships", "empty"],
        "context": ["land_coast", "land_coast", "candidates", "empty_sea"],
        "n_candidates": [30, 2, 4, 0],
        "n_ships": [2, 1, 2, 0],
        "n_fp_0.25": [0, 0, 1, 0],
    })
    ships = pd.DataFrame({
        "image": ["coast_a.jpg", "coast_a.jpg", "coast_b.jpg", "ships.jpg", "ships.jpg"],
        "size_px": [40.0, 90.0, 25.0, 30.0, 150.0],
        "found": [True, False, True, True, True],
        "conf": [0.7, 0.0, 0.5, 0.8, 0.95],
    })
    return workload_from_catalogue(
        tiles, ships, RealWorkloadConfig(tiles_per_day=tiles_per_day, det_thr=0.25, seed=seed))


# ------------------------------------------------------------------ the control law


def test_pressure_maps_to_a_threshold_between_the_bounds():
    lod = LoDConfig(esc_min=0, esc_max=40, pressure_lo=0.9, pressure_hi=3.5)
    assert escalation_for_pressure(lod, 0.0) == lod.esc_min
    assert escalation_for_pressure(lod, 0.9) == lod.esc_min      # at the knee, still generous
    assert escalation_for_pressure(lod, 99.0) == lod.esc_max
    mid = escalation_for_pressure(lod, 2.2)
    assert lod.esc_min < mid < lod.esc_max


def test_threshold_never_falls_as_the_buffer_fills():
    lod = LoDConfig()
    seq = [escalation_for_pressure(lod, p) for p in (0, 0.5, 1.0, 1.6, 2.5, 3.5, 10.0)]
    assert seq == sorted(seq)


def test_degenerate_bounds_do_not_divide_by_zero():
    lod = LoDConfig(pressure_lo=1.0, pressure_hi=1.0)
    assert escalation_for_pressure(lod, 5.0) == lod.esc_min


# ------------------------------------------------------------------ the online loop


def test_online_matches_the_batch_path_when_the_policy_is_fixed():
    """With a fixed coastal policy, encoding online must change nothing at all --
    otherwise the WP8 comparison would be measuring the refactor, not the idea."""
    wl, _ = _workload()
    passes = [_pass(h, 3e6) for h in (3, 9, 15, 21)]
    lod = LoDConfig(conf_low=0.25, coast_mode="tile")
    batch = simulate(encode_lod(wl, lod), passes, START, ValueGreedy(), encoder="x")
    online, _ = simulate_online(wl, lod, passes, START, ValueGreedy(), encoder="x")
    assert online.bytes_offered == batch.bytes_offered
    assert abs(online.bytes_sent - batch.bytes_sent) < 1e-6
    assert metrics(online, wl, lod)["ship_recall"] == metrics(batch, wl, lod)["ship_recall"]


def test_online_matches_the_batch_path_for_a_non_buffering_policy():
    """Regression: NoBuffer discards its *leftovers*, not the data captured since the last
    pass. Clearing the queue on entry to a window made the online path deliver nothing."""
    wl, _ = _workload(seed=3)
    passes = [_pass(h, 2e6) for h in (6, 12, 18)]
    lod = LoDConfig(conf_low=0.25, coast_mode="tile")
    batch = simulate(encode_lod(wl, lod), passes, START, NoBuffer())
    online, _ = simulate_online(wl, lod, passes, START, NoBuffer())
    assert batch.bytes_sent > 0
    assert abs(online.bytes_sent - batch.bytes_sent) < 1e-6


def test_online_matches_the_batch_path_for_fifo_too():
    wl, _ = _workload(seed=2)
    passes = [_pass(h, 2e6) for h in (4, 12, 20)]
    lod = LoDConfig(conf_low=0.25, coast_mode="adaptive", coast_escalate=5)
    batch = simulate(encode_lod(wl, lod), passes, START, FIFO())
    online, _ = simulate_online(wl, lod, passes, START, FIFO())
    assert abs(online.bytes_sent - batch.bytes_sent) < 1e-6


def test_a_starved_link_pushes_the_encoder_to_cheap_coastal_products():
    """Same workload, same code, only the link changes: a tiny downlink must make the
    encoder stop spending 32 kB on coastal tiles."""
    wl, _ = _workload(tiles_per_day=40_000)
    lod = LoDConfig(conf_low=0.25, coast_mode="queue")
    roomy = [_pass(h, 5e8) for h in (2, 6, 10, 14, 18, 22)]
    starved = [_pass(h, 5e5) for h in (2, 6, 10, 14, 18, 22)]
    r_res, r_tr = simulate_online(wl, lod, roomy, START, ValueGreedy())
    s_res, s_tr = simulate_online(wl, lod, starved, START, ValueGreedy())
    assert r_res.bytes_offered > s_res.bytes_offered      # cheaper products under pressure
    # compare the typical decision, not the tail: once the last pass is behind us both
    # links look equally hopeless, and that is correct behaviour rather than a difference
    median = lambda xs: sorted(xs)[len(xs) // 2]
    assert median(r_tr["escalate"]) < median(s_tr["escalate"])


def test_queue_mode_records_a_trace_only_for_coastal_tiles():
    wl, _ = _workload()
    passes = [_pass(h, 4e6) for h in (6, 18)]
    _, tr = simulate_online(wl, LoDConfig(conf_low=0.25, coast_mode="queue"), passes, START,
                            ValueGreedy())
    n_coast = sum(1 for _, ctx, _ in wl.tiles if ctx == "coast")
    assert len(tr["escalate"]) == n_coast == len(tr["pressure"])
    assert all(p >= 0 for p in tr["pressure"])


def test_fixed_modes_leave_the_trace_empty():
    wl, _ = _workload()
    passes = [_pass(h, 4e6) for h in (6, 18)]
    _, tr = simulate_online(wl, LoDConfig(conf_low=0.25, coast_mode="tile"), passes, START,
                            ValueGreedy())
    assert tr["escalate"] == []
