import numpy as np
import pandas as pd
import pytest

from sat7.real_workload import CONTEXTS, RealWorkloadConfig, workload_from_catalogue
from sat7.scheduler import (FIFO, LoDConfig, SizeModel, WorkloadConfig, encode_lod,
                            generate_workload, simulate)


def _catalogue():
    """Four hand-made tiles, one per context, with known ships and false alarms."""
    tiles = pd.DataFrame({
        "image": ["cloudy.jpg", "shippy.jpg", "coasty.jpg", "empty.jpg", "quiet.jpg"],
        "ctx3": ["cloud", "ships", "coast", "empty", "empty"],
        # what the classic pre-filter thought: only "quiet" is cleared as empty sea
        "context": ["cloud", "candidates", "land_coast", "candidates", "empty_sea"],
        "n_ships": [1, 2, 1, 0, 0],
        "n_fp_0.05": [0, 3, 1, 2, 0],
        "n_fp_0.25": [0, 1, 0, 1, 0],
    })
    ships = pd.DataFrame({
        "image": ["cloudy.jpg", "shippy.jpg", "shippy.jpg", "coasty.jpg"],
        "size_px": [30.0, 20.0, 120.0, 55.0],
        "found": [True, True, True, False],
        "conf": [0.8, 0.9, 0.3, 0.0],
    })
    return tiles, ships


def test_real_ships_keep_their_measured_confidence_and_length():
    tiles, ships = _catalogue()
    wl, st = workload_from_catalogue(tiles, ships, RealWorkloadConfig(tiles_per_day=400, seed=1))
    assert wl.source.startswith("real")
    assert st.ships == len(wl.ships) > 0
    assert {s.confidence for s in wl.ships} <= set(ships.conf)
    assert {s.size_px for s in wl.ships} <= set(ships.size_px)
    assert st.detected == sum(s.confidence >= 0.25 for s in wl.ships)


def test_ships_under_cloud_stay_in_the_denominator_but_are_never_sent():
    """The cloud gate drops the tile; the ships on it must still count as missed."""
    tiles, ships = _catalogue()
    cfg = RealWorkloadConfig(tiles_per_day=400, mix="orbit", orbit_mix=(1.0, 0, 0, 0), seed=2)
    wl, st = workload_from_catalogue(tiles, ships, cfg)
    assert st.context_counts["cloud"] == st.tiles
    assert st.ships > 0                                  # ships on cloud tiles are kept
    assert encode_lod(wl) == []                          # ... and nothing is ever encoded


def test_false_alarms_come_from_the_tile_they_were_measured_on():
    tiles, ships = _catalogue()
    cfg = RealWorkloadConfig(tiles_per_day=400, mix="orbit", orbit_mix=(0, 0, 0, 1.0),
                             det_thr=0.25, seed=3)
    wl, st = workload_from_catalogue(tiles, ships, cfg)
    # the empty pool is empty.jpg (1 false alarm at conf>=0.25) and quiet.jpg (none), so the
    # total must land strictly between "none of them" and "all of them"
    assert 0 < st.false_alarms < st.tiles
    assert st.false_alarms == sum(wl.false_alarms)
    items = encode_lod(wl)
    assert sum(i.kind == "fp" for i in items) == st.false_alarms
    assert all(i.value == 0 for i in items if i.kind == "fp")


def test_lower_threshold_never_finds_fewer_false_alarms():
    tiles, ships = _catalogue()
    counts = []
    for thr in (0.05, 0.25):
        _, st = workload_from_catalogue(
            tiles, ships, RealWorkloadConfig(tiles_per_day=800, det_thr=thr, seed=4))
        counts.append(st.false_alarms)
    assert counts[0] >= counts[1]


def test_size_model_charges_a_big_ship_more_than_a_small_one():
    sm = SizeModel()
    assert sm.l1(20) < sm.l1(200)
    assert sm.l2(20) < sm.l2(200)
    tiles, ships = _catalogue()
    cfg = RealWorkloadConfig(tiles_per_day=2000, mix="orbit", orbit_mix=(0, 1.0, 0, 0), seed=5)
    wl, _ = workload_from_catalogue(tiles, ships, cfg)
    flat = sum(i.size for i in encode_lod(wl, LoDConfig(conf_low=0.25)))
    sized = sum(i.size for i in encode_lod(wl, LoDConfig(conf_low=0.25, size_model=sm)))
    assert flat != sized                                 # real lengths change the byte bill


def test_dataset_mix_uses_every_tile_and_orbit_mix_follows_its_weights():
    tiles, ships = _catalogue()
    _, ds = workload_from_catalogue(tiles, ships,
                                    RealWorkloadConfig(tiles_per_day=4000, mix="dataset", seed=6))
    assert all(ds.context_counts[c] > 0 for c in CONTEXTS)
    _, orb = workload_from_catalogue(
        tiles, ships, RealWorkloadConfig(tiles_per_day=4000, mix="orbit",
                                         orbit_mix=(0.0, 0.5, 0.0, 0.5), seed=6))
    assert orb.context_counts["cloud"] == orb.context_counts["coast"] == 0
    assert abs(orb.context_counts["ships"] / orb.tiles - 0.5) < 0.05


def test_thumbnail_gating_only_skips_tiles_both_opinions_call_empty():
    """quiet.jpg: pre-filter sees no candidate and the detector fired nothing -> skippable.
    empty.jpg has no ships either, but the pre-filter still flags candidates -> keep it."""
    tiles, ships = _catalogue()
    cfg = RealWorkloadConfig(tiles_per_day=4000, mix="orbit", orbit_mix=(0, 0, 0, 1.0),
                             det_thr=0.25, seed=7)
    wl, st = workload_from_catalogue(tiles, ships, cfg)
    assert 0 < sum(wl.gate_empty) < st.tiles          # only the "quiet" half is gated

    ungated = encode_lod(wl, LoDConfig(conf_low=0.25, gate_thumbnails=False))
    gated = encode_lod(wl, LoDConfig(conf_low=0.25, gate_thumbnails=True, thumb_audit=0.0))
    n = lambda items: sum(i.kind == "thumb" for i in items)
    assert n(ungated) == st.tiles                      # every non-cloud tile gets one
    assert n(gated) == st.tiles - sum(wl.gate_empty)   # ... minus exactly the gated ones


def test_thumbnail_audit_sample_keeps_some_gated_tiles():
    """Gating must never go fully silent: an audit fraction still goes down."""
    tiles, ships = _catalogue()
    cfg = RealWorkloadConfig(tiles_per_day=8000, mix="orbit", orbit_mix=(0, 0, 0, 1.0), seed=8)
    wl, _ = workload_from_catalogue(tiles, ships, cfg)
    none = encode_lod(wl, LoDConfig(conf_low=0.25, gate_thumbnails=True, thumb_audit=0.0))
    some = encode_lod(wl, LoDConfig(conf_low=0.25, gate_thumbnails=True, thumb_audit=0.5))
    assert sum(i.kind == "thumb" for i in some) > sum(i.kind == "thumb" for i in none)


def test_coastal_mosaic_is_cheaper_but_drops_undetected_ships():
    """The mosaic only describes ships the detector saw; the whole tile carries them all."""
    tiles, ships = _catalogue()
    cfg = RealWorkloadConfig(tiles_per_day=2000, mix="orbit", orbit_mix=(0, 0, 1.0, 0),
                             det_thr=0.25, seed=9)
    wl, _ = workload_from_catalogue(tiles, ships, cfg)
    whole = encode_lod(wl, LoDConfig(conf_low=0.25, coast_mode="tile"))
    mosaic = encode_lod(wl, LoDConfig(conf_low=0.25, coast_mode="mosaic"))
    assert sum(i.size for i in mosaic) < sum(i.size for i in whole)
    # coasty.jpg's only ship was never found (conf 0.0), so the mosaic covers no ship at all
    covered = lambda items: {s for i in items for s in i.ships}
    assert covered(whole) and not covered(mosaic)


def test_adaptive_coast_escalates_only_where_the_two_stages_disagree():
    """Whole tile where the classic stage saw objects the network did not confirm; the
    cheap mosaic everywhere else."""
    tiles, ships = _catalogue()
    tiles = tiles.assign(n_candidates=[0, 0, 30, 0, 0])   # only coasty.jpg looks suspicious
    cfg = RealWorkloadConfig(tiles_per_day=2000, mix="orbit", orbit_mix=(0, 0, 1.0, 0),
                             det_thr=0.25, seed=10)
    wl, _ = workload_from_catalogue(tiles, ships, cfg)
    assert all(u > 0 for u in wl.unconfirmed)

    hot = encode_lod(wl, LoDConfig(conf_low=0.25, coast_mode="adaptive", coast_escalate=5))
    cold = encode_lod(wl, LoDConfig(conf_low=0.25, coast_mode="adaptive", coast_escalate=999))
    assert all(i.kind != "mosaic" for i in hot if i.kind in ("tile", "mosaic"))
    assert all(i.kind != "tile" for i in cold if i.kind in ("tile", "mosaic"))
    assert sum(i.size for i in cold) < sum(i.size for i in hot)


def test_unconfirmed_count_never_goes_negative():
    tiles, ships = _catalogue()
    tiles = tiles.assign(n_candidates=[0, 0, 0, 0, 0])    # fewer candidates than detections
    wl, _ = workload_from_catalogue(tiles, ships,
                                    RealWorkloadConfig(tiles_per_day=2000, seed=11))
    assert all(u >= 0 for u in wl.unconfirmed)


def test_rejects_an_unknown_mix():
    tiles, ships = _catalogue()
    with pytest.raises(ValueError):
        workload_from_catalogue(tiles, ships, RealWorkloadConfig(mix="whatever"))


def test_synthetic_workload_is_unchanged_by_the_real_workload_support():
    """The WP5 numbers must not move: false alarms there are still drawn, not measured."""
    wl = generate_workload(WorkloadConfig(tiles_per_day=2000, seed=3))
    assert wl.false_alarms is None and wl.source == "synthetic"
    items = encode_lod(wl)
    assert sum(i.kind == "fp" for i in items) > 0
    assert all(i.size == LoDConfig().l1_bytes for i in items if i.kind == "fp")
