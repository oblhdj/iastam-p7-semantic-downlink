from datetime import datetime, timedelta, timezone

from sat7.orbit import Pass
from sat7.priority import (P0, P1, P2, P3, PriorityConfig, classify, encode_priority,
                           encode_semantic, level_histogram)
from sat7.scheduler import LoDConfig, Ship, ValueGreedy, Workload, WorkloadConfig, metrics, simulate

START = datetime(2026, 9, 18, tzinfo=timezone.utc)
LOD = LoDConfig(conf_low=0.25)
PC = PriorityConfig()


def _ship(i, conf, dark=False, coastal=False):
    return Ship(i, 0.0, dark, conf, coastal)


def test_classify_matches_table_I():
    assert classify(_ship(0, 0.90), "ships", PC, LOD) == P1          # confident -> metadata only
    assert classify(_ship(1, 0.40), "ships", PC, LOD) == P2          # uncertain -> + ROI
    assert classify(_ship(2, 0.40, coastal=True), "coast", PC, LOD) == P3  # + context on a coast
    assert classify(_ship(3, 0.10), "ships", PC, LOD) == P0          # below detector thr -> discard


def test_dark_vessel_escalates_one_level_capped_at_p3():
    assert classify(_ship(0, 0.90, dark=True), "ships", PC, LOD) == P2     # P1 -> P2
    assert classify(_ship(1, 0.40, dark=True), "ships", PC, LOD) == P3     # P2 -> P3
    assert classify(_ship(2, 0.40, dark=True, coastal=True), "coast", PC, LOD) == P3  # stays capped


def test_p0_ship_is_never_transmitted():
    ships = [_ship(0, 0.90), _ship(1, 0.10)]                          # one kept, one discarded
    wl = Workload(ships, [(0.0, "ships", (0, 1))], WorkloadConfig())
    items = encode_priority(wl, LOD, PC)
    covered = {s for it in items for s in it.ships}
    assert 0 in covered and 1 not in covered


def test_levels_emit_expected_item_counts():
    # P1 ship -> 1 item; P2 ship -> 2; P3 (coast) ship -> metadata + ROI + one tile-wide context
    ships = [_ship(0, 0.90), _ship(1, 0.40), _ship(2, 0.40, coastal=True)]
    wl = Workload(ships, [(0.0, "ships", (0, 1)), (0.0, "coast", (2,))], WorkloadConfig())
    items = encode_priority(wl, LOD, PC)
    kinds = [it.kind for it in items]
    # 3 ships all reported (P1 x3); the two uncertain ones add a ROI (P2 x2); the coastal one adds
    # the tile-wide contextual image (P3 x1)
    assert kinds.count("P1") == 3 and kinds.count("P2") == 2 and kinds.count("P3") == 1


def test_priority_items_run_through_the_scheduler_and_metrics():
    wl = _real_ish_workload()
    items = encode_semantic(wl, mode="priority", lod=LOD, cfg=PC)
    t = START + timedelta(hours=1)
    passes = [Pass(t, t + timedelta(minutes=4), t + timedelta(minutes=8), 45.0, 5e6)]
    m = metrics(simulate(items, passes, START, ValueGreedy(), encoder="P0-P3"), wl, LOD)
    assert 0 < m["ship_recall"] <= 1 and m["MB_sent"] <= m["MB_capacity"] + 1e-9


def test_both_schemes_are_selectable_on_one_workload():
    wl = _real_ish_workload()
    lod_items = encode_semantic(wl, mode="lod", lod=LOD)
    pri_items = encode_semantic(wl, mode="priority", lod=LOD, cfg=PC)
    assert lod_items and pri_items                       # both produce items
    assert {it.kind for it in pri_items} <= {"P1", "P2", "P3"}


def test_histogram_sums_to_detectable_ships():
    wl = _real_ish_workload()
    hist = level_histogram(wl, LOD, PC)
    non_cloud_ships = sum(len(ids) for t, ctx, ids in wl.tiles if ctx != "cloud")
    assert sum(hist.values()) == non_cloud_ships


def _real_ish_workload():
    wl = Workload(
        ships=[_ship(0, 0.9), _ship(1, 0.4), _ship(2, 0.8, dark=True),
               _ship(3, 0.3, coastal=True), _ship(4, 0.1)],
        tiles=[(0.0, "ships", (0, 1, 2)), (0.0, "coast", (3,)),
               (0.0, "ships", (4,)), (0.0, "empty", ())],
        cfg=WorkloadConfig(seed=1))
    return wl
