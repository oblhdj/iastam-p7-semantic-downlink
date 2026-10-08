from datetime import datetime, timezone

import pytest

from sat7.orbit import (GroundStation, OrbitConfig, STATION_PRESETS, find_passes,
                        find_passes_multi, make_satellite, max_gap_h)

START = datetime(2026, 9, 18, tzinfo=timezone.utc)
HOURS = 12.0


@pytest.fixture(scope="module")
def sat():
    return make_satellite(OrbitConfig(epoch=START))


def test_single_station_passes_are_tagged(sat):
    passes = find_passes(sat, GroundStation(), START, HOURS)
    assert passes and all(p.station == GroundStation().name for p in passes)


def test_multi_station_adds_contacts_and_shortens_the_gap(sat):
    sfax = STATION_PRESETS["sfax"]
    net = [sfax, STATION_PRESETS["svalbard"]]                 # add a near-polar station
    single = find_passes(sat, sfax, START, HOURS)
    multi = find_passes_multi(sat, net, START, HOURS)
    assert len(multi) > len(single)                           # the network sees more passes
    assert multi == sorted(multi, key=lambda p: p.rise)       # one merged chronological timeline
    assert {p.station for p in multi} == {s.name for s in net}
    # the latency-dominating gap must not grow, and with a polar station it shrinks
    assert max_gap_h(multi, START, HOURS) <= max_gap_h(single, START, HOURS) + 1e-9


def test_max_gap_handles_empty():
    assert max_gap_h([], START, HOURS) == HOURS
