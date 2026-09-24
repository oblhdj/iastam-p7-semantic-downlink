from datetime import datetime, timezone

import numpy as np

from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite, rate_bps


def test_leo_period_and_passes():
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    sat = make_satellite(OrbitConfig(epoch=start))
    passes = find_passes(sat, GroundStation(), start, hours=24)
    # a 500 km polar orbit sees a mid-latitude station a few times per day
    assert 2 <= len(passes) <= 8
    for p in passes:
        assert 60 < p.duration_s < 15 * 60
        assert p.max_elevation_deg >= 5
        assert p.capacity_bytes > 0


def test_rate_decreases_with_range():
    link = LINK_PRESETS["smallsat_xband"]
    r = rate_bps(np.array([500.0, 1000.0, 2500.0]), 500.0, link)
    assert r[0] >= r[1] > r[2] > 0
