import numpy as np

from sat7.prefilter import CANDIDATES, CLOUD, EMPTY, PrefilterConfig, match_ships, run_prefilter
from sat7.synthetic import make_tile


def test_empty_sea_is_dropped():
    img, boxes = make_tile(n_ships=0, seed=1)
    res = run_prefilter(img)
    assert res.context == EMPTY and not res.boxes and not boxes


def test_ships_are_found():
    img, boxes = make_tile(n_ships=4, seed=2)
    res = run_prefilter(img)
    assert res.context == CANDIDATES
    assert all(match_ships(boxes, res.boxes))


def test_overcast_tile_is_dropped():
    img, _ = make_tile(n_ships=2, cloud_cover=0.97, seed=3)
    assert run_prefilter(img).context == CLOUD


def test_ships_between_clouds_are_kept():
    # regression: a partly cloudy tile used to be dropped with its visible ships
    img, boxes = make_tile(n_ships=4, cloud_cover=0.4, seed=5)
    res = run_prefilter(img)
    assert res.context != CLOUD
    visible = [b for b, f in zip(boxes, match_ships(boxes, res.boxes)) if f]
    assert len(visible) >= 2


def test_touching_ships_are_split():
    img, boxes = make_tile(n_ships=0, seed=4)
    # two ships side by side, touching
    from sat7.synthetic import draw_ship
    draw_ship(img, (380, 380), 28, 8, 0.0)
    draw_ship(img, (380, 392), 28, 8, 0.0)
    res = run_prefilter(img, PrefilterConfig(watershed_min_area=150))
    assert len(res.boxes) >= 2
