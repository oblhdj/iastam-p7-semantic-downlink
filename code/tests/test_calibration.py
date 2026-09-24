import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from wp2_calibrate import (apply_isotonic, apply_platt, brier, ece, fit_isotonic,  # noqa: E402
                           fit_platt, label_predictions)


def test_pav_matches_the_textbook_answer():
    """y = [1,0,1,1] pools the decreasing pair into 0.5, 0.5."""
    x, y = np.array([1., 2., 3., 4.]), np.array([1., 0., 1., 1.])
    _, fitted = fit_isotonic(x, y)
    assert np.allclose(fitted, [0.5, 0.5, 1.0, 1.0])


def test_isotonic_output_never_decreases():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 2000)
    labels = (rng.uniform(0, 1, 2000) < conf).astype(float)
    _, fitted = fit_isotonic(conf, labels)
    assert np.all(np.diff(fitted) >= -1e-9)


def test_already_calibrated_scores_are_left_alone():
    """If confidence is already the true probability, calibration must not make it worse."""
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.05, 0.95, 20000)
    labels = (rng.uniform(0, 1, 20000) < conf).astype(float)
    raw, _ = ece(conf, labels)
    a, b = fit_platt(conf, labels)
    fitted, _ = ece(apply_platt(conf, a, b), labels)
    assert raw < 0.02 and fitted < 0.02


def test_both_calibrators_fix_a_known_miscalibration():
    """Ground truth is p = conf^2, so raw confidence is badly overconfident."""
    rng = np.random.default_rng(2)
    conf = rng.uniform(0.05, 0.95, 40000)
    labels = (rng.uniform(0, 1, 40000) < conf ** 2).astype(float)
    raw, _ = ece(conf, labels)
    platt, _ = ece(apply_platt(conf, *fit_platt(conf, labels)), labels)
    iso, _ = ece(apply_isotonic(conf, *fit_isotonic(conf, labels)), labels)
    assert raw > 0.1                       # the distortion is real
    assert platt < raw / 3 and iso < raw / 3
    assert brier(conf, labels) > brier(apply_platt(conf, *fit_platt(conf, labels)), labels)


def test_platt_is_monotone():
    a, b = 1.3, -0.8
    p = apply_platt(np.linspace(0.01, 0.99, 50), a, b)
    assert np.all(np.diff(p) > 0)


def test_ece_is_zero_for_a_perfect_forecaster_and_near_one_for_the_worst():
    """Equal-count bins mean one bin straddles the 0/1 boundary and holds both classes,
    so a perfectly wrong forecaster scores ~0.93 rather than exactly 1."""
    conf = np.concatenate([np.zeros(500), np.ones(500)])
    labels = np.concatenate([np.zeros(500), np.ones(500)])
    assert ece(conf, labels)[0] < 1e-9
    assert ece(1 - conf, labels)[0] > 0.9


def test_prediction_labelling_assigns_each_ground_truth_once(tmp_path):
    """Two predictions on one ship: the better one is the true positive, the other is not."""
    import pandas as pd
    (tmp_path / "t.txt").write_text("0 0.5 0.5 0.1 0.1\n")       # one ship, 76.8 px box
    preds = pd.DataFrame({
        "image": ["t.jpg", "t.jpg"],
        "cx": [384.0, 390.0], "cy": [384.0, 384.0],
        "w": [76.8, 76.8], "h": [76.8, 76.8], "conf": [0.9, 0.8],
    })
    out = label_predictions(preds, tmp_path)
    assert out.correct.sum() == 1
    assert bool(out.sort_values("conf", ascending=False).correct.iloc[0])   # the confident one
