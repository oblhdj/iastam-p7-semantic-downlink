import numpy as np
import pytest

from sat7.datasets import recall_of
from sat7.evaluation import compute_ap, describe_detections, evaluate_detections, match_image


def test_perfect_detections_score_one():
    gts = [(100, 100, 40, 40), (400, 300, 30, 60)]
    dets = [(100, 100, 40, 40, 0.9), (400, 300, 30, 60, 0.8)]
    r = evaluate_detections([(dets, gts)])
    assert (r["TP"], r["FP"], r["FN"]) == (2, 0, 0)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["F1"] == 1.0
    assert r["AP50"] >= 0.99


def test_duplicate_is_a_false_positive_and_miss_is_a_false_negative():
    gts1, dets1 = [(100, 100, 40, 40)], [(100, 100, 40, 40, 0.9), (102, 100, 40, 40, 0.6)]
    gts2, dets2 = [(300, 300, 20, 20)], [(600, 600, 20, 20, 0.8)]
    r = evaluate_detections([(dets1, gts1), (dets2, gts2)])
    assert (r["TP"], r["FP"], r["FN"]) == (1, 2, 1)
    assert r["precision"] == round(1 / 3, 4) and r["recall"] == 0.5 and r["F1"] == 0.4
    assert r["AP50"] == pytest.approx(0.58, abs=0.01)       # hand-computed 101-point envelope area


def test_below_threshold_boxes_feed_ap_but_not_p_r():
    gts = [(100, 100, 40, 40)]
    low = [(100, 100, 40, 40, 0.1)]
    r = evaluate_detections([(low, gts)], conf_thr=0.25)
    assert (r["TP"], r["FP"], r["FN"]) == (0, 0, 1) and r["recall"] == 0.0
    assert r["precision"] is None                       # nothing at or above the cut
    assert r["AP50"] >= 0.99                           # but the box is on the PR curve


def test_matching_is_confidence_ordered():
    # the confident box takes the ship even though the weaker one overlaps it better
    gts = [(100, 100, 40, 40)]
    dets = [(104, 100, 40, 40, 0.9), (100, 100, 40, 40, 0.3)]
    tp, gt_conf = match_image(dets, gts)
    assert tp == [True, False] and gt_conf == [0.9]


def test_recall_agrees_with_the_wp1_rule_after_thresholding():
    rng = np.random.default_rng(0)
    images = []
    for _ in range(40):
        gts = [tuple(rng.uniform([50, 50, 10, 10], [700, 700, 80, 80])) for _ in range(rng.integers(0, 6))]
        dets = [(g[0] + rng.normal(0, 4), g[1] + rng.normal(0, 4), g[2], g[3], float(rng.uniform(0.05, 1)))
                for g in gts if rng.random() < 0.8]
        dets += [tuple(rng.uniform([50, 50, 10, 10], [700, 700, 80, 80])) + (float(rng.uniform(0.05, 1)),)
                 for _ in range(rng.integers(0, 3))]
        images.append((dets, gts))
    r = evaluate_detections(images, conf_thr=0.25)
    found = sum((recall_of([d for d in dets if d[4] >= 0.25], gts, iou_thr=0.5) for dets, gts in images), [])
    assert r["recall"] == round(float(np.mean(found)), 4)


def test_size_buckets_split_recall():
    gts = [(100, 100, 10, 10), (300, 300, 50, 40), (600, 600, 150, 60)]
    dets = [(300, 300, 50, 40, 0.9), (600, 600, 150, 60, 0.9)]
    by = evaluate_detections([(dets, gts)])["recall_by_size"]
    assert by["small_<32"]["recall"] == 0.0 and by["medium_32-96"]["recall"] == 1.0
    assert by["large_>96"]["recall"] == 1.0


def test_compute_ap_matches_ultralytics_definition_on_a_step():
    # recall reaches 0.5 at precision 1, then nothing: area = 0.5 (+ the interpolated tail to 0)
    ap = compute_ap(np.array([0.5]), np.array([1.0]))
    assert 0.5 <= ap <= 0.76


def test_no_ground_truth_reports_counts_not_accuracy():
    d = describe_detections([(1, 1, 5, 5, 0.9), (2, 2, 5, 5, 0.1)], conf_thr=0.25)
    assert d["ground_truth"] is False and "NOT accuracy" in d["note"]
    assert d["n_detections"] == 2 and d["n_at_or_above_thr"] == 1
    assert "precision" not in d and "recall" not in d
