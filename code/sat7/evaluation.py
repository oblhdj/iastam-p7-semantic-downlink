"""Detection evaluation: precision / recall / F1 / AP when ground truth exists, counts otherwise.

The repo scored B1/B2 by recall only (`wp12_seams.recall_of`, `datasets.recall_of`) and on small
samples (wp18: 262 / 47 ships -- audit G1/B8). This module adds the rest of the standard detection
metrics on the same box convention (cx, cy, w, h, conf), so B1, B2 and the SAHI ablations can be
compared on precision as well as recall -- which is where "without fusion" shows up at all (it
keeps duplicates: recall is unchanged, precision drops).

Matching (per image): detections in decreasing confidence; each takes the unmatched ground-truth
box it overlaps most, if IoU >= iou_thr (0.5, the WP1 threshold) -> TP, else FP. This is the
VOC/COCO convention; because it is confidence-ordered, the matches of the detections above any
threshold are exactly what you would get by running at that threshold, so P/R/F1 at the onboard
cut and the AP curve come from one matching. (WP1's `recall_of` iterates ground truth instead; the
two agree on recall to within ties -- checked in tests and in wp24.)

AP: ultralytics' `compute_ap` (precision envelope, 101-point interpolation), so AP50 here is
comparable with wp1's mAP50 0.804 -- except that predictions dumped at conf >= 0.05 (not 0.001)
truncate the low-confidence tail, which can only lower AP. Single class (ship): mAP = AP.

Without ground truth there is nothing to be accurate against: `describe_detections` reports counts
and confidences only and says so. Labels: these functions emit no numbers of their own; results
are REAL when the detections are the trained detector's on real imagery with real labels.
"""
from __future__ import annotations

import numpy as np

from .b2_sahi_fusion import _iou

IOU_THRS_COCO = tuple(np.round(np.arange(0.50, 0.96, 0.05), 2))   # 0.50:0.05:0.95
SIZE_BUCKETS = (("small_<32", 0, 32), ("medium_32-96", 32, 96 + 1e-9), ("large_>96", 96 + 1e-9, np.inf))


def match_pairs(dets, gts, iou_thr: float = 0.5) -> dict[int, int]:
    """Confidence-ordered greedy matching for one image: {detection index: ground-truth index}.

    Detections in decreasing confidence each take the unmatched ground truth they overlap most,
    if IoU >= iou_thr (VOC/COCO convention)."""
    pairs, taken = {}, [False] * len(gts)
    for i in sorted(range(len(dets)), key=lambda k: -float(dets[k][4])):
        best, bj = iou_thr, -1
        for j, g in enumerate(gts):
            if taken[j]:
                continue
            v = _iou(dets[i][:4], g[:4])
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            taken[bj] = True
            pairs[i] = bj
    return pairs


def match_image(dets, gts, iou_thr: float = 0.5):
    """Returns (tp, gt_conf): tp[i] is True when detection i (input order) is a true positive;
    gt_conf[j] is the confidence of the detection that matched ground truth j (0.0 if none)."""
    pairs = match_pairs(dets, gts, iou_thr)
    tp = [i in pairs for i in range(len(dets))]
    gt_conf = [0.0] * len(gts)
    for i, j in pairs.items():
        gt_conf[j] = float(dets[i][4])
    return tp, gt_conf


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision envelope, 101-point interpolation (ultralytics compute_ap)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def _ap_at(images, iou_thr: float) -> float:
    confs, tps, n_gt = [], [], 0
    for dets, gts in images:
        tp, _ = match_image(dets, gts, iou_thr)
        confs += [float(d[4]) for d in dets]
        tps += tp
        n_gt += len(gts)
    if n_gt == 0:
        return float("nan")
    if not confs:
        return 0.0
    order = np.argsort(-np.asarray(confs), kind="stable")
    tp = np.asarray(tps, float)[order]
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    return compute_ap(tpc / (n_gt + 1e-16), tpc / (tpc + fpc))


def evaluate_detections(images, conf_thr: float = 0.25, iou_thr: float = 0.5,
                        ap5095: bool = True) -> dict:
    """Score detections against ground truth over a set of images.

    `images` is an iterable of (detections, ground_truths) pairs, both lists of centre-form boxes
    (detections carry conf as the 5th element). Metrics at `conf_thr` (the onboard cut) plus AP
    over every detection supplied (so pass the raw dump, e.g. conf >= 0.05, to get a useful curve).
    """
    images = [(list(d), list(g)) for d, g in images]
    tp_n = fp_n = 0
    gt_sizes, gt_hit = [], []
    n_det_all = 0
    for dets, gts in images:
        tp, gt_conf = match_image(dets, gts, iou_thr)
        n_det_all += len(dets)
        for d, t in zip(dets, tp):
            if float(d[4]) >= conf_thr:
                tp_n += t
                fp_n += not t
        gt_sizes += [max(float(g[2]), float(g[3])) for g in gts]
        gt_hit += [c >= conf_thr and c > 0 for c in gt_conf]
    n_gt = len(gt_hit)
    fn_n = n_gt - int(sum(gt_hit))
    p = tp_n / (tp_n + fp_n) if tp_n + fp_n else float("nan")
    r = sum(gt_hit) / n_gt if n_gt else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) and not np.isnan(p) and not np.isnan(r) else float("nan")
    hit, sizes = np.asarray(gt_hit, bool), np.asarray(gt_sizes, float)
    by_size = {}
    for name, lo, hi in SIZE_BUCKETS:
        m = (sizes >= lo) & (sizes < hi)
        by_size[name] = {"n_gt": int(m.sum()),
                         "recall": round(float(hit[m].mean()), 4) if m.any() else None}
    out = {"images": len(images), "n_gt": n_gt, "n_detections_supplied": n_det_all,
           "conf_thr": conf_thr, "iou_thr": iou_thr,
           "TP": int(tp_n), "FP": int(fp_n), "FN": int(fn_n),
           "precision": round(p, 4) if not np.isnan(p) else None,
           "recall": round(r, 4) if not np.isnan(r) else None,
           "F1": round(f1, 4) if not np.isnan(f1) else None,
           "recall_by_size": by_size,
           f"AP{int(round(iou_thr * 100))}": round(_ap_at(images, iou_thr), 4)}
    if ap5095:
        aps = [_ap_at(images, t) for t in IOU_THRS_COCO]
        out["AP50-95"] = round(float(np.mean(aps)), 4)
    return out


def describe_detections(dets, conf_thr: float = 0.25) -> dict:
    """No ground truth: counts and confidence statistics only -- explicitly NOT accuracy."""
    c = np.asarray([float(d[4]) for d in dets], float)
    kept = c[c >= conf_thr]

    def stats(a):
        if not a.size:
            return None
        return {"min": round(float(a.min()), 4), "median": round(float(np.median(a)), 4),
                "mean": round(float(a.mean()), 4), "max": round(float(a.max()), 4)}
    return {"ground_truth": False,
            "note": "no ground truth: counts and confidences only -- these are NOT accuracy metrics",
            "n_detections": int(c.size), "conf_thr": conf_thr, "n_at_or_above_thr": int(kept.size),
            "confidence_all": stats(c), f"confidence_at_or_above_{conf_thr}": stats(kept)}


if __name__ == "__main__":
    # two images: a perfect hit + a duplicate (FP), and a miss + a false alarm
    gts1 = [(100, 100, 40, 40)]
    dets1 = [(100, 100, 40, 40, 0.9), (102, 100, 40, 40, 0.6)]
    gts2 = [(300, 300, 20, 20)]
    dets2 = [(600, 600, 20, 20, 0.8)]
    r = evaluate_detections([(dets1, gts1), (dets2, gts2)], conf_thr=0.25)
    assert (r["TP"], r["FP"], r["FN"]) == (1, 2, 1), r
    assert r["precision"] == round(1 / 3, 4) and r["recall"] == 0.5
    print("P/R/F1:", r["precision"], r["recall"], r["F1"], "AP50", r["AP50"])
    perfect = evaluate_detections([(dets1[:1], gts1)])
    assert perfect["AP50"] > 0.99 and perfect["F1"] == 1.0
    d = describe_detections(dets1 + dets2)
    assert d["ground_truth"] is False and d["n_at_or_above_thr"] == 3
    print("evaluation self-check OK")
