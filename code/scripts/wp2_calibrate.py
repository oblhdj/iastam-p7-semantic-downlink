"""WP2: is the detector's confidence a probability? Calibrate it, then re-cut the LoD ladder.

Why this matters here, concretely. The level-of-detail encoder branches on raw confidence:
``conf > 0.9`` sends 40 bytes of metadata, ``0.25..0.9`` sends a ~900 byte chip. WP6 measured
that only **7.5%** of real ships ever clear 0.9, so the cheap rung is effectively dead code --
the ladder was cut for a confidence distribution this detector does not produce.

A threshold is only meaningful if the number it compares against means something. So:

  1. label every prediction TP / FP (IoU >= 0.5 against ground truth, same rule as WP1)
  2. measure calibration as it stands: reliability diagram + ECE, on the test split
  3. fit a mapping on the **val** split -- never on test -- two ways:
       Platt   logistic in logit space, 2 parameters, smooth and monotone
       Isotonic pool-adjacent-violators, non-parametric, monotone
  4. re-measure ECE on test, and report which mapping to use
  5. translate the LoD rungs into calibrated probabilities, so "0.9" means "90% of these
     are really ships" instead of "the network emitted 0.9"

Val predictions come from scripts/wp2_predict_split.py (needs the 3.12 torch env). This
script itself is pure numpy/pandas and runs in .venv.

    .venv/Scripts/python.exe scripts/wp2_calibrate.py

Outputs: results/wp2_calibration.json, wp2_reliability.png, wp2_thresholds.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-6


# ----------------------------------------------------------------- labelling


def gt_boxes(label_path: Path, w: int = 768, h: int = 768) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 4))
    rows = [l.split() for l in label_path.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4))
    a = np.array([[float(v) for v in r[1:5]] for r in rows])
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def label_predictions(preds: pd.DataFrame, labels_dir: Path, iou_thr: float = 0.5) -> pd.DataFrame:
    """Tag each prediction correct / not, greedily by confidence (one GT per prediction)."""
    out = []
    for name, p in preds.groupby("image"):
        gt = gt_boxes(labels_dir / str(name).replace(".jpg", ".txt"))
        p = p.sort_values("conf", ascending=False)
        pb = np.stack([p.cx - p.w / 2, p.cy - p.h / 2, p.cx + p.w / 2, p.cy + p.h / 2], axis=1)
        taken = np.zeros(len(gt), bool)
        correct = np.zeros(len(pb), bool)
        for i in range(len(pb)):
            if not len(gt):
                break
            x1 = np.maximum(pb[i, 0], gt[:, 0]); y1 = np.maximum(pb[i, 1], gt[:, 1])
            x2 = np.minimum(pb[i, 2], gt[:, 2]); y2 = np.minimum(pb[i, 3], gt[:, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
            iou = inter / (area(pb[i]) + area(gt) - inter + EPS)
            iou[taken] = 0.0
            j = int(np.argmax(iou))
            if iou[j] >= iou_thr:
                taken[j] = True
                correct[i] = True
        out.append(pd.DataFrame({"conf": p.conf.to_numpy(), "correct": correct}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["conf", "correct"])


# ----------------------------------------------------------------- calibration metrics


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> tuple[float, pd.DataFrame]:
    """Expected calibration error with equal-count bins, plus the reliability table."""
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order].astype(float)
    edges = np.linspace(0, len(conf), bins + 1).astype(int)
    rows, total = [], len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        c, a = conf[lo:hi], correct[lo:hi]
        rows.append({"n": hi - lo, "mean_conf": float(c.mean()), "accuracy": float(a.mean())})
    t = pd.DataFrame(rows)
    return float((t.n / total * (t.mean_conf - t.accuracy).abs()).sum()), t


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((conf - correct.astype(float)) ** 2))


# ----------------------------------------------------------------- the two mappings


def fit_platt(conf: np.ndarray, correct: np.ndarray, iters: int = 200) -> tuple[float, float]:
    """Logistic a*logit(p)+b by Newton/IRLS. Two parameters, monotone by construction."""
    x = np.log(np.clip(conf, EPS, 1 - EPS) / (1 - np.clip(conf, EPS, 1 - EPS)))
    y = correct.astype(float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * x + b
        p = 1 / (1 + np.exp(-z))
        w = np.clip(p * (1 - p), 1e-9, None)
        g = np.array([np.sum((p - y) * x), np.sum(p - y)])
        h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        try:
            step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
        except np.linalg.LinAlgError:
            break
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(a), float(b)


def apply_platt(conf: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.log(np.clip(conf, EPS, 1 - EPS) / (1 - np.clip(conf, EPS, 1 - EPS)))
    return 1 / (1 + np.exp(-(a * x + b)))


def fit_isotonic(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool adjacent violators: the monotone step function closest to the labels."""
    order = np.argsort(conf)
    x, y = conf[order].astype(float), correct[order].astype(float)
    vals, wts = list(y), [1.0] * len(y)
    i = 0
    while i < len(vals) - 1:                      # merge any decreasing pair, then back up
        if vals[i] > vals[i + 1]:
            w = wts[i] + wts[i + 1]
            v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / w
            vals[i:i + 2] = [v]
            wts[i:i + 2] = [w]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.repeat(vals, [int(round(w)) for w in wts])[:len(x)]
    return x, fitted


def apply_isotonic(conf: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(conf, x, y, left=y[0], right=y[-1])


# ----------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()
    out = args.results

    val_pred = out / "wp2_val_predictions.csv"
    if not val_pred.exists():
        raise SystemExit(f"{val_pred.name} missing -- run wp2_predict_split.py --split val first "
                         "(needs the 3.12 torch env)")

    print("labelling predictions (IoU >= 0.5)...")
    val = label_predictions(pd.read_csv(val_pred), args.data / "labels" / "val")
    test = label_predictions(pd.read_csv(out / "wp1_predictions.csv"), args.data / "labels" / "test")
    print(f"  val  {len(val):6d} predictions, {val.correct.mean():.3f} correct")
    print(f"  test {len(test):6d} predictions, {test.correct.mean():.3f} correct")

    vc, vy = val.conf.to_numpy(), val.correct.to_numpy()
    tc, ty = test.conf.to_numpy(), test.correct.to_numpy()

    # ---- fit on val only
    a, b = fit_platt(vc, vy)
    ix, iy = fit_isotonic(vc, vy)
    mapped = {"raw": tc, "platt": apply_platt(tc, a, b), "isotonic": apply_isotonic(tc, ix, iy)}

    report = {"val_predictions": len(val), "test_predictions": len(test),
              "platt": {"a": a, "b": b}, "fitted_on": "val", "evaluated_on": "test", "test": {}}
    print("\ncalibration on the TEST split (fitted on val):")
    tables = {}
    for name, p in mapped.items():
        e, tbl = ece(p, ty, args.bins)
        tables[name] = tbl
        report["test"][name] = {"ECE": e, "Brier": brier(p, ty), "mean_conf": float(p.mean())}
        print(f"  {name:9s} ECE {e:.4f}   Brier {brier(p, ty):.4f}   mean {p.mean():.3f} "
              f"(actual accuracy {ty.mean():.3f})")

    best = min(report["test"], key=lambda k: report["test"][k]["ECE"])
    report["recommended"] = best
    print(f"\n-> lowest ECE: {best} "
          f"({report['test']['raw']['ECE']:.4f} raw -> {report['test'][best]['ECE']:.4f})")

    # ---- what the LoD rungs actually mean, before and after
    rung_rows = []
    for target in (0.25, 0.4, 0.5, 0.75, 0.9, 0.95):
        row = {"intended_probability": target}
        for name, p in mapped.items():
            # the raw score whose calibrated probability reaches the target
            ok = tc[p >= target]
            row[f"raw_conf_for_{name}"] = float(ok.min()) if len(ok) else float("nan")
            row[f"share_of_preds_{name}"] = float((p >= target).mean())
        rung_rows.append(row)
    rungs = pd.DataFrame(rung_rows)
    rungs.to_csv(out / "wp2_thresholds.csv", index=False)
    print("\nWhat a level-of-detail rung should compare against:")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(rungs[["intended_probability", f"raw_conf_for_{best}", f"share_of_preds_{best}",
                     "share_of_preds_raw"]].to_string(index=False))

    hi = rungs.loc[rungs.intended_probability == 0.9, f"raw_conf_for_{best}"].iloc[0]
    report["lod_rungs"] = {
        "conf_high_now": 0.9,
        "conf_high_calibrated_equivalent": None if np.isnan(hi) else float(hi),
        "share_above_conf_high_now": float((tc > 0.9).mean()),
        "share_above_calibrated_equivalent": float((mapped[best] >= 0.9).mean()),
    }
    (out / "wp2_calibration.json").write_text(json.dumps(report, indent=2))

    # ---- reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    a0 = axes[0]
    a0.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    for name, tbl in tables.items():
        a0.plot(tbl.mean_conf, tbl.accuracy, "o-", label=f"{name} (ECE {report['test'][name]['ECE']:.3f})")
    a0.set(xlabel="confidence the detector reports", ylabel="fraction actually correct",
           title="Reliability on the test split\n(mapping fitted on val)", xlim=(0, 1), ylim=(0, 1))
    a0.legend(fontsize=8, frameon=False)
    a0.grid(alpha=0.25)

    a1 = axes[1]
    a1.hist(tc, bins=40, color="#9CA3AF", label="raw confidence")
    a1.axvline(0.9, color="#C0392B", ls="--", lw=1.5,
               label=f"current L0 rung 0.9 ({100 * (tc > 0.9).mean():.1f}% of preds)")
    if not np.isnan(hi):
        a1.axvline(hi, color="#2E8B57", ls="-", lw=1.5,
                   label=f"raw score meaning P=0.9 ({100 * (mapped[best] >= 0.9).mean():.1f}%)")
    a1.set(xlabel="raw confidence", ylabel="predictions",
           title="Where the rungs sit in the real distribution")
    a1.legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "wp2_reliability.png", dpi=160)
    print(f"\nSaved wp2_calibration.json, wp2_thresholds.csv, wp2_reliability.png in {out}")


if __name__ == "__main__":
    main()
