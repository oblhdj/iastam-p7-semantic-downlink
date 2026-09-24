"""WP6 step 2: how many bytes does one ship cost? Fit bytes = a * (ship px)^b.

WP4 measured the size of 551 real crops per product, and reported the *median*.
But the p90 is 3-6x the median: a 300 px cargo ship is not a 12 px fishing boat.
The scheduler ranks items by value / bytes, so using one median for every ship
hides exactly the trade-off we claim to exploit.

Here we fit a power law per product on the WP4 measurements (log-log OLS) and
keep the residual spread, so every real ship in the catalogue can be given its
own payload cost from its own measured length.

Output: results/wp6_size_model.json  (a, b, R^2, residual sigma in log space)

    python scripts/wp6_fit_size_model.py
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

# products used by the level-of-detail encoder
WANTED = {
    "L1": ("crop tight q40", 40),      # small chip   (L1)
    "L2": ("crop wake q80", 80),       # ROI + wake   (L2)
    "context": ("crop context q80", 80),
}


def fit(g: pd.DataFrame) -> dict:
    x, y = np.log(g.ship_px.to_numpy()), np.log(g.bytes.to_numpy())
    b, loga = np.polyfit(x, y, 1)
    pred = loga + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "a": float(np.exp(loga)), "b": float(b),
        "r2": 1 - ss_res / ss_tot,
        "sigma_log": float(np.sqrt(ss_res / (len(x) - 2))),
        "n": int(len(x)),
        "median_bytes": float(g.bytes.median()),
        "px_range": [float(g.ship_px.min()), float(g.ship_px.max())],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=Path, default=ROOT / "results" / "wp4_lod_sizes.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    d = pd.read_csv(args.sizes)
    d = d[d.ship_px.notna() & (d.bytes > 0)]
    model = {}
    for level, (product, quality) in WANTED.items():
        g = d[(d["product"] == product) & (d.quality == quality)]
        if g.empty:
            raise SystemExit(f"no WP4 rows for {product} q{quality}")
        model[level] = fit(g) | {"product": product, "quality": quality}

    (args.out / "wp6_size_model.json").write_text(json.dumps(model, indent=2))
    for level, m in model.items():
        print(f"{level:8s} {m['product']:18s} bytes = {m['a']:.1f} * px^{m['b']:.2f}   "
              f"R2={m['r2']:.3f}  n={m['n']}  median={m['median_bytes']:.0f} B")

    fig, axes = plt.subplots(1, len(model), figsize=(4 * len(model), 3.6), sharey=True)
    for ax, (level, m) in zip(np.atleast_1d(axes), model.items()):
        g = d[(d["product"] == m["product"]) & (d.quality == m["quality"])]
        ax.scatter(g.ship_px, g.bytes, s=6, alpha=0.3, color="#1B6CA8")
        px = np.linspace(g.ship_px.min(), g.ship_px.max(), 100)
        ax.plot(px, m["a"] * px ** m["b"], color="#E07A1F", lw=2,
                label=f"{m['a']:.0f}*px^{m['b']:.2f} (R2={m['r2']:.2f})")
        ax.axhline(m["median_bytes"], ls="--", color="#6B7280", lw=1,
                   label=f"WP4 median {m['median_bytes']:.0f} B")
        ax.set(xscale="log", yscale="log", xlabel="ship length (px)",
               title=f"{level}: {m['product']}")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.2, which="both")
    np.atleast_1d(axes)[0].set_ylabel("payload (bytes, REAL measurement)")
    fig.suptitle("Payload cost of one ship vs its length -- MEASURED on 551 Airbus crops", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "wp6_size_model.png", dpi=160)
    print(f"\nSaved wp6_size_model.json, wp6_size_model.png in {args.out}")


if __name__ == "__main__":
    main()
