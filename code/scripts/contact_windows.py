"""List the satellite's passes over the ground station and the bytes each pass can carry.

    python scripts/contact_windows.py                       # 24 h, CubeSat S-band, Sfax
    python scripts/contact_windows.py --link smallsat_xband --hours 48
    python scripts/contact_windows.py --tle sentinel.tle    # a real satellite (2-line file)

Outputs: results/passes.csv and results/passes.png
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig, find_passes, make_satellite


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--start", default="2026-09-18T00:00:00", help="UTC start, ISO format")
    ap.add_argument("--link", choices=sorted(LINK_PRESETS), default="cubesat_sband")
    ap.add_argument("--altitude", type=float, default=500.0, help="km (ignored with --tle)")
    ap.add_argument("--lat", type=float, default=GroundStation.lat_deg)
    ap.add_argument("--lon", type=float, default=GroundStation.lon_deg)
    ap.add_argument("--min-elev", type=float, default=5.0)
    ap.add_argument("--tle", type=Path, help="file with the two TLE lines of a real satellite")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    tle = None
    if args.tle:
        lines = [l.strip() for l in args.tle.read_text().splitlines() if l.strip()]
        tle = (lines[-2], lines[-1])
    sat = make_satellite(OrbitConfig(altitude_km=args.altitude, epoch=start), tle=tle)
    gs = GroundStation(lat_deg=args.lat, lon_deg=args.lon, min_elevation_deg=args.min_elev)
    link = LINK_PRESETS[args.link]
    passes = find_passes(sat, gs, start, args.hours, link, args.altitude)

    df = pd.DataFrame([{
        "rise_utc": p.rise.strftime("%Y-%m-%d %H:%M:%S"),
        "set_utc": p.set.strftime("%H:%M:%S"),
        "duration_min": round(p.duration_s / 60, 1),
        "max_elev_deg": round(p.max_elevation_deg, 1),
        "capacity_MB": round(p.capacity_bytes / 1e6, 1),
    } for p in passes])
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "passes.csv", index=False)

    print(f"Ground station {gs.name} ({gs.lat_deg}, {gs.lon_deg}), min elevation {gs.min_elevation_deg} deg")
    print(f"Link preset: {link.name}  |  {len(passes)} passes in {args.hours:g} h\n")
    print(df.to_string(index=False) if len(df) else "(no pass)")
    if len(df):
        print(f"\nTotal contact time: {df.duration_min.sum():.1f} min  |  "
              f"total capacity: {df.capacity_MB.sum():.0f} MB")
        gaps = [(b.rise - a.set).total_seconds() / 3600 for a, b in zip(passes, passes[1:])]
        if gaps:
            print(f"Longest gap without contact: {max(gaps):.1f} h")

        fig, ax = plt.subplots(figsize=(9, 3.6))
        labels = [p.rise.strftime("%H:%M") for p in passes]
        bars = ax.bar(labels, df.capacity_MB, color="#1B6CA8")
        for b, e in zip(bars, df.max_elev_deg):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{e:.0f} deg",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Capacity per pass (MB)")
        ax.set_xlabel("Pass start (UTC); label = max elevation")
        ax.set_title(f"Contact windows over {gs.name} -- {link.name} (simulation, assumed link)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.out / "passes.png", dpi=160)
        print(f"\nSaved {args.out / 'passes.csv'} and {args.out / 'passes.png'}")


if __name__ == "__main__":
    main()
