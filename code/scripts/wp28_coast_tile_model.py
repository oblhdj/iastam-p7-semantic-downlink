"""WP28 -- the coastal context tile, measured: per-tile sizes and what they do to the headline.

The coastal context tile is the largest product in the downlink budget. Until 10 Oct 2026 the
encoders charged every one of them a flat 32,420 B -- WP7's whole-tile q40 ladder, which was sampled
on 400 random ship-bearing tiles (mostly open sea, which compresses well). A real coastal tile
carries land texture and is about twice that. This script

  1. (--measure) encodes EVERY real coastal test tile at JPEG q40, baseline and progressive, and
     writes one row per tile to results/wp28_coast_tile_bytes.csv -- numbers only, no pixels. The
     `packet_B` column (baseline JPEG + 18 B image header + CCSDS packet headers and CRC) is what
     sat7.real_workload attaches to a day, so each coastal tile is charged its own measured size.
  2. recomputes the data-reduction headline both ways on the same days wp6_simulate_real averages
     over, product by product, and writes results/wp28_coast_tile_model.json.

Labels: coastal tile bytes REAL (measured encodings of real Airbus tiles); thumbnails, chips and ROI
crops still MODELED (LoDConfig / wp6 size model); the day (tiles per day, context mix, capture times)
ASSUMPTION; the raw reference is every non-cloud tile as uncompressed 8-bit RGB.

    .venv/Scripts/python.exe scripts/wp28_coast_tile_model.py --measure    # needs data/yolo_ships
    .venv/Scripts/python.exe scripts/wp28_coast_tile_model.py              # from the committed table
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sat7.real_workload import (COAST_SIZES_CSV, RealWorkloadConfig, load_catalogue, load_size_model,
                                workload_from_catalogue)
from sat7.scheduler import (COAST_TILE_BYTES_MEASURED, COAST_TILE_BYTES_WP7, RAW_TILE_BYTES, LoDConfig,
                            encode_lod, encode_raw)

TILE, QUALITY = 768, 40
IMG_HEADER_B = 18              # sat7.semantic image header: id, x0, y0, w, h, quality, codec
PACKET_OVERHEAD_B = 8          # CCSDS space packet: 6 B primary header + 2 B CRC-16
PACKET_DATA_B = 4096 - 2       # user data per packet (sat7.semantic.Packetizer)
KIND_NAMES = {"tile": "coastal context tile (JPEG q40)", "thumb": "thumbnail", "L2": "ROI + wake crop",
              "L1": "small chip", "fp": "false-alarm chip", "L0": "metadata record",
              "mosaic": "coastal mosaic"}


def packet_bytes(payload: int) -> int:
    """Bytes on the wire for one product of `payload` bytes, segmented into space packets."""
    return payload + PACKET_OVERHEAD_B * math.ceil(payload / PACKET_DATA_B)


def measure(data: Path, tiles: pd.DataFrame, out: Path) -> pd.DataFrame:
    from sat7.imagery import load_image
    from sat7.semantic import encode_jpeg
    img_dir = data / "images" / "test"
    rows = []
    for name in sorted(tiles.image[tiles.ctx3 == "coast"]):
        px = load_image(img_dir / name)[0]
        base = len(encode_jpeg(np.ascontiguousarray(px), QUALITY, False))
        prog = len(encode_jpeg(np.ascontiguousarray(px), QUALITY, True))
        rows.append({"image": name, "jpeg_q40_baseline_B": base, "jpeg_q40_progressive_B": prog,
                     "packet_B": packet_bytes(base + IMG_HEADER_B)})
    df = pd.DataFrame(rows)
    df.to_csv(out / COAST_SIZES_CSV, index=False, lineterminator="\n")
    return df


def _dist(v) -> dict:
    v = np.asarray(v, float)
    return {"tiles": int(len(v)), "median_B": float(np.median(v)), "mean_B": round(float(v.mean()), 1),
            "p10_B": round(float(np.percentile(v, 10)), 1), "p90_B": round(float(np.percentile(v, 90)), 1)}


def day_budget(wl, lod: LoDConfig) -> dict:
    by = defaultdict(lambda: [0, 0.0])
    for it in encode_lod(wl, lod):
        by[it.kind][0] += 1
        by[it.kind][1] += it.size
    return {k: {"items": n, "bytes": b} for k, (n, b) in by.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true", help="re-measure the tiles (needs the Airbus data)")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--tiles", type=int, default=40_000, help="ASSUMPTION: tiles imaged per day")
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3, help="days averaged (wp6_simulate_real uses 3)")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    res = args.results

    if args.measure:
        tiles0, _ = load_catalogue(res)
        table = measure(args.data, tiles0, res)
        print(f"measured {len(table)} coastal tiles -> {res / COAST_SIZES_CSV}")
    table = pd.read_csv(res / COAST_SIZES_CSV)
    tiles, ships = load_catalogue(res)             # now carries the measured sizes
    coast_images = set(tiles.image[tiles.ctx3 == "coast"])
    if set(table.image) != coast_images:
        raise SystemExit("the size table does not cover exactly the catalogue's coastal tiles; rerun --measure")
    mean_packet = float(table.packet_B.mean())
    if abs(mean_packet - COAST_TILE_BYTES_MEASURED) > 0.5:
        raise SystemExit(f"scheduler.COAST_TILE_BYTES_MEASURED = {COAST_TILE_BYTES_MEASURED} but the table's "
                         f"mean is {mean_packet:.1f}; update the constant")

    size_model = load_size_model(res / "wp6_size_model.json")
    models = {"measured_per_tile": LoDConfig(conf_low=args.det_thr, size_model=size_model),
              "modeled_wp7_flat": LoDConfig(conf_low=args.det_thr, size_model=size_model,
                                            coast_tile_bytes=COAST_TILE_BYTES_WP7)}
    days = []
    for k in range(args.repeats):
        wl, st = workload_from_catalogue(tiles, ships, RealWorkloadConfig(
            tiles_per_day=args.tiles, det_thr=args.det_thr, seed=args.seed + k))
        raw = encode_raw(wl)
        day = {"seed": args.seed + k, "tiles": st.tiles, "ships": st.ships, "context_counts": st.context_counts,
               "raw_tiles": len(raw), "D_raw_bytes": float(sum(i.size for i in raw))}
        for name, lod in models.items():
            day[name] = day_budget(wl, lod)
        days.append(day)

    def mean_mb(f) -> float:
        return float(np.mean([f(d) for d in days])) / 1e6

    d_raw = mean_mb(lambda d: d["D_raw_bytes"])
    out_models = {}
    for name in models:
        kinds = sorted({k for d in days for k in d[name]},
                       key=lambda k: -np.mean([d[name].get(k, {"bytes": 0})["bytes"] for d in days]))
        d_tx = mean_mb(lambda d: sum(v["bytes"] for v in d[name].values()))
        prod = {}
        for k in kinds:
            mb = mean_mb(lambda d: d[name].get(k, {"bytes": 0.0})["bytes"])
            n = float(np.mean([d[name].get(k, {"items": 0})["items"] for d in days]))
            prod[k] = {"product": KIND_NAMES.get(k, k), "items_per_day": round(n, 1), "MB": round(mb, 3),
                       "share": round(mb / d_tx, 4), "mean_B": round(mb * 1e6 / n, 1) if n else None,
                       "size_source": "MEASURED (wp28 table)" if (k == "tile" and name == "measured_per_tile")
                       else "MODELED"}
        out_models[name] = {
            "D_tx_offered_MB": round(d_tx, 2), "reduction_x": round(d_raw / d_tx, 1),
            "DR_percent": round(100 * (1 - d_tx / d_raw), 3), "products": prod,
            "per_day": [{"seed": d["seed"],
                         "D_tx_offered_MB": round(sum(v["bytes"] for v in d[name].values()) / 1e6, 2),
                         "reduction_x": round(d["D_raw_bytes"] / sum(v["bytes"] for v in d[name].values()), 1)}
                        for d in days]}

    n_coast = out_models["measured_per_tile"]["products"]["tile"]["items_per_day"]
    rest = out_models["measured_per_tile"]["D_tx_offered_MB"] - out_models["measured_per_tile"]["products"]["tile"]["MB"]
    sens = [{"coast_tile_B": s, "D_tx_MB": round(rest + n_coast * s / 1e6, 1),
             "reduction_x": round(d_raw / (rest + n_coast * s / 1e6), 0)}
            for s in (25_900, int(COAST_TILE_BYTES_WP7), 50_000, int(round(mean_packet)), 81_919)]

    report = {
        "label": "coastal tile bytes REAL (measured JPEG q40 of every real coastal test tile); thumbnails, "
                 "chips and ROI crops MODELED; day (tiles/day, context mix, capture times) ASSUMPTION",
        "definition": "reduction = D_raw / D_tx; D_raw = every non-cloud tile as uncompressed 8-bit RGB "
                      f"({RAW_TILE_BYTES:,} B); D_tx = bytes the LoD ladder offers to the downlink (all of "
                      "it is delivered at this load -- the scheduled figure is results/wp6_real_table.csv)",
        "days": {"n": args.repeats, "seeds": [d["seed"] for d in days], "tiles_per_day": args.tiles,
                 "det_thr": args.det_thr, "D_raw_MB_mean": round(d_raw, 1),
                 "raw_tiles_mean": round(float(np.mean([d["raw_tiles"] for d in days])), 1)},
        "coastal_tile_q40": {
            "measured_all_coastal_test_tiles": {
                "jpeg_baseline": _dist(table.jpeg_q40_baseline_B),
                "jpeg_progressive": _dist(table.jpeg_q40_progressive_B),
                "packet_B (baseline + image header + CCSDS)": _dist(table.packet_B)},
            "flat_fallback_COAST_TILE_BYTES_MEASURED": COAST_TILE_BYTES_MEASURED,
            "earlier_modeled_COAST_TILE_BYTES_WP7": COAST_TILE_BYTES_WP7,
            "why_they_differ": "the WP7 ladder sampled 400 random ship-bearing tiles, mostly open sea; a "
                               "coastal tile carries land texture. WP13 measured 65.5 kB at q40 on its "
                               "own coastal sample."},
        "headline": {
            "canonical": {"model": "measured_per_tile", **{k: out_models["measured_per_tile"][k]
                                                           for k in ("D_tx_offered_MB", "reduction_x", "DR_percent")}},
            "earlier_modeled_estimate": {"model": "modeled_wp7_flat", **{k: out_models["modeled_wp7_flat"][k]
                                                                         for k in ("D_tx_offered_MB", "reduction_x", "DR_percent")}},
            "still_modeled_share_of_canonical_bytes": round(
                1 - out_models["measured_per_tile"]["products"]["tile"]["share"], 4)},
        "models": out_models,
        "sensitivity_flat_coast_tile_size": sens,
    }
    (res / "wp28_coast_tile_model.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== WP28 coastal tile model: {args.repeats} days of {args.tiles} tiles, D_raw {d_raw:,.1f} MB ===")
    t = report["coastal_tile_q40"]["measured_all_coastal_test_tiles"]
    print(f"coastal tile q40, {len(table)} tiles: baseline median {t['jpeg_baseline']['median_B']:.0f} B "
          f"(mean {t['jpeg_baseline']['mean_B']:.0f}), progressive median {t['jpeg_progressive']['median_B']:.0f} B; "
          f"on the wire mean {mean_packet:.1f} B; earlier model {COAST_TILE_BYTES_WP7:.0f} B")
    for name, m in out_models.items():
        print(f"\n[{name}] D_tx {m['D_tx_offered_MB']} MB -> {m['reduction_x']}x (DR {m['DR_percent']}%)")
        for k, p in m["products"].items():
            print(f"   {p['product']:34s} {p['items_per_day']:9.1f} /day  {p['MB']:8.2f} MB  "
                  f"{100 * p['share']:5.1f}%  [{p['size_source']}]")
    print("\nsaved wp28_coast_tile_model.json" + (f" and {COAST_SIZES_CSV}" if args.measure else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
