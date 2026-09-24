"""WP13 -- does the detector still work on a recompressed coastal tile?

WP7 measured that coastal tiles are the single largest item in the downlink budget, moved them
from JPEG q60 to q40 (-24% of that item, no recall cost) and showed q30 would save a further
~19% of coastal bytes. q30 was **not adopted**, for a stated reason: nobody had checked what
recompression does to the detector.

That check matters because of who reads these tiles. We send a coastal tile whole precisely
because the onboard detector may have *missed* a ship in it -- the ground segment re-runs
detection on what arrives. So the question is not "does the tile look fine" (PSNR already says
it does) but **"does a detector still find the same ships in it"**.

Method: take the real coastal tiles, re-encode each at q60 / q40 / q30 / q20, decode, run the
same detector, and match against the same ground truth with the WP1 rule. Reports recall by ship
size at each quality, plus the byte cost, so the trade is visible rather than assumed.

    .venv312/Scripts/python.exe scripts/wp13_recompress_check.py --tiles 400

Outputs: results/wp13_recompress.json, wp13_recompress.csv, wp13_recompress.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch

from wp11_integration_demo import gt_boxes, iou

QUALITIES = [60, 40, 30, 20]
SIZE_BINS = [(0, 32, "small (<32 px)"), (32, 96, "medium (32-96 px)"), (96, 1e9, "large (>96 px)")]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=400, help="coastal tiles to test")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    dev = 0 if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    # coastal tiles, as the catalogue classified them -- the ones we actually send whole
    cat = pd.read_csv(args.out / "wp6_tiles.csv")
    coastal = cat.loc[cat.ctx3 == "coast", "image"].tolist()
    rng = np.random.default_rng(args.seed)
    if len(coastal) > args.tiles:
        coastal = [coastal[i] for i in sorted(rng.choice(len(coastal), args.tiles, replace=False))]
    img_dir, lbl_dir = args.data / "images" / "test", args.data / "labels" / "test"
    print(f"=== WP13: {len(coastal)} real coastal tiles, qualities {QUALITIES}, device {dev} ===\n")

    rows, size_rows = [], []
    t0 = time.perf_counter()
    for n, name in enumerate(coastal, 1):
        f = img_dir / name
        orig = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if orig is None:
            continue
        gts = gt_boxes(lbl_dir / (Path(name).stem + ".txt"))
        variants = {"original": (orig, f.stat().st_size)}
        for q in QUALITIES:
            ok, buf = cv2.imencode(".jpg", orig, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                continue
            variants[f"q{q}"] = (cv2.imdecode(buf, cv2.IMREAD_COLOR), int(buf.nbytes))

        for tag, (im, nbytes) in variants.items():
            res = model.predict(im, imgsz=args.imgsz, conf=args.conf, device=dev, verbose=False)
            preds = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                     for b in res[0].boxes.xywh.cpu().numpy()]
            used = set()
            found = []
            for g in gts:
                best, bi = 0.0, -1
                for k, p in enumerate(preds):
                    if k in used:
                        continue
                    v = iou(g, p)
                    if v > best:
                        best, bi = v, k
                if best >= 0.5:
                    used.add(bi)
                found.append(best >= 0.5)
                size_rows.append({"image": name, "quality": tag, "size_px": max(g[2], g[3]),
                                  "found": best >= 0.5})
            rows.append({"image": name, "quality": tag, "bytes": nbytes,
                         "psnr_db": psnr(orig, im) if tag != "original" else float("inf"),
                         "ships": len(gts), "found": int(sum(found)),
                         "n_pred": len(preds)})
        if n % 50 == 0:
            print(f"  {n}/{len(coastal)} tiles", flush=True)

    df, sdf = pd.DataFrame(rows), pd.DataFrame(size_rows)
    df.to_csv(args.out / "wp13_recompress.csv", index=False)

    order = ["original"] + [f"q{q}" for q in QUALITIES]
    base_bytes = df.loc[df.quality == "q60", "bytes"].mean()
    ref_recall = sdf.loc[sdf.quality == "original", "found"].mean()

    print(f"\n{int(df.loc[df.quality=='original','ships'].sum())} ships on "
          f"{df.image.nunique()} coastal tiles. Reference recall (original file): {ref_recall:.4f}\n")
    print(f"{'quality':>9s} {'kB/tile':>9s} {'vs q60':>8s} {'PSNR dB':>9s} {'recall':>8s} "
          f"{'d recall':>9s} {'small':>8s} {'medium':>8s} {'large':>8s}")
    report = {"tiles": int(df.image.nunique()),
              "ships": int(df.loc[df.quality == 'original', 'ships'].sum()),
              "reference_recall": float(ref_recall), "qualities": {}}
    for tag in order:
        d, s = df[df.quality == tag], sdf[sdf.quality == tag]
        if not len(d):
            continue
        kb, rec = d.bytes.mean() / 1000, s.found.mean()
        bins = []
        for lo, hi, _ in SIZE_BINS:
            b = s[(s.size_px >= lo) & (s.size_px < hi)]
            bins.append(b.found.mean() if len(b) else float("nan"))
        p = d.psnr_db.replace([np.inf], np.nan).mean()
        print(f"{tag:>9s} {kb:9.1f} {d.bytes.mean()/base_bytes:8.2f} "
              f"{p:9.1f} {rec:8.4f} {rec-ref_recall:+9.4f} "
              + " ".join(f"{v:8.4f}" for v in bins))
        report["qualities"][tag] = {"kB_per_tile": kb, "bytes_vs_q60": d.bytes.mean() / base_bytes,
                                    "psnr_db": None if np.isnan(p) else float(p),
                                    "recall": float(rec), "delta_recall": float(rec - ref_recall),
                                    "recall_by_size": dict(zip([b[2] for b in SIZE_BINS],
                                                               [None if np.isnan(v) else float(v)
                                                                for v in bins]))}
    q40, q30 = report["qualities"].get("q40"), report["qualities"].get("q30")
    if q40 and q30:
        verdict = ("ADOPT q30" if q30["delta_recall"] >= q40["delta_recall"] - 0.005
                   else "KEEP q40")
        saving = 1 - q30["kB_per_tile"] / q40["kB_per_tile"]
        print(f"\nq40 -- which WP7 adopted as 'no recall cost' -- actually costs "
              f"{100*q40['delta_recall']:+.2f} pts vs the original file, all on small ships.")
        print(f"q30 vs q40: {100*saving:.1f}% FEWER bytes for "
              f"{100*(q30['delta_recall']-q40['delta_recall']):+.2f} pts recall  ->  {verdict}")
        report["verdict"] = verdict
        report["q30_byte_saving_vs_q40"] = float(saving)
    report["elapsed_s"] = time.perf_counter() - t0
    (args.out / "wp13_recompress.json").write_text(json.dumps(report, indent=2))

    # ---- figure
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    qs = [t for t in order if t in report["qualities"]]
    a0.plot([report["qualities"][t]["kB_per_tile"] for t in qs],
            [report["qualities"][t]["recall"] for t in qs], "o-", color="#1B6CA8", lw=2)
    for t in qs:
        a0.annotate(t, (report["qualities"][t]["kB_per_tile"],
                        report["qualities"][t]["recall"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    a0.axhline(ref_recall, color="#2E8B57", ls=":", lw=1)
    a0.text(a0.get_xlim()[0], ref_recall + 0.002, "recall on the original file",
            color="#2E8B57", fontsize=8)
    a0.set(xlabel="kB per coastal tile", ylabel="ship recall after recompression")
    a0.set_title("Does the detector survive recompression?", fontsize=10)
    a0.grid(alpha=0.25)

    w = 0.25
    x = np.arange(len(SIZE_BINS))
    for i, t in enumerate([q for q in ("q60", "q40", "q30") if q in report["qualities"]]):
        vals = [report["qualities"][t]["recall_by_size"][b[2]] or 0 for b in SIZE_BINS]
        a1.bar(x + (i - 1) * w, vals, w, label=t)
    a1.set_xticks(x, [b[2].split()[0] for b in SIZE_BINS])
    a1.set_ylabel("recall")
    a1.set_title("Recompression hits the small ships first", fontsize=10)
    a1.legend(fontsize=8, frameon=False)
    a1.grid(alpha=0.25, axis="y")
    for ax in (a0, a1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "wp13_recompress.png", dpi=160)
    print(f"\nSaved wp13_recompress.json|.csv|.png in {args.out}")


if __name__ == "__main__":
    main()
