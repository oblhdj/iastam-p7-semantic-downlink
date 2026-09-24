"""WP1 completion: export the detector to ONNX FP32 and INT8, and measure BOTH.

The technical plan ticks "Detector metrics on the test split, FP32 and INT8" as a must-have.
Only FP32 existed: `wp1_metrics.json` records ``"onnx": {"error": "No module named 'onnx'"}``
because the 3.12 environment had no pip, so onnx could never be installed. Environment fixed
(WP: see HANDOFF), so this closes the deliverable.

INT8 here is **onnxruntime dynamic quantization** of the ONNX graph. Ultralytics' own
``int8=True`` targets TFLite/TensorRT/OpenVINO, not ONNX, so it is not the right tool.
Dynamic quantization needs no calibration set; it mainly quantizes MatMul/Gemm, so on a
convolution-heavy detector expect a large size reduction and only a modest speed change.

Method. Rather than ``model.val()`` (mAP only, and it writes into ultralytics' *global*
runs_dir), all three models are run over the **same** tiles and matched to ground truth here,
with the same greedy IoU >= 0.5 rule used in WP1. That makes the FP32 -> INT8 comparison
paired, and yields **recall by ship size** -- which is what the FMEA row "INT8 degradation ->
small ships lost" actually claims, and what mAP alone would hide.

Evaluating ONNX on CPU is slow (onnxruntime here is the CPU build, which is also the honest
choice for an "onboard" latency figure), so accuracy uses a subset. A paired comparison on
~600 tiles resolves a degradation far smaller than any that would matter.

Deliberately does NOT touch results/wp1_predictions.csv -- the WP6-WP10 catalogues are built
from it and regenerating it here would silently invalidate them.

    .venv312/Scripts/python.exe scripts/wp1_export_int8.py
    .venv312/Scripts/python.exe scripts/wp1_export_int8.py --n-val 1200 --n-timing 40
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

import numpy as np
import pandas as pd

from wp2_predict_split import gt_boxes, iou           # same matching rule as WP1

SIZE_BINS = [(0, 32, "small (<32 px)"), (32, 96, "medium (32-96 px)"), (96, 1e9, "large (>96 px)")]


def run_model(path: Path, files, imgsz: int, device, conf: float, batch: int = 16) -> pd.DataFrame:
    """Predict over ``files`` and match every ground-truth ship. One row per GT ship."""
    from ultralytics import YOLO
    model = YOLO(str(path), task="detect")
    # the ONNX graphs are exported with a fixed batch of 1 (dynamic=False), which is the
    # realistic onboard shape; feeding them a batch raises "invalid dimensions for input"
    if path.suffix == ".onnx":
        batch = 1
    lbl_dir = files[0].parent.parent.parent / "labels" / files[0].parent.name
    rows = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        res = model.predict([str(x) for x in chunk], imgsz=imgsz, conf=conf, device=device,
                            verbose=False)
        for f, r in zip(chunk, res):
            preds = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                     for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            used = set()
            for g in gt_boxes(lbl_dir / (f.stem + ".txt")):
                best, best_i = 0.0, -1
                for k, p in enumerate(preds):
                    if k in used:
                        continue
                    v = iou(g, p[:4])
                    if v > best:
                        best, best_i = v, k
                if best >= 0.5:
                    used.add(best_i)
                rows.append({"image": f.name, "size_px": max(g[2], g[3]), "found": best >= 0.5,
                             "conf": preds[best_i][4] if best_i >= 0 and best >= 0.5 else 0.0,
                             "n_pred": len(preds)})
    return pd.DataFrame(rows)


def recall_table(df: pd.DataFrame) -> dict:
    out = {"ships": int(len(df)), "recall": float(df.found.mean()),
           "mean_conf_found": float(df.loc[df.found, "conf"].mean()) if df.found.any() else 0.0}
    for lo, hi, name in SIZE_BINS:
        sub = df[(df.size_px >= lo) & (df.size_px < hi)]
        out[name] = {"ships": int(len(sub)),
                     "recall": float(sub.found.mean()) if len(sub) else float("nan")}
    return out


def time_model(path: Path, files, imgsz: int, device, warmup: int = 3) -> float:
    from ultralytics import YOLO
    model = YOLO(str(path), task="detect")
    for f in files[:warmup]:
        model.predict(str(f), imgsz=imgsz, device=device, verbose=False)
    t = []
    for f in files:
        t0 = time.perf_counter()
        model.predict(str(f), imgsz=imgsz, device=device, verbose=False)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.25, help="operational threshold for recall")
    ap.add_argument("--n-val", type=int, default=600, help="tiles for the paired accuracy check")
    ap.add_argument("--n-timing", type=int, default=25, help="tiles for the latency median")
    ap.add_argument("--reuse-exports", action="store_true", default=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    onnx_path = args.weights.with_suffix(".onnx")
    int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    report: dict = {"weights": str(args.weights), "imgsz": args.imgsz, "conf": args.conf}

    # ---------------------------------------------------------------- export (reuse if present)
    if not (args.reuse_exports and onnx_path.exists()):
        t = time.perf_counter()
        onnx_path = Path(YOLO(str(args.weights)).export(format="onnx", imgsz=args.imgsz,
                                                        opset=13, dynamic=False, simplify=True))
        print(f"exported {onnx_path.name} in {time.perf_counter() - t:.1f}s")
    if not (args.reuse_exports and int8_path.exists()):
        from onnxruntime.quantization import QuantType, quantize_dynamic
        t = time.perf_counter()
        quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
        print(f"quantized {int8_path.name} in {time.perf_counter() - t:.1f}s")

    report["size_MB"] = {"pytorch": round(args.weights.stat().st_size / 1e6, 2),
                         "onnx_fp32": round(onnx_path.stat().st_size / 1e6, 2),
                         "onnx_int8": round(int8_path.stat().st_size / 1e6, 2)}
    report["size_MB"]["int8_smaller_x"] = round(
        report["size_MB"]["onnx_fp32"] / report["size_MB"]["onnx_int8"], 2)
    print("model sizes (MB):", report["size_MB"])

    files = sorted((args.data / "images" / "test").glob("*.jpg"))
    rng = np.random.default_rng(0)
    val_files = [files[i] for i in sorted(rng.choice(len(files), min(args.n_val, len(files)),
                                                     replace=False))]
    dev = 0 if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------- paired accuracy
    print(f"\npaired accuracy on {len(val_files)} tiles (conf >= {args.conf}) ...")
    report["accuracy"] = {}
    frames = {}
    for tag, path, d in (("pytorch_fp32", args.weights, dev),
                         ("onnx_fp32", onnx_path, "cpu"),
                         ("onnx_int8", int8_path, "cpu")):
        t = time.perf_counter()
        try:
            df = run_model(path, val_files, args.imgsz, d, args.conf)
            frames[tag] = df
            report["accuracy"][tag] = recall_table(df)
            print(f"  {tag:13s} recall {df.found.mean():.4f}  ({time.perf_counter() - t:.0f}s)")
        except Exception as e:
            report["accuracy"][tag] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  {tag:13s} FAILED: {type(e).__name__}: {str(e)[:120]}")

    if "onnx_fp32" in frames and "onnx_int8" in frames:
        a, b = frames["onnx_fp32"], frames["onnx_int8"]
        delta = {"recall": float(b.found.mean() - a.found.mean())}
        for lo, hi, name in SIZE_BINS:
            ma = (a.size_px >= lo) & (a.size_px < hi)
            mb = (b.size_px >= lo) & (b.size_px < hi)
            delta[name] = float(b.loc[mb, "found"].mean() - a.loc[ma, "found"].mean())
        # same ships, so a McNemar-style count of who flipped is meaningful
        flips = pd.DataFrame({"fp32": a.found.to_numpy(), "int8": b.found.to_numpy()})
        delta["lost_by_int8"] = int(((flips.fp32) & (~flips.int8)).sum())
        delta["gained_by_int8"] = int(((~flips.fp32) & (flips.int8)).sum())
        report["int8_minus_fp32"] = delta
        print("\nINT8 - FP32 (paired, same tiles):")
        print(json.dumps(delta, indent=2))

    # ---------------------------------------------------------------- latency
    print(f"\nlatency on {args.n_timing} tiles ...")
    report["latency_ms_per_tile"] = {}
    todo = [("onnx_fp32_cpu", onnx_path, "cpu"), ("onnx_int8_cpu", int8_path, "cpu"),
            ("pytorch_cpu", args.weights, "cpu")]
    if torch.cuda.is_available():
        todo.insert(0, ("pytorch_gpu", args.weights, dev))
    for tag, path, d in todo:
        try:
            ms = time_model(path, files[:args.n_timing], args.imgsz, d)
            report["latency_ms_per_tile"][tag] = round(ms, 1)
            print(f"  {tag:16s} {ms:8.1f} ms/tile")
        except Exception as e:
            report["latency_ms_per_tile"][tag] = f"error: {type(e).__name__}"

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wp1_export.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved wp1_export.json in {args.out}")


if __name__ == "__main__":
    main()
