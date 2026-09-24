# WP1 completion — ONNX FP32 and INT8, measured (24 Sept 2026)

Closes the must-have "Detector metrics on the test split, FP32 and INT8". It had never been
measured: `wp1_metrics.json` carries `"onnx": {"error": "No module named 'onnx'"}` because the
3.12 environment had no pip. Script: `../code/scripts/wp1_export_int8.py`. Data: `results/wp1_export.json`,
log `results/wp1_export_run.log`. All figures REAL (600 test tiles, paired; latency 25 tiles).

INT8 here is **onnxruntime dynamic quantization** (`quantize_dynamic`, QInt8 weights).
Ultralytics' `int8=True` targets TFLite/TensorRT/OpenVINO, not ONNX, so it is the wrong tool.

## Headline: dynamic INT8 is not worth it on CPU — 3.6x smaller, 8.5x slower, and it costs small ships

| model | size MB | recall (600 tiles) | small <32px | medium | large | ms/tile CPU |
|---|---|---|---|---|---|---|
| pytorch FP32 | 6.27 | 0.8031 | 0.6062 | 0.9614 | 0.9952 | 46.6 |
| ONNX FP32 | 12.34 | **0.8031** | 0.6062 | 0.9614 | 0.9952 | **38.6** |
| ONNX INT8 | **3.43** | 0.7887 | 0.5841 | 0.9486 | 0.9952 | **328.8** |

(pytorch on GPU: 12.5 ms/tile.)

* **ONNX FP32 export is exactly lossless** — recall identical to pytorch to 4 decimals, and
  mean confidence agrees to 5e-6. It is also the fastest CPU option, **1.21x faster than
  pytorch CPU** at identical accuracy. *This*, not INT8, is the onboard build.
* **INT8 is 8.5x SLOWER than ONNX FP32 on CPU** (328.8 vs 38.6 ms/tile) and 7.1x slower than
  pytorch CPU. Expected direction for a convolution-heavy detector: dynamic quantization mainly
  hits MatMul/Gemm and inserts quantize/dequantize pairs around convolutions that then run in a
  slower path. It buys 3.6x less storage, which we are not short of.
* ⚠ **Contradicts the report's "1.5-3.3x faster" [LIT]**, which is TensorRT / static QDQ with a
  calibration set, not ONNX dynamic quantization. Either drop the claim or re-scope it to
  TensorRT, which we have not measured.

## The FMEA row "INT8 degradation -> small ships lost" is CONFIRMED
Paired on the same 970 ships, so this is a like-for-like flip count, not two independent runs:

| bin | INT8 - FP32 |
|---|---|
| overall | **-1.44 pts** |
| small (<32 px) | **-2.21 pts** |
| medium (32-96 px) | -1.29 pts |
| large (>96 px) | **0.00** |

16 ships lost, 2 gained. **The damage is monotone in smallness and exactly zero on large
vessels** — quantization noise competes with the signal only where the signal is weakest, which
is already our worst bin. The earlier 40-tile pilot called INT8 "lossless"; it was too small to
see a 1.4-point effect. `results/wp1_export_SMOKETEST_ONLY_40tiles.json` should not be cited.

## ⚠ Correction to a headline detector number
Medium (0.961 vs 0.958) and large (0.995 vs 0.994) reproduce `wp1_metrics.json` almost exactly.
Small does not: **0.606 here vs the headline 0.739.** The cause is the confidence threshold, not
the subset. `wp1_metrics.json`'s by-size table is computed at **conf >= 0.05**; the pipeline's
operational cut is **0.25**. Recomputed from `wp1_gt_matched.csv`:

| conf cut | overall | small | medium | large |
|---|---|---|---|---|
| >= 0.05 | 0.861 | **0.739** | 0.958 | 0.994 |
| >= 0.25 (post-hoc filter, full split) | 0.725 | **0.496** | 0.896 | 0.989 |
| >= 0.25 (native, 600-tile subset) | 0.803 | **0.606** | 0.961 | 0.995 |

The last two differ because re-running detection at 0.25 changes NMS and lets a ground-truth
ship match a different surviving box, so post-hoc filtering is a lower bound. Either way,
**"small-ship recall 0.739" and "operational threshold 0.25" cannot both appear in the same
sentence.** Quote 0.739 only with `conf >= 0.05` attached. This is the same effect WP6 measured
from the other side: cutting 0.25 -> 0.05 buys +8.8 points of ships for +8% bytes.

## What to do
1. Ship **ONNX FP32** onboard: lossless, 1.21x faster than pytorch on CPU, one dependency.
2. Drop dynamic INT8. If the size reduction is ever needed, the honest path is static
   quantization with a calibration set, or TensorRT on a real accelerator — and it must be
   re-measured on small ships specifically, because that is where this one broke.
3. Fix the by-size recall claim in the report to carry its confidence threshold.
