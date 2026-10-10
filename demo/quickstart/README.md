# Quickstart — the onboard chain, live, on any laptop CPU

The full pipeline needs the 7.7 GB Airbus dataset and the torch/CUDA env (`code/.venv312`). This
folder runs the **same `sat7` code** in about **2 seconds on a laptop CPU**, with **no GPU, no
torch and no dataset download**. It is for showing judges the system *executing*. The measured
results stay in `code/results/` and `paper/PHASE3_RESULTS.md`.

```bash
python -m pip install -r demo/quickstart/requirements.txt
python demo/quickstart/run_demo.py
```

What runs live, on a 3×3 swath of 768 px tiles (2304 × 2304 px):

| step | code (unchanged repo modules) | what you see |
|---|---|---|
| [1] pre-filter each tile | `sat7.prefilter.run_prefilter` | cloud / coast / sea context, beside the context recorded for the real tile |
| [2] B1 detector per tile, then B2 SAHI + global-coordinate fusion over the swath | `OnnxYolo` (below) + `sat7.perception.detect_image(mode="sahi")` | detector calls, boxes, timing |
| [3] P0–P3 decision per detection | `sat7.priority.classify` | box, confidence, length, level, bytes, and the reason |
| [4] semantic packet vs raw | `sat7.semantic.encode_image` (binary records, JPEG crops of the pixels, CCSDS space packets) + `decode_downlink` | **measured** packet bytes by level, headers + CRC, JPEG headers, the ground decode, modeled bytes beside them, raw bytes, reduction `1 − D_tx/D_raw` |
| [5] downlink order | `sat7.scheduler.ValueGreedy` | metadata first, ROI crops next, coastal context last |
| [6] reference | the committed REAL detections for the same 9 tile IDs (`wp1_predictions.csv`) through the same P0–P3 rules | what the real tiles give (sizes modeled: no Airbus pixels are bundled) |
| [7] canon | read from `code/results/*` | mAP50 0.804, B1 0.339 → B2 0.717 @ 1.56× (same scenes), 341× (coastal tiles measured; thumbnails and crops modeled), P0–P3 vs LoD |

It writes `demo/quickstart/_out/swath_annotated.jpg` (detections coloured by P-level, SAHI
windows, tile seams; watermarked when the tiles are synthetic), `downlink.bin` (the actual
packet stream, decodable with `sat7.semantic.decode_downlink`) and `packet.json` (every
detection with its level and bytes, and every item in downlink order). `_out/` is gitignored.

**Runtime.** On the team laptop the run takes 1.7 s, or 2.1 s wall-clock including Python
start-up, at 29 ms per detector call. Forced to one CPU thread, a call takes 108 ms, so the 25
calls need about 2.7 s. A laptop ten times slower per core still finishes in about 30 s. These
are live timings of this demo, not project claims; the project's measured CPU figure is
38.6 ms/tile in `wp1_export.json`.

## Labels — the README convention, plus one

Every line of output carries a label. **REAL / SIM / SIM-over-REAL / ASSUMPTION** mean exactly
what they mean in the [top-level README](../../README.md); those numbers are read from committed
results files, never typed in. This demo adds one more:

* **SYNTH** — computed live on the synthetic stand-in tiles. It shows that the chain runs end to
  end. **It is not a measurement**, and it is never compared with the headline.
* **REAL-sample** — what SYNTH becomes when you run on the real tiles (`--data`, below): real
  numbers, but from a 9-tile sample, not the 40k-tile day behind the headline.

The 9 tiles were chosen so that every P-level appears, so they hold far more ships than an
average tile. That is why their reduction is **not** the 341× headline: a real day is mostly
empty sea, which sends almost nothing. Measured packet streams (re-run 10 Oct 2026): synthetic
tiles 45,681 B (99.68 %, 310×; synthetic coast compresses easily) and the real tiles with `--data`
158,034 B (98.88 %, 90×). The size model charges 153,948 B for that same real packet now that a
coastal tile is costed at its measured mean (66.1 kB); on the earlier flat 32.4 kB it charged
86,552 B — two real coastal context tiles alone are 133 kB (see
`code/results/wp25_semantic_packets.json`). That gap is what moved the headline from 557× to 341×.

## Why the tiles are synthetic

The [Airbus Ship Detection Challenge rules](https://www.kaggle.com/competitions/airbus-ship-detection/rules)
§7A allow non-commercial use only. §7B commits entrants not to "publish, redistribute or
otherwise provide" the Competition Data to anyone who has not accepted the rules. A handful of
demo tiles is still redistribution, so **no Airbus pixel is checked in here.**

Each file in `tiles/` is a **procedurally rendered stand-in** for one real test tile, named in
`tiles/manifest.json` (`stands_in_for`). Only the *layout* comes from the real tile, and only
through committed repo artefacts:

* ship positions and sizes come from that tile's committed detector boxes
  (`wp1_predictions.csv`, conf ≥ 0.25, de-fragmented). A box flagged as a false alarm in
  `wp6_pred_flags.csv` is drawn as a bright speck, not a ship.
* the scene type comes from the pre-filter context recorded in `wp6_tiles.csv`.

The sea, hulls, wakes, coastline and clouds are invented by `make_synthetic_tiles.py`. The
renderer was adjusted by eye until the trained detector fires on its hulls. The pre-filter
classifies all 9 stand-ins exactly as it classified the real tiles; a test pins this.

| # | stands in for | scene | exercises |
|---|---|---|---|
| 1 | `00113a75c.jpg` | seven ships, mostly confident | P1 metadata, P0 fragments |
| 2 | `0002756f7.jpg` | two uncertain ships + a sub-threshold box | P2 / P0 |
| 3 | `00f34434e.jpg` | a 170 px ship beside small craft | size-dependent ROI bytes |
| 4 | `03204a586.jpg` | coast, ships near shore | P3 + shared coastal context tile |
| 5 | `00293fb9e.jpg` | empty sea | sends nothing |
| 6 | `01933be65.jpg` | one small ship | P2 |
| 7 | `05d407505.jpg` | coast, a large ship | P3 |
| 8 | `0c5bf1395.jpg` | no ship, one measured false alarm | a false alarm costs bytes too |
| 9 | `024e6ba29.jpg` | cloud | dropped by the pre-filter before detection |

Regenerate them with `python demo/quickstart/make_synthetic_tiles.py` (deterministic, seed 7; the
sha256 of each file is in the manifest).

## Running on the real tiles (team laptop only)

If you have the dataset (`code/data/yolo_ships`, built by `code/scripts/wp0_build_dataset.py`):

```bash
python demo/quickstart/run_demo.py --data code/data/yolo_ships --out demo/_live/quickstart_real
```

This runs the 9 real tiles instead and adds a **reproducibility check** to section [6]. The live
ONNX-on-CPU boxes are paired with the committed PyTorch-on-GPU boxes in `wp1_predictions.csv`,
and the check counts decision flips at the 0.25 cut and the 0.670 P1 threshold.

Verified 9 Oct 2026: **40 of 40 boxes paired, max |Δconf| 3.4e-4, 0 decision flips at either
threshold**, and the live pre-filter context matched the catalogue on all 9 tiles. Never commit
those tiles or `_live/`; the rules above apply.

## The model

`model/best.onnx` (12.3 MB, sha256 `5463fc72…22d4fd9`) is a byte-identical copy of
`code/runs/ships/weights/best.onnx`. That is the trained YOLOv8n (3.15 M params) exported by wp1
as ONNX FP32 (opset 13, 768 × 768 input), which wp1 measured as **lossless against PyTorch**
(identical recall in every size bucket, `code/results/wp1_export.json`). The decoder in
`run_demo.py` (`OnnxYolo`, `decode`, `nms_xyxy`) mirrors ultralytics' predict post-processing:
RGB/255, score > conf, greedy NMS at IoU 0.7, boxes clipped to the image. That is why its boxes
reproduce the committed ones above.

Licence note: the weights are fine-tuned from Ultralytics YOLOv8n, and the export's own metadata
carries Ultralytics' AGPL-3.0 licence string. The repo's MIT licence covers the code, not these
weights. They were also trained on Airbus Competition Data under the non-commercial terms above.

## Where this differs from the full pipeline, stated plainly

* **No learned gate.** The 47k-parameter gate is a torch model and is not run here. In the P0–P3
  scheme the gate only suppresses the coastal context tile on tiles it calls empty, so the
  per-detection levels are unaffected.
* **No AIS.** Every detection has `dark=False`, so the dark-vessel bump (one level up) never fires.
* **Onboard costing.** Every detection is costed on its own confidence, because the satellite
  cannot tell a false alarm from a ship. The catalogue runs (wp23) cost a known false alarm as a
  zero-value 40 B report instead.
* **3×3 swath, not 4×4.** SAHI here is 16 windows for 9 tiles. The measured **1.56×** (25 windows
  for 16 tiles, `wp26_b0_b4.json`) is quoted from file in section [7], not from this run.
* **P3 on coasts.** `sat7.priority.classify` escalates *every* detection above 0.25 on a coastal
  tile to P3, confident ones included (`PriorityConfig.coast_escalation = "all"`, the measured
  default; `"uncertain"` escalates only those below the P1 threshold). wp23's numbers were produced
  with the same default.
* **Thresholds** (0.25 cut, 0.670 P1/P2 boundary) and the P-level policy are **ASSUMPTION**, as
  everywhere in the repo, and swept in report 11 / report 23.

## Files

| file | what |
|---|---|
| `run_demo.py` | the demo: one script, sections [1]–[7] |
| `requirements.txt` | numpy, opencv-python-headless, onnxruntime, sgp4, skyfield (no torch) |
| `model/best.onnx` | the trained detector, ONNX FP32 |
| `tiles/*.jpg`, `tiles/manifest.json` | 9 synthetic stand-ins + what each stands in for |
| `make_synthetic_tiles.py` | regenerates `tiles/` from committed results |
| `../../code/tests/test_quickstart.py` | pins decode rules, tile hashes, pre-filter contexts, P-level/byte semantics, the cross-check, and an end-to-end run (skipped without onnxruntime) |
