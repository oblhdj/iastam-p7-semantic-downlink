# IASTAM 6.0 — Problem 7: Transmitting Information Rather Than Raw Data

> ### ⚠ STALE (18 Sept) — read `../docs/STATUS.md` instead
> Every performance number below is from the **synthetic** Beta(5,2) workload and is superseded.
> "LoD + value-greedy **93.0% / 97.6%**" became **0.685** on real detections (WP6); the
> pre-filter section was re-measured on Airbus and the classic filter **failed as a gate**
> (0.645 recall), replaced by a learned gate (WP3, 0.988). The four "Known limitations / TODO"
> items at the bottom are **all done** (WP0 dedup, WP3, WP6). Kept for history only.


Onboard pipeline + downlink simulator for a ship-monitoring satellite.
The satellite decides **what** to send (confidence-aware level of detail) and
**in which order across passes** (value-aware scheduler), instead of downlinking raw images.

## Setup (Windows, Git Bash or PowerShell)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q          # 17 tests
```

## What is here
| Path | What it does | Course link (GI2) |
|---|---|---|
| `sat7/rle.py` | Airbus RLE masks ↔ boxes ↔ YOLO labels, IoU | — |
| `sat7/prefilter.py` | Classic CV stage: HSV cloud mask, top-hat + Otsu, opening, connected components, watershed, Canny | Computer Vision parts 1–3 |
| `sat7/orbit.py` | Contact windows over Sfax (Skyfield/SGP4) and capacity per pass `C = Σ R_k Δt_k` | Réseau sans fil |
| `sat7/scheduler.py` | Encoders (raw / Φsat-2-style patch / our LoD) and policies (No-buffer / FIFO / Value-greedy) | Optimisation, Design Patterns (Strategy) |
| `sat7/synthetic.py` | Synthetic sea tiles for tests and demos only | — |
| `scripts/contact_windows.py` | Pass table + chart | |
| `scripts/simulate_day.py` | **Contribution A** experiment: 5 strategies + load sweep, with 95% Wilson CIs | Statistiques |
| `scripts/airbus_to_yolo.py` | Airbus → YOLO dataset (`data.yaml`, splits) | |
| `scripts/eval_prefilter.py` | Pre-filter recall / drop rate / runtime (`--demo` or Airbus) | |

## Run
```bash
.venv/Scripts/python scripts/contact_windows.py
.venv/Scripts/python scripts/simulate_day.py
.venv/Scripts/python scripts/eval_prefilter.py --demo
# with the Kaggle Airbus data downloaded to D:/data/airbus:
.venv/Scripts/python scripts/airbus_to_yolo.py --airbus-dir D:/data/airbus --max-ship-images 8000
.venv/Scripts/python scripts/eval_prefilter.py --airbus-dir D:/data/airbus --split-csv data/yolo_ships/split.csv
```
Outputs go to `results/`.

## Assumptions (state them in the paper)
* **Orbit**: circular 500 km, 97.4° (sun-synchronous-like), epoch 2026-09-18. Real TLE supported (`--tle`).
* **Ground station**: Sfax (34.74 N, 10.76 E), 5° minimum elevation.
* **Link** `cubesat_sband`: 2 MHz, SNR 12 dB at zenith, free-space 1/d² scaling, Shannon rate capped at 8 Mbps, 80% efficiency;
  only `--link-share` (default 25%) of each pass is given to this payload.
* **Workload** (`WorkloadConfig`): 40 000 tiles/day; contexts 15% cloud / 20% ships / 5% coast / 60% empty;
  10% dark vessels; detector confidence ~ Beta(5, 2); 2% false positives on empty tiles. **Synthetic** until the
  real detector outputs are plugged in.
* **Level of detail** (`LoDConfig`): L0 40 B, L1 2 KB, L2 30 KB (progressive), coastal tile 80 KB (progressive),
  thumbnail 1 KB; dark vessel weight 5; confidence thresholds 0.4 / 0.9.
* **Energy**: no onboard hardware available. `eval_prefilter.py --power-w W` gives *runtime on this laptop × assumed power* —
  label it an ESTIMATE.

## Results so far (simulation, synthetic workload — not real data)
One day, CubeSat S-band, 25% link share (see `results/sim_table.csv`):

| Strategy | Ships delivered | Dark vessels delivered |
|---|---|---|
| Bent pipe (raw, FIFO) | 0.4% | 0.6% |
| Φsat-2 style (fixed patch, FIFO) | 66.6% | 67.1% |
| Ours: LoD + no buffering | 43.9% | 42.6% |
| Ours: LoD + FIFO | 68.5% | 69.3% |
| **Ours: LoD + value-greedy** | **93.0%** | **97.6%** |

Same encoder, only the scheduler changes: 68.5% → 93.0% (effect of Contribution A).

Classic pre-filter on synthetic tiles: 100% ship recall on clear-sky tiles (191/191), 72% at 30% cloud cover,
33% at 70% (many of those ships are painted under the cloud). Cloud edges create false candidates. Must be re-measured on Airbus.

## Known limitations / TODO
* Airbus crops overlap → check near-duplicates across splits (FAISS) before reporting test scores.
* Pre-filter thresholds tuned on synthetic tiles only; tune on Airbus.
* Plug real YOLO confidences into the scheduler instead of the Beta(5, 2) assumption.
* Latency inside a pass is approximated linearly; link model ignores weather and pointing losses.
