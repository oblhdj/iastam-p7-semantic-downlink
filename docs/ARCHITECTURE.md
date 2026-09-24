# IASTAM P7 — architecture, and what is actually built

Status of every stage as of **24 Sept 2026**, verified against the repo rather than the plan.
Legend: **[DONE]** measured and tested · **[PART]** works but incomplete · **[GAP]** not started
· **[!]** the plan claims it is done and it is not.

## The pipeline

```
                          ONBOARD                                    GROUND
  ┌──────────┐
  │  imager  │  tiles, 768x768
  └────┬─────┘
       v
  ┌─────────────────────┐   cloud / coast / sea + candidate boxes
  │ [1] classic filter  │  [DONE] sat7/prefilter.py
  │     HSV, top-hat,   │  [GAP]  as a GATE it fails: recall 0.645 for an 18% saving
  │     watershed, Canny│  [GAP]  learned replacement written but NEVER RUN
  └────┬──────────┬─────┘
       │          └──────────────► n_candidates ──┐  (free uncertainty signal)
       v                                          │
  ┌─────────────────────┐                         │
  │ [2] detector YOLOv8 │  [DONE] mAP50 0.804, recall 0.739 small / 0.958 med / 0.994 large
  │     768 px, FP32    │  [!]    INT8 / ONNX export FAILS — plan ticks "FP32 and INT8"
  └────┬────────────────┘  [GAP]  never run on the val split; no edge-hardware timing
       │ boxes + confidence
       v
  ┌─────────────────────┐
  │ [3] calibration     │  [GAP]  NOT STARTED. Only 7.5% of real ships clear conf 0.9,
  │                     │         so the 40 B L0 rung almost never fires. The ladder
  └────┬────────────────┘         below is cut for a distribution the detector never makes.
       v
  ┌─────────────────────────────────────────┐
  │ [4] level of detail  sat7/scheduler.py  │  [DONE] payload per ship from a measured
  │   conf > 0.9      -> L0   40 B          │         power law: 137.5·px^0.57 (L1),
  │   0.25..0.9       -> L1   real bytes    │         43.3·px^1.14 (L2), R² 0.76 / 0.89
  │   dark            -> L2   real bytes    │  [DONE] thumbnails gated by two opinions
  │   coastal tile    -> whole / mosaic  <──┼──── unconfirmed candidates
  │   every tile      -> thumbnail          │  [DONE] coastal q40, adaptive + queue-aware
  └────┬────────────────────────────────────┘
       v
  ┌─────────────────────┐   <──── buffer pressure ────┐
  │ [5] scheduler       │  [DONE] value-greedy + aging + progressive truncation
  │   value per byte    │  [DONE] optimality gap MEASURED: 0.069% at scale, 4.62% lumpy
  │   8 GB store        │  [PART] storage drop policy untested at scale
  └────┬────────────────┘  [GAP]  joint multi-pass optimum not solved
       v
  ┌─────────────────────┐
  │ [6] downlink        │  [DONE] SGP4 orbit, Sfax station, 6 passes/36 h, 210 MB
  └────┬────────────────┘  [GAP]  single ground station only; no multi-station sweep
       v                                        ┌──────────────────────────────┐
       └───────────────────────────────────────►│ [7] ground: ships, positions │
                                                │  [GAP] no integration demo   │
                                                └──────────────────────────────┘
```

## Stage by stage

| # | Stage | Status | Evidence |
|---|---|---|---|
| 0 | Data integrity | **[DONE]** | 53,195 tiles, 6,282 duplicate pairs, 0 groups leaked |
| 1 | Classic pre-filter | **[DONE]** measured, **[GAP]** as a gate | recall 0.645 / 17.7% saving → negative result, reported |
| 1b | Learned gate | **[GAP]** | `wp3_train_gate.py` exists, never run, no `models/` |
| 2 | Detector | **[DONE]** FP32, **[!]** INT8 | `wp1_metrics.json` → `onnx: "No module named 'onnx'"` |
| 3 | Calibration | **[GAP]** | no script, no reliability diagram, no ECE |
| 4 | Level of detail | **[DONE]** | WP4 + WP7: real per-ship bytes, real product ladder |
| 5 | Scheduler | **[DONE]** | WP5/6/8 + optimality gap |
| 6 | Orbit / link | **[DONE]** single station | `passes.csv`, 11.4 h max gap |
| 7 | Integration demo | **[GAP]** | nothing runs end-to-end |
| — | FMEA / fault tests | **[GAP]** | FMEA is prose; 0 of 61 tests inject a fault |
| — | SAHI / tile seams | **[GAP]** | `sahi_cost.png` is a **geometric formula**, not an experiment |

## Code map

| module | lines | tests | role |
|---|---|---|---|
| `sat7/scheduler.py` | 623 | 14 | items, LoD encoder, policies, `simulate`, `simulate_online` |
| `sat7/dedup.py` | 510 | 11 | pHash + ORB near-duplicate finder, group split |
| `sat7/real_workload.py` | 210 | 13 | real detections → `Workload` (WP6/7/8 signals) |
| `sat7/prefilter.py` | 186 | 5 | classic CV stage |
| `sat7/optimum.py` | 158 | 11 | exact DP optimum + upper bound |
| `sat7/orbit.py` | 133 | 2 | SGP4 passes, link budget |
| `sat7/rle.py` | 83 | 5 | Airbus RLE ↔ boxes |
| **61 tests total** | | | |

## The two feedback loops (what makes this more than a pipeline)

```
  detector ──uncertainty──► level of detail ──bytes──► scheduler
      ▲                            ▲                       │
      └── classic filter disagrees─┘                       │
                                   └───── buffer pressure ─┘
```

1. **Cross-check loop** — the cheap classic stage and the network disagree → that tile
   probably hides a missed ship → spend more bytes on it. Free: both stages already run.
2. **Pressure loop** — the buffer state sets how much detail the encoder may spend.
   Built in WP8; this is what stops a fixed policy being wrong in every regime but one.

## Honest status of the headline numbers

| claim | value | basis |
|---|---|---|
| data reduction vs bent pipe | **557×** | real detections, real payload sizes (421× → 517× → 511× → 557× as WP7/WP9/WP2 landed) |
| ships delivered | **0.685** ⚠ **at an assumed 15% cloud** | = 100% of what the satellite still knows. Band across 0–50% cloud: **0.40–0.82** (WP10). Never quote it bare |
| vs FIFO under congestion | **2.26×** | 160k tiles/day (2.10× on the dataset mix; both fell slightly because the cheaper encoding helps FIFO too) |
| vs *fair* Phi-sat-2 baseline | **+6.3 points at 40k, +4.1 at 160k** | corrected; was +16.7 with an unfair baseline |
| greedy vs exact optimum | **0.251%** at scale | exact DP, brute-force verified (6.67% on lumpy windows). Grew from 0.069% when `conf_high` 0.670 made item sizes lumpier — the price of the 8.4% byte saving |
| queue-aware robustness | worst-case regret **0.0058 vs 0.0123** | held out on 2 untuned link budgets (refreshed 16:32) |

**The binding constraint is the detector, not the downlink.** Everything downstream of stage 2
now delivers essentially all of what stage 2 hands it.

## Ranked gaps — audited 24 Sept, 14:30

### A. Stale results — ✅ RESOLVED 24 Sept 15:29–15:32

All five pre-WP9 result sets were re-run (`results/rerun_post_wp9.log`, 6 commands, 0 failures).
Every number improved or held and **no conclusion changed**. Residual debt is documentation,
not computation: `wp6_report.md`, `wp7_report.md`, `wp5_*_report.md` and `wp8_report.md` carry
REFRESHED banners but their **inner tables still show pre-refresh numbers** — the `.csv` files
are authoritative. `code/README.md` (18 Sept) is fully stale and still quotes the synthetic
93.0%/97.6%.

### B. Built but never run — ✅ RESOLVED 24 Sept 15:47–15:53

| script | result |
|---|---|
| `wp2_predict_split.py` | val dumped: 17,201 boxes, 8,173 ships, 0 images overlapping test |
| `wp2_calibrate.py` | detector underconfident; isotonic cuts ECE 8.9× (0.0878 → 0.0099); `conf_high` should be **0.670**, not 0.9 |
| `wp3_train_gate.py` | 47k-param gate: **98.8% of ships at 40.2% of empties dropped** vs the classic filter's 55.5% — **+43.3 points** |

### C. INT8 — ✅ RESOLVED 24 Sept 15:47, and the early number was wrong

INT8 is **8.5× slower** on CPU (328.8 vs 38.6 ms/tile), not 7× — the earlier 593 vs 82 ms was
measured under CPU contention. It is **not lossless**: paired on 970 ships it costs **−1.44 pts
overall, −2.21 on small ships, exactly 0.00 on large**, confirming the FMEA row. **ONNX FP32 is
the onboard build** — bit-identical recall, 1.21× faster than pytorch on CPU.

### D. Claimed in the plan, still absent

* "Detector metrics FP32 **and INT8**" — ✅ now genuinely true (C).
* "Pre-filter safety and saving metrics on real tiles (WP3)" — ✅ now genuinely true (B).
* All three *nice-to-haves* (calibration, offline LP bound, sensitivity) are now done too.
* **So exactly one must-have tick is still false: the interim paper + 2-min video.**
* Interim paper + 2-min video — ticked; **video script does not exist**, PDFs still carry the
  synthetic 97.1%, the unfair baseline and pre-WP6 framing.

### E. Never started

1. ~~**Integration demo**~~ — ✅ **DONE 24 Sept (WP11)**, `results/wp11_integration_report.md`.
   The chain runs on real JPEGs and the catalogue replay is **validated**: every structural
   field agrees exactly over 400 tiles and **zero decisions change** at any threshold.
   It found two things: **52% of all reported ship loss is 12 real cloud-tile ships resampled
   287×**, and the classic pre-filter (51.1 ms/tile) **costs 3.8× the detector it gates**.
2. **SAHI / tile seams** — `sahi_cost.png` is a geometric formula, not an experiment.
3. **Edge-hardware timing** — "runs onboard" is unsupported; all figures are a laptop RTX 5060.
4. **Multi-station** — single ground station; 11.4 h max gap dominates latency.
5. **2-D sensitivity** — WP10 was one-at-a-time; `thumb_value` x cloud plainly interact.
6. **Burst/correlated arrivals** — tiles are resampled i.i.d.; real passes see strips.
7. **q30 coastal tiles** — a further -18% bytes, pending a detector-on-recompressed check.
8. **CI** — 86 tests run only when someone remembers.
9. **`aging_per_hour`** — WP10 showed it does nothing (2.4e-4). Remove it or justify it.

### F. Documentation debt

Ten result documents (`wp5`, `wp5b`, `wp6`, `wp7`, `wp8`, `wp9`, `wp10`), none folded into the
paper. The code is well ahead of anything written down, and the interim is due **26 Sept**.
