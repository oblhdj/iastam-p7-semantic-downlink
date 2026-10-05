# Phase-3 Results — Energy-Aware Semantic Downlink for Maritime Object Detection

**IASTAM 6.0 · Track 4 · Problem 7 · ENIS, Sfax** — Makni, Ben Saad, Ketata, Megdiche.
_Draft of the Phase-3 measured results. The accepted Phase-2 paper reported no numbers by design;
this is the measured counterpart, section-for-section against its framework. Draw slides / the
final submission from this. Every figure is labelled **REAL / SIM / LIT / TARGET / ASSUMPTION**;
the `code/results/*.csv|json` files are authoritative over any prose here. Traceability to the
paper's claims: `demo/PAPER_COVERAGE.md`._

---

## Abstract (now with measurements)

Onboard semantic processing turns a day of maritime imaging that **could not be downlinked at all**
— 60,124 MB against a ~210 MB/day link — into **108 MB that delivers 68.5 % of the ships at an
assumed 15 % cloud** (band 0.40–0.82 over 0–50 % cloud), a **557× data reduction** [SIM-over-REAL].
Against a *fair* Phi-sat-2-style baseline this is **+6.3 points of recall for ~7× the bytes**
[SIM]; against raw transmission over the same link it is 0.685 vs 0.0024, because raw tiles do not
fit. The onboard pipeline — a trained YOLOv8n detector (mAP50 **0.804** [REAL]), a 47k-parameter
learned gate, SAHI with global-coordinate fusion, a confidence-aware level-of-detail encoder, and a
value-aware multi-pass scheduler — is **provably within 0.251 % of the optimal schedule** at
operational scale [REAL]. The per-stage energy model shows the gating redesign saves **≥57.6 % of
processing energy** and that, once pixels stop being sent, **compute dominates the radio 8.9 : 1**
[SIM]. An optional inter-satellite relay cuts **worst-case latency 11.6 h → 6.2 h** without changing
what is delivered [latency SIM, energy TARGET]. All timings are a laptop RTX 5060; onboard execution
is argued as a **TARGET** against flown Myriad-class hardware (§10), not demonstrated on flight hardware.

## 1. Experimental setup

* **Dataset — REAL.** Airbus Ship Detection (optical), **53,195 tiles of 768×768, 81,723 ships**.
  19.1 % of tiles have a near-duplicate twin; a naive split would leak **1,617 groups** into test,
  ours leaks **0** (train 42,555 / val 5,320 / test 5,320). [report 01]
  *Divergence from the paper:* it suggested xView/VisDrone/DOTA/HRSID; we use optical Airbus (which
  the paper permits) and evaluate SAHI on **synthetic swaths stitched from tiles**, because an
  Airbus tile is already 768² and cannot be sliced (§10).
* **Detector — REAL.** YOLOv8n (3.15 M params, ~12.5 GFLOPs/tile at 768). Test **mAP50 0.804,
  P 0.817, R 0.729**; 10.7 ms/tile GPU. ONNX FP32 is lossless and 1.21× faster on CPU (38.6 ms/tile);
  dynamic INT8 **rejected** (8.5× slower, −2.2 pts on small ships). [report 02]
* **Orbit / link — SIM.** SGP4 propagation, ground station at Sfax: **5 passes/day, 34 min contact,
  11.4 h max gap**, cubesat S-band. [orbit]
* **Hardware — caveat.** Every T_k is a laptop RTX 5060 + CPU. "Runs onboard" is a TARGET (§10).

## 2. B0→B4 campaign (paper Table II)  — all five configurations ran

| cfg | onboard | sent | ship recall | key figure | label |
|-----|---------|------|-------------|-----------|-------|
| **B0** | none | full image | — | 60,124 MB/day raw reference | SIM |
| **B1** | YOLO/tile | detections | **0.782** (small 0.587 / med 0.937 / large 1.0) | detector-only reduction | REAL |
| **B2** | SAHI+YOLO+fusion | detections | **0.745** | 1.56× compute vs regular tiling | REAL |
| **B3** | +semantic policy (LoD + scheduler) | metadata/ROI | **0.685** @15 % cloud | **557×** reduction, **108 MB/day**, latency med 4.5 h | SIM-over-REAL |
| **B4** | +direct/optional relay | metadata/ROI | **= B3** (reroute) | worst-case latency **11.6 h → 6.2 h**, relay 44 % of items, +34 % comm energy | latency SIM, energy TARGET |

The campaign runs end to end (`wp18_campaign_runner.py`) with sanity gates: B3 reproduces the
authoritative headline row within 2 % bytes / 0.01 recall; B4's direct-only path reproduces B3 and
its reroute leaves recall/MB unchanged (relay changes only latency + energy). [report 21]

## 3. Detection quality (paper §VIII-E.2)

mAP50 **0.804**, precision 0.817, recall 0.729 [REAL]. By object size, recall is **small 0.739 /
medium 0.958 / large 0.994 at conf ≥ 0.05** — but at the operational 0.25 cut small-ship recall is
**0.50–0.61**; never quote 0.739 beside "threshold 0.25". **Small-object recall is the single
largest residual loss** and the honest weak point. The detector is **underconfident** (says 0.352,
right 0.408 of the time); isotonic calibration cuts ECE **8.9×** (0.0878 → 0.0099) and moves the
cheap confidence rung to its correct place (`conf_high` 0.9 → 0.670). [reports 02, 03]

## 4. Downlink reduction (paper §VIII-E.1, eq 27)

`DR = 1 − D_tx/D_raw = 1 − 108/60,124 = ` **99.82 % (557×)** [SIM-over-REAL]. The byte budget is not
what intuition expects: **86 % of the "semantic" downlink was coastal tiles (62 %) + blanket
thumbnails (24 %), ship information only 13 %**. Re-keying thumbnail gating onto the learned gate
returns **3.8 % of the whole downlink at zero measured recall cost** [report 15]. ⚠ The coastal
q60→q40 recompression we first adopted as "free" was later measured to **cost 4.0 points of recall,
all on small ships** (q30 rejected outright) — so byte savings on coastal tiles are a *recall
trade*, not free, and the honest lever there is sending *fewer* coastal tiles, not lower-quality
ones. [reports 06, 14, 15]
*Correction carried from Phase 2:* the Phi-sat-2 baseline had been unfair (ocean-only). The **fair**
baseline (+coastal) scores **0.622**, so the margin is **+6.3 points for ~7× the bytes**, not +16.7.
Raw transmission over the same link delivers **0.0024** — raw tiles do not fit. [report 06]

## 5. Information preservation (paper §VIII-E.3)

At nominal load **downlink efficiency = 1.0**: we deliver **100 % of what the onboard software still
knew** — the gap from 1.0 to 0.685 is the detector + cloud, not the downlink. The real chain
(real JPEGs → prefilter → gate → detector → LoD → scheduler → ground) reproduces the stored
catalogue to 4 decimals with **0 decisions changed** at every threshold, so the semantic records
preserve the detector's decisions exactly. [reports 05, 12]

## 6. Energy model (paper §V, eqs 8–19)

`E_proc = Σ P_k·T_k`, `E_comm = P_tx·D_tx/R_tx`, `ES = 1 − E_proposed/E_baseline`, built in
`wp17_energy_model.py` reading each stage time live with a sanity gate. [report 17]
* **ES_proc = 57.6 %** for the default all-CPU onboard build — a **power-free time ratio** (P_cpu
  cancels), so it holds on any processor; a lower bound, since the gate also drops empties the
  per-tile accounting doesn't credit.
* **Once pixels stop being sent, compute dominates the radio 8.9 : 1** — so energy optimisation
  belongs on the detector/gate, not the transmitter.
* Versus a bent pipe, **ES_total ≈ 98.2 %** — spend ~51 kJ/day of compute+radio to avoid ~2,851
  kJ/day of raw transmission. Powers P_k are ASSUMPTION and swept; ES_total stays 97–99 % across all.

## 7. Latency & the relay (paper §V-D, §VII, eqs 25–29)

`T_total = T_inf + T_enc + T_comm + T_dec`; median delivery **4.5 h** [SIM]. The optional relay uses
the paper's exact `J = λ_E·E + λ_T·T` path choice over real ISL windows: a complementary-coverage
relay (RAAN +90°) carries 44 % of items and **halves worst-case latency, 11.6 h → 6.2 h** [SIM],
for **+34 % communication energy** [TARGET — joules await a real P_isl/R_isl]. B4 is a **reroute**,
not extra capacity: recall and bytes are B3's verbatim. **Relay always costs more energy than direct
(~1.8×) — it is a latency buy, not a power saving.** [reports 19–21]

## 8. Optimality & robustness (strengthens the paper's §VI objective)

* **Greedy is 0.251 % from the exact optimum** at operational scale (exact DP, verified against
  brute force), 4.62 % on saturated small windows; FIFO loses 67–73 %. [report 08]
* **Perfect foresight over a whole day is worth ≤ 0.2 %** → do **not** build arrival prediction. The
  schedule is provably optimal up to 40k tiles/day. [report 09]
* **Scheduling only pays once the link saturates.** At 40k the scheduler ≈ FIFO (both deliver what's
  known); its **2.26× advantage over FIFO appears at 160k** tiles/day. State both operating points.
* **Sensitivity:** every invented constant swept; **conclusions hold in 69 of 70 settings**. **Cloud
  fraction dominates everything** (recall range 0.42) — quote every recall with its cloud fraction.
  Constraint for the paper: `thumb_value` must stay below ~0.05 (baseline 0.01 has 5× margin). [report 11]

## 9. Ablations (paper §VIII-G)

| ablation | result | evidence |
|---|---|---|
| without SAHI | B1 0.782 vs B2 0.745 @1.56× compute — SAHI is a measured trade, not a free win | reports 13, 16 |
| without adaptive downlink | fixed coastal policy loses to queue-aware in every regime but one | report 07 |
| without ROI | LoD-rung ablations; the ROI rung earns its bytes | report 06 |
| without relay | direct-only = B3 exactly (sanity gate a) | report 21 |
| without tile overlap / without fusion | partly isolated (edge-trigger vs overlap); not yet their own rows | reports 13, 16 |

## 10. Limitations (paper §X, extended honestly)

1. **Onboard = TARGET, not a prototype.** All timings are a laptop. But the work (3.15 M-param
   detector + 47 k gate) is the class our baseline **Φ-sat-2 already runs onboard** on a ~1 W Myriad 2
   VPU [LIT]; at 40k tiles/day it is a **few-% duty cycle even on a VPU 6× slower** than our laptop,
   and ES_proc is hardware-independent. A prototype still owes wall-clock + quantised accuracy on a
   real board, thermal, and rad-tolerance. [report 22]
2. **Dataset.** Airbus optical + synthetic swaths for SAHI (§1). A large-scene optical set would
   strengthen B2.
3. **SAHI is selective, not blanket** — a cut ship is usually still detected, so blanket SAHI is a
   poor trade; SAHI costs 1.56× (down from 2.25×) and the result is window-size dependent. [reports 13, 16]
4. **Cloud fraction** moves recall more than any other quantity and the Airbus set (0.4 % cloud)
   cannot settle it — a climatology is needed [LIT]. Always state the assumed fraction.
5. **P_tx** is the weakest number (invented); it only enters E_comm, which is 9× smaller than E_proc.

## 11. Conclusion — the central question, answered

The paper asked whether **less transmitted data can preserve more useful mission information.**
Measured: **yes.** A 557× data cut delivers **100 % of what the satellite still knew** at nominal
load and **+6.3 points over a fair baseline**, the schedule is within **0.25 %** of optimal, and the
energy saved by not sending pixels dwarfs the extra compute (ES_total ≈ 98 %). The remaining loss is
**the detector on small ships**, not the downlink — which is where Phase-4 effort belongs. What is
not yet proven, and we say so, is the flight-hardware wall-clock and the quantised accuracy on a real
onboard board: this is a **demo of the idea, measured end to end on real data — not a flown prototype.**

---
_Sources: `demo/PAPER_COVERAGE.md` (paper→evidence map), `reports/01–22`, `code/results/*`,
`docs/STATUS.md`. Superseded figures to avoid (`docs/START_HERE.md §7`): 442×/511×→557×,
97.1 %→0.685, +16.7→+6.3, 2.7×→2.26×, "INT8 lossless"→false, coastal q40 "free"→costs 4 pts._
