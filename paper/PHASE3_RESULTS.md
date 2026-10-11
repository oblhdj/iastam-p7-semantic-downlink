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
— 60,124 MB against a ~210 MB/day link — into **176 MB that delivers 68.5 % of the ships at an
assumed 15 % cloud** (band 0.40–0.82 over 0–50 % cloud), a **341× data reduction** [SIM-over-REAL;
coastal tiles at their measured JPEG size, thumbnails and crops still modeled — the earlier
all-modeled estimate was 557×, §4].
Against a *fair* Phi-sat-2-style baseline this is **+6.3 points of recall for ~11× the bytes**
[SIM]; against raw transmission over the same link it is 0.685 vs 0.0024, because raw tiles do not
fit. The onboard pipeline — a trained YOLOv8n detector (mAP50 **0.804** [REAL]), a 47k-parameter
learned gate, SAHI with global-coordinate fusion, a confidence-aware level-of-detail encoder, and a
value-aware multi-pass scheduler — is **provably within 0.88 % of the optimal schedule** at
operational scale [REAL]. The per-stage energy model shows the gating redesign saves **≥57.6 % of
processing energy** and that, once pixels stop being sent, **compute dominates the radio 5.5 : 1**
at laptop powers (28 W CPU, an assumption; on a 1–2 W edge processor the radio can dominate, §6b)
[SIM]. An optional inter-satellite relay cuts **worst-case latency 35.2 h → 10.9 h** (per-item
median 7.9 h → 3.4 h) without changing what is delivered [latency SIM, energy TARGET]. All timings are a laptop RTX 5060; onboard execution
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
| **B1** | YOLO, one call on the whole 3072 px scene | detections | **0.339** (small 0.046 / med 0.542 / large 0.856), precision 0.681 | no slicing: the scene is shrunk 4× to the detector's 768 px input | REAL |
| **B2** | SAHI+YOLO+fusion, same scenes | detections | **0.717** (small 0.505 / med 0.924 / large 0.990), precision 0.696 | 25 windows per scene = 1.56× the calls of a plain 16-tile pass | REAL |
| **B3** | +semantic policy (LoD + scheduler) | metadata/ROI | **0.685** delivered @15 % cloud | **341×** reduction, **176 MB/day** (coastal tiles measured; thumbnails/crops modeled), latency med 4.5 h | SIM-over-REAL |
| **B4** | +direct/optional relay | metadata/ROI | **= B3** (reroute) | worst-case latency **35.2 h → 10.9 h** (per-item median 7.9 → 3.4 h), relay 66 % of items, +52 % comm energy | latency SIM, energy TARGET |

**Two inputs, stated.** B1 and B2 are measured on the same 150 scenes — each 4×4 real test tiles,
1,164 ships, matched at IoU 0.5 (`wp26_b0_b4.py`, read by `wp18_campaign_runner.py`). B3 and B4 are
one simulated 40,000-tile day over the per-tile detection catalogue. B1 ↔ B2 is a like-for-like
step; B2's detection recall and B3's *delivered* recall are measured on different inputs and are not
points on one curve. (`wp26` also carries B3/B4 on those same scenes with real packets — delivered
recall 0.624, 97.9 MB/day — as a cross-check, not the canonical row.)

**Reference, not the paper's B1:** the detector on native 768 px tiles — one call per tile, nothing
to shrink — reaches **recall 0.766, precision 0.754** on all 5,320 test tiles / 8,173 ships
(`wp24_detection_eval.json`). Until 10 Oct 2026 this table reported a 262-ship sample of that
(0.782) as B1 and a 47-ship swath sample (0.745) as B2; both are kept in `wp18_campaign.json`
under `legacy_small_samples`.

The campaign runs end to end (`wp18_campaign_runner.py`) with sanity gates: B3 reproduces the
authoritative headline row within 2 % bytes / 0.01 recall (its own day: 178.5 MB, 0.680; the row
above is the three-day mean); B4's direct-only path reproduces B3 and its reroute leaves recall/MB
unchanged (relay changes only latency + energy). [report 21]

## 2b. B3 and B4 under the paper's Table I levels (pure P0–P3)

The B3 row above uses our level-of-detail ladder. The paper's Table I names four levels — **P0**
discard, **P1** metadata, **P2** metadata + compressed ROI, **P3** metadata + ROI + contextual
image — and `sat7.priority` implements exactly those. This is the same campaign with that encoder:
same days, same passes, same scheduler, same metrics (`--semantic priority`). Everything here is
**SIM-over-REAL** at an assumed 15 % cloud; the level thresholds (0.25 and 0.670) and the values of
a ROI and of a context image are **ASSUMPTIONS**.

**B3, mean of three simulated days** (`wp6_real_table.csv` · `wp6_real_table_paper.csv`):

| | LoD ladder (headline, §2) | pure P0–P3 (paper Table I) |
|---|---|---|
| ship recall delivered | **0.685** (days 0.680–0.690) | **0.622** (days 0.617–0.625) |
| what the encoder can describe (ceiling) | 0.685 | 0.622 |
| share of the ceiling delivered | 1.00 | 1.00 |
| dark-vessel recall [dark flag ASSUMPTION] | 0.679 | 0.624 |
| sent per day | **176.3 MB** | **88.4 MB** |
| data reduction vs raw | **341×** | **680×** |
| latency per ship, median / p90 | 4.5 h / 9.6 h | 4.4 h / 9.4 h |

**B4, one simulated day (seed 0), relay as a reroute** (`wp18_campaign.json` ·
`wp18_campaign_paper.json`; latency SIM, a window estimate on the relay side; energy **TARGET**):

| | LoD ladder | pure P0–P3 |
|---|---|---|
| that day's B3: recall, bytes | 0.680, 178.5 MB | 0.617, 90.5 MB |
| items routed, share taking the relay | 45,264, 66 % | 23,732, 44 % |
| per-item latency, median | 7.9 h → 3.4 h | 4.6 h → 1.8 h |
| per-item latency, worst case | 35.2 h → 10.9 h | 11.6 h → 6.2 h |
| communication energy [TARGET] | 8.5 → 12.9 kJ | 4.3 → 5.8 kJ |

How to read it:

* **P0–P3 is half the bytes for 6.3 points less recall.** The points are ships the detector never
  fired on, on coastal tiles: the ladder sends every coastal tile whole and so recovers them; P0–P3
  sends a context tile only where a detection already exists, and is credited with that detection.
* **On recall alone, pure P0–P3 equals the fair Phi-sat-2-style baseline** — 0.622 for both, the
  detected ships — at 88.4 MB against 16.3 MB. What its extra bytes carry is the ROI and the
  context image of Table I, evidence for the ground to verify a detection; ship recall does not
  score that. The **+6.3 points over the fair baseline is a result of the LoD ladder, not of P0–P3.**
* **P0–P3 fits the link.** 88.4 MB is under the 134.5 MB the link share sustains per 24 h; the
  ladder's 176 MB is 130 % of it (§5). That is why the worst-case item waits 11.6 h, not 35.2 h,
  before any relay.
* The scheduler matters for P0–P3 in the same place it does for the ladder: at 160k tiles/day
  value-greedy delivers 0.621 of the ships and FIFO 0.325 (`wp6_real_sweep_paper.csv`).

The LoD row remains the headline; the P0–P3 column is the paper's own policy, measured the same way.
Reproduce: `python scripts/wp6_simulate_real.py --semantic priority --tag paper` and
`python scripts/wp18_campaign_runner.py --semantic priority --tag paper`. Both refuse to run P0–P3
without a tag, so the canonical files cannot be overwritten.

## 3. Detection quality (paper §VIII-E.2)

mAP50 **0.804**, precision 0.817, recall 0.729 [REAL]. By object size, recall is **small 0.739 /
medium 0.958 / large 0.994 at conf ≥ 0.05** — but at the operational 0.25 cut small-ship recall is
**0.50–0.61**; never quote 0.739 beside "threshold 0.25". **Small-object recall is the single
largest residual loss** and the honest weak point. The detector is **underconfident** (says 0.352,
right 0.408 of the time); isotonic calibration cuts ECE **8.9×** (0.0878 → 0.0099) and moves the
cheap confidence rung to its correct place (`conf_high` 0.9 → 0.670). [reports 02, 03]

How a large scene reaches the detector decides what it finds [REAL]: one call on a whole 3072 px
scene keeps recall **0.339** (small ships 0.046); SAHI on the same scenes, **0.717** (small 0.505)
[`wp26`]. On native 768 px tiles, one call per tile, recall is **0.766** at precision 0.754 over all
8,173 test ships [`wp24`].

## 4. Downlink reduction (paper §VIII-E.1, eq 27)

`DR = 1 − D_tx/D_raw = 1 − 176.3/60,124 = ` **99.71 % (341×)** [SIM-over-REAL; **coastal tiles
measured, thumbnails and crops still modeled**], mean of three simulated days
(`wp6_real_table.csv`). Each coastal context tile is charged its own measured size — baseline JPEG
q40 plus image and packet headers, median 66.5 kB over all 1,415 coastal test tiles
(`wp28_coast_tile_model.json`). The remaining 24 % of the bytes (thumbnails, ship crops, records)
are model sizes.
*Earlier estimate, kept on record:* until 10 Oct 2026 every coastal tile was charged a flat 32.4 kB
from a size ladder measured on open-sea tiles; that gave 108 MB and **557×**. Reports 05–06 narrate
that estimate and `wp6_real_table_modeledcoast.csv` reproduces it.
The byte budget is not what intuition expects, and measuring made it more lopsided: **coastal tiles
are 76 % of the downlink, thumbnails 16 %, everything that describes a ship 6 %** (the all-modeled
budget of report 06 read 62 % / 24 % / 13 %). Re-keying thumbnail gating onto the learned gate
returns **2.3 % of the whole downlink at zero measured recall cost** [report 15]. ⚠ The coastal
q60→q40 recompression we first adopted as "free" was later measured to **cost 4.0 points of recall,
all on small ships** (q30 rejected outright) — so byte savings on coastal tiles are a *recall
trade*, not free, and the honest lever there is sending *fewer* coastal tiles, not lower-quality
ones. [reports 06, 14, 15]
*Correction carried from Phase 2:* the Phi-sat-2 baseline had been unfair (ocean-only). The **fair**
baseline (+coastal) scores **0.622**, so the margin is **+6.3 points for ~11× the bytes**, not +16.7.
Raw transmission over the same link delivers **0.0024** — raw tiles do not fit. [report 06]

## 5. Information preservation (paper §VIII-E.3)

At nominal load **downlink efficiency = 1.0**: we deliver **100 % of what the onboard software still
knew** — the gap from 1.0 to 0.685 is the detector + cloud, not the downlink. **This is a
single-day statement:** one day's 176 MB is given the ~210 MB of passes that fall in a 36 h window.
The 25 % link share sustains only **134.5 MB per 24 h**, so a day offers **130 %** of it. Simulated
over six consecutive days, value-greedy still recovers 0.677–0.688 of the ships every day — by
truncating low-value products — while FIFO's median latency climbs from 9 h to 43 h and its recall
then collapses (§12). Recall counts a progressively truncated item as delivered once it clears
`min_fraction` (10 % of its bytes); that is a modelling assumption, not a measurement. A 35 % share
carries the day outright.
The real chain
(real JPEGs → prefilter → gate → detector → LoD → scheduler → ground) reproduces the stored
catalogue to 4 decimals. Over all 5,320 test tiles and 8,173 ships, **one ship changes side** at
the 0.25 onboard cut (`wp11_integration.json` counts it under `det_thr` and under `conf_low`, which
are the same 0.25 threshold in that run) and none at `conf_high`, which the script checks at 0.9
rather than the 0.670 now in use. The ship is still found: two predictions overlapped it almost
equally and a different one is credited (report 12). So the semantic records preserve the
detector's decisions on all but one ship in 8,173, not exactly. [reports 05, 12]

## 5b. Adaptive downlink policy — contributions folded under paper §IV-B

These are built and measured but have **no home in the accepted Phase-2 narrative**; they belong
under the paper's §IV-B *adaptive downlink policy*, and are among the strongest results.

* **Value-aware multi-pass scheduler.** The downlink is a knapsack that refills every orbit; items
  are ordered by value-per-byte, aged so nothing starves, and progressive products (ROI/wake, tiles)
  are **truncated** to fill a short pass. This is the mechanism that turns "what to drop" from a
  fixed rule into a per-pass decision. [reports 05–06; `sat7/scheduler.py`]
* **Queue-aware level of detail.** Buffer pressure — queued bytes against the capacity the satellite
  can read off its own ephemeris — sets how much detail the coastal encoder spends, generous while
  the link can drain and stingy once it cannot. No fixed coastal policy wins in more than one load
  regime; the queue-aware law is within **0.2 points** of the best fixed policy at every load. [report 07]
* **Optimality bounds.** An exact DP (brute-force verified) puts value-greedy **within 0.88 %** of
  optimal at operational scale, and perfect foresight over a whole day is worth **≤ 0.3 %** — so
  arrival prediction is deliberately *not* built. [reports 08–09]
* **Leakage-free split.** 19.1 % of Airbus tiles have a near-duplicate twin; a naive split leaks
  **1,617 groups** into test, ours leaks **0** — every recall number above rests on this. [report 01]
* **Dark-vessel / AIS prioritisation.** A no-AIS match raises an object's *priority* (value), kept
  decoupled from its payload after [report 10] caught a rule that inverted value on 13.6 % of ships.

## 6. Energy model (paper §V, eqs 8–19)

`E_proc = Σ P_k·T_k`, `E_comm = P_tx·D_tx/R_tx`, `ES = 1 − E_proposed/E_baseline`, built in
`wp17_energy_model.py` reading each stage time live with a sanity gate. [report 17]
**Every energy figure here is an estimate.** No power was measured in this project: P_cpu 28 W,
P_gpu 60 W, P_tx 15 W and P_isl 12 W are ASSUMPTIONS (swept); the stage times T_k are measured, on
a laptop; transmit times come from the simulated link. Bytes are multiplied by 8 before they are
divided by a rate in bit/s (`sat7/accounting.py`; checked against wp17 and wp19 in `wp29`).
* **ES_proc = 57.6 %** for the default all-CPU onboard build — a **power-free time ratio** (P_cpu
  cancels), so it holds on any processor; a lower bound, since the gate also drops empties the
  per-tile accounting doesn't credit.
* **Once pixels stop being sent, compute dominates the radio 5.5 : 1 at laptop powers (28 W CPU)**
  (45.6 kJ of processing against 8.4 kJ of transmission per day) — so at those powers energy
  optimisation belongs on the detector/gate, not the transmitter. The ratio follows the assumed
  processor power: over edge-class powers it runs from 0.20 : 1 to 37.9 : 1 (§6b).
* Versus a bent pipe **that sent every raw byte**, **ES_total ≈ 98.1 %** — ~54 kJ/day of
  compute+radio against ~2,851 kJ/day of raw transmission; ES_total stays 97–99 % across the power
  sweep. That bent pipe is hypothetical: it would need 92.6 days of contact per day of imaging. One
  limited to this link's 34.2 min of contact a day spends at most **30.8 kJ/day — less than ours** —
  and delivers 0.2 % of the ships. Read ES_total as energy per unit of imagery accounted for, not as
  a smaller daily energy bill.
* The default costs **one detector call per tile**, which is what the day simulation runs. A labelled
  SAHI variant (`wp17_energy_model.json` → `sahi_variant`: 1.5625 detector calls per tile; the gate
  still once per tile, which is how it is trained and run; fusion time 0 ms, an ASSUMPTION — it has
  never been timed) gives **70.0 kJ/day** of processing instead of 45.6, compute : radio **8.4 : 1**,
  and **ES_proc 47.0 %** instead of 57.6 % [estimate — assumed powers]. Were the gate run once per
  window instead, ES_proc would be 46.6 %.

## 6b. What is measured and what is assumed in every joule

| quantity | source | label |
|---|---|---|
| stage times T_k (detector, gate, classic filter) | measured on a laptop, read live from `wp1`, `wp3`, `wp11` | REAL (laptop) |
| decode time 0.71 ms, semantic stage 0.5 ms, fusion 0 ms | not in any results file / invented / never timed | ASSUMPTION |
| transmit time D_tx · 8 / R_tx | orbit and link simulation (`passes.csv`) | SIM |
| P_cpu, P_gpu, P_tx, P_isl | never measured | ASSUMPTION |
| **ES_proc = 57.6 %** | ratio of stage times; the power cancels | **REAL (laptop), power-free** |
| every figure in kJ, every proc : comm ratio, ES_total | assumed power × measured or simulated time | ESTIMATE |

**ES_proc is the one energy figure here that needs no power.** It is a ratio of laptop stage times.
It carries to another processor only if every stage slows by the same factor; an accelerator that
speeds up the networks but not the classic CV stage would change it.

**Edge-class sensitivity** (`wp17_energy_edge.json`) [ESTIMATE]. The same model with every onboard
stage on one processor of power P_proc. The powers are literature-typical classes — a 1–2 W VPU
(Myriad 2 / Myriad X), 5–15 W Jetson-class modules, a 10–30 W heterogeneous flight computer
(report 22) — and are **ASSUMPTIONS**, not measurements and not a hardware selection. An edge
processor is also slower by an unknown factor, so each power is paired with report 22's three
detector latencies, applied to every stage alike [TARGET]. P_tx = 15 W [ASSUMPTION], E_comm 8.4 kJ/day.

Processing energy, kJ/day (compute : radio in brackets):

| P_proc | 38.6 ms/tile (laptop, measured) | 100 ms/tile [TARGET] | 250 ms/tile [TARGET] |
|---|---|---|---|
| 1 W | 1.6 (0.20 : 1) | 4.2 (0.51 : 1) | 10.6 (1.26 : 1) |
| 2 W | 3.3 (0.39 : 1) | 8.4 (1.01 : 1) | 21.1 (2.53 : 1) |
| 5 W | 8.2 (0.98 : 1) | 21.1 (2.53 : 1) | 52.8 (6.32 : 1) |
| 10 W | 16.3 (1.95 : 1) | 42.2 (5.05 : 1) | 105.6 (12.6 : 1) |
| 15 W | 24.5 (2.93 : 1) | 63.3 (7.58 : 1) | 158.4 (19.0 : 1) |
| 30 W | 48.9 (5.85 : 1) | 126.7 (15.2 : 1) | 316.7 (37.9 : 1) |
| processor busy, share of the day | 1.9 % | 4.9 % | 12.2 % |

* **ES_proc is 57.6 % in every cell.**
* **"Compute dominates the radio 5.5 : 1" (§6) is a laptop-power statement.** Over this grid the
  ratio runs from 0.20 : 1 to 37.9 : 1; on a 1–2 W processor at laptop speed the radio dominates.
  Which side to optimise depends on the processor that is flown.
* **ES_total against the hypothetical bent pipe** runs from 0.79 (30 W, 250 ms, P_tx 8 W) to 0.997
  (1 W, laptop speed, P_tx 30 W) with P_tx swept over 8–30 W, wider than the 97–99 % of the
  laptop-class sweep in §6.
* P_isl (12 W, ASSUMPTION) is not in this grid; it is swept in `wp19` and `wp27`.

## 6c. From laptop measurement to a flown edge processor — the mapping behind "onboard = TARGET"

`T_target = T_laptop × s` and `E_target = P_board × T_target`. T_laptop is measured (on a laptop),
the slowdown s is an ASSUMPTION, and P_board is a literature power class (LIT) — so every output
is a **TARGET**, not a measurement. Default all-CPU ONNX build, one detector call per tile,
40,000 tiles/day [ASSUMPTION]. Source: `wp17_energy_model.json`, `wp17_energy_edge.json`.

**Step 1 — latency: measured stage time × assumed slowdown**

| stage | laptop time | label | s = 1× [ASSUMPTION] | s = 2.59× [ASSUMPTION] | s = 6.48× [ASSUMPTION] |
|---|---|---|---|---|---|
| decode / preprocess | 0.71 ms | ASSUMPTION (in no results file) | 0.71 ms | 1.84 ms | 4.60 ms |
| learned gate (47k params) | 0.94 ms | measured (laptop CPU) | 0.94 ms | 2.44 ms | 6.09 ms |
| detector (YOLOv8n @768, ONNX FP32) | 38.6 ms | measured (laptop CPU) | 38.6 ms | 100 ms | 250 ms |
| fusion | 0 ms | ASSUMPTION (never timed) | 0 ms | 0 ms | 0 ms |
| LoD encode + schedule | 0.5 ms | ASSUMPTION (invented) | 0.5 ms | 1.30 ms | 3.24 ms |
| **onboard latency per tile** | **40.75 ms** | measured + ASSUMPTION | **40.8 ms [TARGET]** | **105.6 ms [TARGET]** | **263.9 ms [TARGET]** |
| processor busy, share of the day | | | 1.9 % [TARGET] | 4.9 % [TARGET] | 12.2 % [TARGET] |

The slowdown is applied to every stage alike [ASSUMPTION]; s = 2.59× and 6.48× are the factors that
put the detector at report 22's 100 ms and 250 ms. Report 22's FLOPs envelope for a Myriad X
(~12.5 GFLOPs/tile on a >1 TOPS engine [LIT] → 30–150 ms/tile) sits inside this range.

**Step 2 — energy: TARGET latency × literature board power**

| board power class [LIT] | s [ASSUMPTION] | latency / tile [TARGET] | energy / tile [TARGET] | E_proc / day [TARGET] |
|---|---|---|---|---|
| 1 W — Myriad-class VPU, low end (Myriad 2 ~1–2 W, Myriad X ~1.5 W) | 1× | 40.8 ms | 41 mJ | 1.6 kJ |
| | 2.59× | 105.6 ms | 106 mJ | 4.2 kJ |
| | 6.48× | 263.9 ms | 264 mJ | 10.6 kJ |
| 2 W — Myriad-class VPU, high end | 1× | 40.8 ms | 82 mJ | 3.3 kJ |
| | 2.59× | 105.6 ms | 211 mJ | 8.4 kJ |
| | 6.48× | 263.9 ms | 528 mJ | 21.1 kJ |
| 10 W — heterogeneous flight computer, low end (Unibap iX5-100, 10–30 W) | 1× | 40.8 ms | 408 mJ | 16.3 kJ |
| | 2.59× | 105.6 ms | 1,056 mJ | 42.2 kJ |
| | 6.48× | 263.9 ms | 2,639 mJ | 105.6 kJ |
| *reference: laptop CPU, 28 W [ASSUMPTION — not measured]* | 1× | 40.75 ms (measured + ASSUMPTION) | 1,141 mJ [ESTIMATE] | 45.6 kJ [ESTIMATE] |

* **TARGET for a Myriad-class processor: 41–264 ms and 41–528 mJ per tile, 1.6–21.1 kJ/day, with the
  processor busy 2–12 % of the day.** The radio spends 8.4 kJ/day [P_tx 15 W ASSUMPTION, transmit
  time SIM], so on such a board compute and radio are the same order (0.20 : 1 to 2.53 : 1).
* **Latency.** Even the slowest case adds 0.26 s per tile, against a median delivery latency of
  4.5 h [SIM] that is all link wait. Onboard speed limits throughput (the busy share), not delivery
  time.
* **What the table does not carry.** The board power is a datasheet class, not the draw under this
  workload. The slowdown is not derived from any benchmark of our model on a VPU. A VPU typically
  runs INT8/FP16, whereas the times above are FP32 — and our one INT8 test (dynamic) lost
  2.2 points on small ships [report 02], so accuracy on the target is open as well as speed.
* **Conclusion, unchanged.** Onboard execution is **not demonstrated**. The pipeline was never run
  on a Myriad or any flight processor; no power was measured on any device. The table shows that
  the measured workload, under stated assumptions, fits the power and duty budget of hardware that
  already flies CNN inference (Φ-sat-1/2 [LIT]) — an argument for feasibility, to be settled only
  by a hardware-in-the-loop run [report 22 §5].

## 7. Latency & the relay (paper §V-D, §VII, eqs 25–29)

`T_total = T_inf + T_enc + T_comm + T_dec`; median delivery **4.5 h** [SIM].
**Convention.** The four stages are sequential for one item and are added. Every latency quoted in
this document is `T_comm`: from capture to the last byte at the ground station — the wait for a
contact plus the transmission, and on the relay path both legs (wait for an inter-satellite window,
ISL transmission, storage on the relay, relay-to-ground transmission). The other three are measured
[REAL, laptop]: `T_inf` 40 ms per tile on the CPU build (62 ms with SAHI's 1.56 calls), `T_enc`
0.14 ms for a ship tile to 9.1 ms for a coastal one, `T_dec` 0.04 ms. Together they are at most
**71 ms**, 4×10⁻⁶ of the median, so the hours do not change; a tile is finished long before the next
one is captured (one every 2.16 s). [`sat7/accounting.py`, `wp29`]
The optional relay uses
the paper's exact `J = λ_E·E + λ_T·T` path choice over real ISL windows: a complementary-coverage
relay (RAAN +90°) carries 66 % of items and **cuts worst-case latency 35.2 h → 10.9 h** and the
per-item median 7.9 h → 3.4 h [SIM], for **+52 % communication energy** [TARGET — joules await a real
P_isl/R_isl]. The 4.5 h is per *ship* (its first report); the per-item figures include thumbnails
and coastal tiles, which wait behind higher-value items on a link the day more than fills (§5) — the 35 h is
the last of them, sent in the final pass of the 36 h horizon. B4 is a **reroute**,
not extra capacity: recall and bytes are B3's verbatim. **Relay always costs more energy than direct
(~1.8×) — it is a latency buy, not a power saving.** [reports 19–21]

## 8. Optimality & robustness (strengthens the paper's §VI objective)

* **Greedy is 0.88 % from the exact optimum** at operational scale (exact DP, verified against
  brute force), 3.3 % on saturated small windows; FIFO loses 74 % on the same queues. [report 08]
* **Perfect foresight over a whole day is worth ≤ 0.3 %** → do **not** build arrival prediction. At
  40k tiles/day the schedule is within 0.005 % of the bound. [report 09]
* **Scheduling pays as the link fills.** At 40k the scheduler and FIFO deliver the same ships, but
  FIFO's median latency is 9.2 h against 4.5 h; the recall advantage appears once the link
  saturates — **1.88× at 80k and 3.59× at 160k** tiles/day. State the operating point.
* **Sensitivity:** every invented constant swept; value-greedy ≥ FIFO in all 80 settings, and
  **"ours ≥ the fair baseline" holds in 74 of 80** — all six failures are at 160k tiles/day
  (`thumb_value` ≥ 0.05, `conf_high` ≥ 0.9, `conf_low` ≤ 0.10), where the margin over the fair
  baseline is down to 1.3 points (§10 item 7). **Cloud fraction dominates everything** (recall
  range 0.42) — quote every recall with its cloud fraction. [report 11]

## 9. Ablations (paper §VIII-G)

| ablation | result | evidence |
|---|---|---|
| without SAHI | B1 0.339 vs B2 0.717 on the same scenes: slicing is worth 38 points against one call on the whole scene. Against a plain tile-by-tile pass it is not (0.717 vs 0.730, §10 item 3) | `wp26`; reports 13, 16 |
| without adaptive downlink | fixed coastal policy loses to queue-aware in every regime but one | report 07 |
| without ROI | LoD-rung ablations; the ROI rung earns its bytes | report 06 |
| without relay | direct-only = B3 exactly (sanity gate a) | report 21 |
| without tile overlap / without fusion | measured on 24 swaths, 716 ships (`wp24`): SAHI recall 0.772 / precision 0.765; **no overlap** 0.726 / 0.764; **no fusion** 0.782 / 0.495 (duplicates). A tile-aligned grid without overlap scores 0.774 / 0.794 — see §10 item 3 | `wp24`; reports 16, 23 |

## 9b. The six ablations and the trade-off curves, from one command (`wp30_paper_ablations.py`)

`python scripts/wp30_paper_ablations.py` writes `wp30_ablations.json`, `wp30_curves.csv` and
`wp30_curves.png`. The three detection rows are read from the files §9 already cites; the three
downlink rows use the paper's Table I levels only (pure P0–P3, `sat7.priority`, §2b): one simulated
40,000-tile day (seed 0), 13,033 detections sent, 25 % link share, 15 % cloud. The script stops
with an error unless that day reproduces the P0–P3 row of `wp23_semantic_compare.json`.

| ablation (paper §VIII-G) | full → without | label | source |
|---|---|---|---|
| without SAHI | recall 0.717 → 0.339, precision 0.696 → 0.681 (small ships 0.505 → 0.046); 150 scenes, 1,164 ships | REAL | `wp26_b0_b4.json` B2 vs B1 |
| without tile overlap | recall 0.772 → 0.726, precision 0.765 → 0.764; 24 swaths, 716 ships | REAL | `wp24_detection_eval.json` sahi_B2 vs no_overlap |
| without detection fusion | recall 0.772 → 0.782, precision 0.765 → 0.495 (1,270 → 2,426 boxes: duplicates) | REAL | `wp24_detection_eval.json` sahi_B2 vs sahi_no_fusion |
| without adaptive semantic downlink (one fixed level) | adaptive: 90.5 MB, information preservation 0.761. Fixed P1: 0.6 MB, 0.500. Fixed P2: 19.8 MB, 0.800. Fixed P3: 166.4 MB, 1.000, worst-case item 35.2 h instead of 11.6 h. Recall 0.617 in all four | SIM-over-REAL | `wp30_ablations.json` |
| without ROI transmission (cap at P1) | 90.5 → 0.6 MB, information preservation 0.761 → 0.500, recall unchanged at 0.617. The same day as fixed P1 by construction | SIM-over-REAL | `wp30_ablations.json` |
| without optional relay | per-ship median latency 1.9 → 4.5 h, worst-case item 5.9 → 11.6 h; transmit energy 6.2 → 4.2 kJ. Recall and bytes unchanged; the relay carried 44 % of items | latency SIM, energy ASSUMPTION powers | `wp30_ablations.json` (`sat7.comms`) |

Information preservation here is the joint program's (`sat7.joint`): the weighted credit of the
level that reached the ground, P1 0.5 / P2 0.8 / P3 1.0, over the ships the detector fired on. The
credits are an **ASSUMPTION**. The relay row uses the capacity-aware simulator (`sat7.comms`); §2b's
B4 figures for the same day (11.6 → 6.2 h, 4.3 → 5.8 kJ) come from `wp18`'s window estimate.

**Trade-off curves (paper eqs 32–35)**, `wp30_curves.csv`, column `curve` [SIM-over-REAL]:

* **D_tx vs recall.** Recall is set by the detector, not the link: on a starved link it climbs
  0.089 → 0.439 between 0.08 and 0.42 MB, reaches 0.617 at 0.84 MB and stays there to 166 MB,
  because the P1 records are sent first and total 0.6 MB. Moving the detector's cut is what moves
  recall: 0.262 at 52 MB (cut 0.80), 0.617 at 90.5 MB (0.25), 0.733 at 105.7 MB (0.05).
* **D_tx vs information preservation.** 0.50 at 0.6 MB (metadata only), 0.80 at 19.8 MB (a ROI for
  every detection), 1.00 at 166 MB (context for every detection). The adaptive policy sits at 0.761
  for 90.5 MB: most of its bytes are coastal context tiles, which this credit table rewards little.
  **On this metric a fixed P2 does better for a fifth of the bytes**; what the adaptive policy buys
  with the rest is the coastal context, which neither recall nor this credit scores.
* **E_total vs T_total, direct vs relay.** Adaptive policy: 49.8 kJ at a 4.6 h per-item median
  direct, 51.8 kJ at 2.0 h with the relay. Fixed P3: 53.0 kJ / 5.1 h direct, 56.8 kJ / 2.2 h relay.
  45.6 kJ of every total is onboard processing, equal in all variants; powers are ASSUMPTIONS.

## 10. Limitations (paper §X, extended honestly)

1. **Onboard = TARGET, not a prototype.** All timings are a laptop. But the work (3.15 M-param
   detector + 47 k gate) is the class our baseline **Φ-sat-2 already runs onboard** on a ~1 W Myriad 2
   VPU [LIT]; at 40k tiles/day it is a **few-% duty cycle even on a VPU 6× slower** than our laptop,
   and ES_proc is hardware-independent. A prototype still owes wall-clock + quantised accuracy on a
   real board, thermal, and rad-tolerance. [report 22]
2. **Dataset.** Airbus optical + synthetic scenes and swaths stitched from tiles for B1/B2 (§1). A
   large-scene optical set would strengthen B2.
3. **SAHI is selective, not blanket — and here is the negative result.** A cut ship is usually still
   detected, so blanket SAHI is a poor trade; SAHI costs 1.56× (down from 2.25×) and the result is
   window-size dependent [reports 13, 16]. **On our stitched scenes SAHI + fusion (recall 0.717) does
   not beat a plain tile-by-tile pass with no overlap (0.730)**, while making 1.56× the detector calls
   [`wp26`, same 150 scenes]. The reason is the seams: the scenes are stitched from independent
   tiles, so no ship ever crosses a seam and the overlap has nothing to recover; overlapping windows
   only re-detect the same ships across two unrelated tiles. B2's gain is measured against the
   whole-scene call (B1), not against plain tiling; whether overlap beats plain tiling needs real
   large scenes.
4. **Cloud fraction** moves recall more than any other quantity and the Airbus set (0.4 % cloud)
   cannot settle it — a climatology is needed [LIT]. Always state the assumed fraction.
5. **P_tx** is the weakest number (invented); it only enters E_comm, which is 5.5× smaller than E_proc.
6. **Byte sizes are partly modeled.** Coastal tiles are measured; thumbnails (flat 1,000 B) and ship
   crops (a fitted size model) are not, and they are 24 % of the bytes. Measuring the coastal tile
   alone moved the headline from 557× to 341×; the same exercise on the rest would move it again,
   by less. One day now offers 130 % of what the link share sustains per 24 h (§5), so "the
   downlink is not the bottleneck" holds for recall — and only because truncated products still
   count as delivered — not for bytes, and not for the latency of low-value items.
7. **Our lead over the fair baseline is not robust under congestion — a second negative result.**
   Of the 80 swept settings (`wp10_sensitivity.csv`), "value-greedy ≥ FIFO" holds in all 80 and
   "ours ≥ the fair Phi-sat-2-style baseline" in **74**. All six failures are at 160k tiles/day,
   where the lead is only **1.3 points** to begin with (6.3 points at 40k, where nothing fails):
   `thumb_value` 0.05 (−0.5 points) and 0.10 (−1.9), `conf_high` 0.90 (−0.02) and 0.99 (−0.4), `conf_low` 0.05 (−1.6) and 0.10 (−0.8). Each makes the encoder spend more bytes — on thumbnails, on crops for confident ships, or
   on low-confidence detections — which a saturated link cannot afford. The 6.3-point claim is a
   nominal-load claim.

## 11. Conclusion — the central question, answered

The paper asked whether **less transmitted data can preserve more useful mission information.**
Measured: **yes.** A 341× data cut (coastal tiles measured; thumbnails and crops modeled) delivers
**100 % of what the satellite still knew** at nominal
load and **+6.3 points over a fair baseline**, the schedule is within **0.9 %** of optimal, and the
energy saved by not sending pixels dwarfs the extra compute (ES_total ≈ 98 %). The remaining loss is
**the detector on small ships**, not the downlink — which is where Phase-4 effort belongs. What is
not yet proven, and we say so, is the flight-hardware wall-clock and the quantised accuracy on a real
onboard board: this is a **demo of the idea, measured end to end on real data — not a flown prototype.**

## 12. Validation checklist — `wp29_validation.py`

38 checks on the committed results, none failing (`code/results/wp29_validation.json`; 36 unit
tests in `code/tests/test_validation.py`). PARTIAL means every check holds but a stated gap remains.

| # | question | status | what was checked · what remains |
|---|---|---|---|
| 6 | data reduction uses actual, consistently defined sizes | **PARTIAL** | formula, raw baseline (H×W×3 per non-cloud tile) and the coastal size table all reproduce · 24 % of the bytes are still model sizes; the day is simulated alone |
| 7 | detection metrics rest on valid ground truth | **PASS** | all 8,173 labelled test ships, leak-free split, P / R / F1 / AP, one ground truth for B1–B4 · delivered recall uses the catalogue's older matching rule (0.725 vs 0.766 before cloud) |
| 8 | energy assumptions are documented | **PASS** | wp17 and wp19 reproduce from `P × bytes × 8 / rate` and `Σ P_k·T_k`; every power is labelled ASSUMPTION and swept · no power was measured |
| 9 | latency includes the required stages | **PASS** | simulated latency = wait + transmission; the three other stages are measured and add ≤ 71 ms · laptop timings |
| 10 | relay energy and latency cover both links | **PARTIAL** | ISL and relay-to-ground legs both carry time and energy in wp19 and wp27 · the B4 row's relay latency is a window estimate |
| 11 | compared things share inputs and assumptions | **PARTIAL** | B1/B2, ours/FIFO/baselines, direct/relay, LoD/P0–P3 each share one input · ES_total's bent pipe is not transmittable; B4 is asymmetric; B1/B2 and B3/B4 are different inputs |
| 12 | every number traces to evidence | **PARTIAL** | 48 headline numbers × the ten judge-facing documents all match their results files · "100 %" is a single-day claim and rests on truncated products counting as delivered |

---
_Sources: `demo/PAPER_COVERAGE.md` (paper→evidence map), `reports/01–22`, `code/results/*`,
`docs/STATUS.md`. Superseded figures to avoid (`docs/START_HERE.md §7`): 442×/511×→557×,
97.1 %→0.685, +16.7→+6.3, 2.7×→2.26×, "INT8 lossless"→false, coastal q40 "free"→costs 4 pts.
Superseded on 10 Oct 2026: 557× (all-modeled sizes)→**341×** (coastal tiles measured); 108 MB→176 MB;
B1 0.782 / B2 0.745 (small samples, different inputs)→**0.339 / 0.717** (same scenes); 2.26×→3.59×;
0.251 %→0.88 %; 8.9 : 1→5.5 : 1; relay 11.6→6.2 h→35.2→10.9 h. `docs/` and `reports/05–22` still
carry the earlier figures._
