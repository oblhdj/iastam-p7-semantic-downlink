# IASTAM 6.0 — Problem 7 — project state (24 Sept 2026)

## ▶ RESUME HERE (last worked 24 Sept ~16:00)
All work stopped cleanly; **86 tests pass**; no processes left running. Items 1–4 below are all
**done** — the engineering backlog is empty and the remaining risk is the 26 Sept paper/video.
See **NEXT STEP** for what is actually left.

1. ~~Re-run 5 stale result sets~~ — **DONE 24 Sept 15:29–15:31**, all 6 commands, 0 failures
   (`results/rerun_post_wp9.log`). Every number improved or held; **no conclusion changed**:
   | result | was (pre-WP9) | now |
   |---|---|---|
   | optimality gap, saturated small windows | 8.95% mean | **4.62%** |
   | optimality gap, operational scale | 0.09% | **0.069%** |
   | joint bound, total gap (worst load) | 12.7% | **9.6%** |
   | price of not seeing the future | 0.66% | **0.4%** |
   | WP8 queue-aware regret (worst / mean) | 0.0101 / 0.0037 | **0.0075 / 0.0018** |
   | dataset-mix: ours / FIFO | 0.743 / 0.280 = 2.65× | **0.769 / 0.349 = 2.20×** |
   ⚠ **The "2.7× FIFO" claim is superseded — use 2.2×** (FIFO improved too). The four report
   `.md` files carry a REFRESHED banner with the new figures, but their inner tables still show
   the old numbers; the `.csv` files are authoritative.

2. ~~Finish the INT8 run~~ — **DONE 24 Sept 15:47.** Write-up: `code/results/wp1_int8_report.md`.
   Corrected numbers (the old 593 vs 82 ms was measured under CPU contention):
   **INT8 is 8.5× SLOWER on CPU** (328.8 vs 38.6 ms/tile) for 3.6× less storage.
   ⚠ **It is NOT lossless** — the 40-tile pilot was underpowered. Paired on 970 ships:
   **−1.44 pts overall, −2.21 small, −1.29 medium, exactly 0.00 large** (16 lost, 2 gained).
   The FMEA row "INT8 degradation → small ships lost" is **CONFIRMED**, monotone in smallness.
   **ONNX FP32 is the onboard build**: bit-identical recall to pytorch and 1.21× faster on CPU
   (38.6 vs 46.6 ms/tile). Drop dynamic INT8; the "1.5–3.3× faster" [LIT] is TensorRT/static QDQ.
3. ~~WP3 learned gate~~ — **DONE 24 Sept 15:53.** Write-up: `code/results/wp3_gate_report.md`.
   ⚠ **It does NOT need Kaggle and the 45–85 min estimate was wrong: it trains in 2m21s**
   (30–46 s/epoch). The estimate assumed a cold JPEG cache; warm, the loader does ~1,400 tiles/s.
   47k parameters. Threshold picked on **val**, quoted on **test** (gap −0.23 pts):
   **98.8% of ships kept while skipping 40.2% of empty tiles**, vs the classic filter's 55.5% at
   41.9% → **+43.3 points of safety at matched compute**. Dominates it at every operating point.
   ⚠ **The script overstated its own cost ~80×**: `infer_ms_per_tile` timed the whole eval loop
   (JPEG decode included) at 4.65 ms. True pure-forward cost is **0.052 ms/tile** on GPU,
   0.94 on CPU. Script patched, `wp3_gate.json` corrected. Had 4.65 been believed the gate would
   have looked like a net loss (break-even is 0.904 ms on GPU).
   Net compute saving **9.2% GPU / 7.2% CPU** — capped because only **20.4% of tiles are empty**
   in this ship-centric dataset. That is a floor, not the operational figure; a real swath is
   mostly empty ocean. The 1-epoch verification output is parked as `wp3_gate*_SMOKETEST_ONLY_1epoch.*`
   (it showed the gate dropping 0.4% of empties — do not cite it).
4. ~~WP2 calibration~~ — **DONE 24 Sept 15:50.** Write-up: `code/results/wp2_calibration_report.md`.
   Val dumped first (17,201 boxes, 8,173 ships, recall 0.871; verified **0 images overlapping test**).
   The detector is **underconfident**: says 0.352, right 0.408 of the time. **Isotonic cuts ECE
   8.9×** (0.0878 → 0.0099), Platt 6.1×; same Brier, so it is pure calibration — a monotone map
   changes no ranking, so **no WP5–WP10 conclusion is at risk**.
   **`conf_high` should be 0.670, not 0.9.** Raw 0.9 was far stricter than "90% certain", which is
   why the cheap 40 B rung fired on only 3.1% of predictions; cut where 0.9 means 0.9 and it
   serves **21.7%** — 7× more. Already priced by WP10's existing sweep (0.670 sits just under its
   0.70 sample, trend monotone): at 160k tiles/day **+1.4 pts recall, −8.1% bytes, −0.13 h
   latency**, margin over the fair baseline +2.6 → +4.1 pts. At 40k, −7.6% bytes for free.
   Honest: this *accepts more risk than today* — the point is the rung now means what it says.
   **Still unmeasured:** `conf_low` 0.4 → 0.329 (recovers 5.3% of predictions we currently
   discard). `wp6_simulate_real.py` has no `--conf-low` flag, so sweep it as WP10 swept conf_high.
5. Full audit of everything missing: **`ARCHITECTURE.md`** (§A–F, audited 24 Sept 14:30).

⚠ Interim paper + 2-min video due **26 Sept**. Ten result documents exist; none are in the
paper, and the PDFs still carry the synthetic 97.1% and the unfair baseline. The video script
does not exist. This was deprioritised by explicit instruction, not by oversight.


**Challenge**: Track 4, Problem 7 "Transmitting Information Rather Than Raw Data".
**Deadline**: interim paper + 2-min video **26 Sept**; mid-review needs ≥60/100. Phase 3: 1–14 Oct.

## Idea in one line
Satellite sends *information about ships*, not pictures: (1) **confidence-aware level of detail**
— more bytes when the model is less sure — and (2) a **value-aware multi-pass scheduler**.

> Pipeline view of what is built and what is missing: **`ARCHITECTURE.md`**.

## Where things stand
| Item | Status | Key numbers (all measured unless noted) |
|---|---|---|
| Data integrity (WP0) | **done** | 53,195 tiles, 6,282 proven duplicate pairs, 19.1% of tiles have a twin, naive split would leak 1,617 groups, ours 0 |
| Detector (WP1) | **trained + exported** | test mAP50 **0.804**, P 0.817, R 0.729; recall by size small(<32px) **0.739 at conf>=0.05** (⚠ at the operational 0.25 cut it is 0.50–0.61 — never quote 0.739 next to "threshold 0.25"), medium 0.958, large 0.994; 10.7 ms/tile GPU. ONNX FP32 lossless, 38.6 ms/tile CPU; dynamic INT8 rejected (8.5× slower, −2.2 pts on small ships) |
| Payload sizes (WP4) | **measured** | chip 899 B, ROI+wake 2.5 kB, tile 42.9 kB, thumbnail 997 B (were assumed 2 kB / 30 kB / 80 kB) |
| Scheduler (WP5/**WP6**) | **now on REAL detections** | **557×** less data than raw (60,124 MB offered → 108.0 MB sent), 68.5% of real ships **at an assumed 15% cloud** (band 0.40–0.82) = **100% of what the satellite still knows**; at 160k ours 66.2% vs FIFO 29.3% (**2.26×**) |
| Byte budget (**WP7**) | **measured + two changes adopted** | 86% of the downlink was coastal tiles (62%) + thumbnails (24%), ship info only 13%. Coastal q60→q40 and gated thumbnails: −19% bytes, **zero recall cost**. Adaptive coastal detail wins under congestion: 0.666 vs 0.638 at **62% of the bytes** |
| Coastal detail (**WP8**) | **queue-aware, held out** | encoder now reads the buffer: no fixed coastal policy is best in more than one regime; ours is within **1.1 points** of the best at every load. Worst-case regret **0.0058 vs 0.0123** (refreshed 16:32); validated on 2 untuned link budgets |
| Optimality gap (**WP5**) | **measured at last** | the plan ticked this as done but no solver existed. Exact DP (`sat7/optimum.py`, verified vs brute force): greedy is **0.251%** from optimal at operational scale, **6.67%** mean / 54.8% worst on saturated small windows (refreshed 16:33). ⚠ Grew from 0.069%/4.62% because `conf_high` 0.670 made items lumpier — the price of the byte saving. FIFO 67.3%. |
| Pre-filter / gate (WP3) | **learned gate trained, wins** | classic filter failed (0.645 recall dropping 17.7% of empties). 47k-param gate: **98.8% of ships at 40.2% of empties dropped** (threshold from val), **+43.3 pts** at matched compute; 0.052 ms/tile GPU. Net compute saving 9.2% GPU / 7.2% CPU, capped by only 20.4% of tiles being empty |
| Orbit/link | done | Sfax: 5 passes/day, 34 min contact, 11.4 h max gap |
| Calibration (WP2) | **done** | underconfident (0.352 said vs 0.408 actual); isotonic cuts ECE **8.9×** to 0.0099; `conf_high` 0.9 → **0.670** is strictly better (+1.4 pts recall, −8.1% bytes at 160k) |
| Integration (**WP11**) | **done — replay validated** | real JPEGs through prefilter→gate→detector→LoD→scheduler→ground. Catalogue cross-check exact on every structural field, **0 decisions changed**. ⚠ Found: **52% of reported ship loss = 12 real cloud ships resampled 287×**; classic prefilter **51.1 ms/tile = 3.8× the detector** |
| SAHI / seams | not started | SAHI: 20% overlap = 1.51× compute; our border-trigger ≈1.01×; large ships straddle seams 53–84% |

## WP6 done (24 Sept) — real detections now drive the scheduler
`scripts/wp6_build_catalogue.py` → `wp6_fit_size_model.py` → `wp6_simulate_real.py`.
Full write-up with every number: **`code/results/wp6_report.md`**. What the real data changed:

* **The downlink is no longer the bottleneck.** At 40k tiles/day we deliver 100% of what the onboard
  software still knows; the gap from 1.0 to 0.685 is the detector + cloud gate. Scheduling only pays
  once the link saturates (80k+ tiles/day, or the dense-scene mix) — at 160k it is **2.4× FIFO**
  with the WP7 defaults. (The `--mix dataset` variant now gives **2.2×**, refreshed 24 Sept 15:31.)
* The interim 97.1% came from the synthetic Beta(5,2) confidences. Real, same setup: **0.685**.
* Real confidence: median 0.68, only **7.5% above 0.9** → the cheap 40 B L0 rung almost never fires.
  The LoD thresholds (0.4 / 0.9) were cut for a distribution the detector does not produce.
* **Coastal tiles hold 30% of all ships**, recall there 0.784 vs 0.895 on open sea → real
  justification for downlinking whole coastal tiles.
* Threshold sweep: cut-off 0.25 → 0.05 gives **+8.8 points of ships for +8% bytes** (the budget is
  coastal tiles and thumbnails, not chips) but 5.7× more false alarms. A ground-segment decision.
* Payload is now per ship: bytes = 137.5·px^0.57 (L1), 43.3·px^1.14 (L2), fitted on the 551 WP4 crops.
* Ablations: conclusions hold with WP4 medians instead of the power law; 3 resampled days spread
  [0.680, 0.690].

## WP7 done (24 Sept) — the byte budget. Full write-up: `code/results/wp7_report.md`
`scripts/wp7_measure_cheap_products.py` → `wp7_budget_sweep.py`.

* **86% of our "semantic downlink" was not semantic**: coastal tiles 62%, thumbnails 24%, ship
  information 13%. That reframes where optimisation effort belongs.
* **Adopted** (measured, no recall cost): coastal tiles q60→q40 (−24% of the biggest item,
  PSNR 40.3→37.9 dB) and thumbnail gating (skip only where the classic pre-filter sees no
  candidate AND the detector fired nothing, 2% audit sample kept). Together −19% bytes.
* Thumbnails have a **~700 B floor** (JPEG headers): 48 px saves 26% but loses 23% of visible
  ships, while **128 px q30 costs the same as 96 px q60 and shows more** — resolution beats
  quality here. So the lever is sending fewer, not smaller.
* **New: adaptive coastal detail.** Unconfirmed candidates (classic stage saw it, network did
  not) predict missed ships: the 50% of coastal tiles with 11+ hold **75% of them**. Send the
  cheap ROI mosaic normally, escalate to the whole tile above the threshold. Under congestion
  this beats every fixed policy on both axes: 0.666 vs 0.638 recall at 62% of the bytes, and
  shorter latency. Not the default — at nominal load it costs 1.6 points for bytes we don't need.

## ⚠ Honest correction to our own claim (WP7 §4)
The interim report's self-criticism weakness #2 was real: the Phi-sat-2 baseline ignored coastal
ships **by construction**. Fixed (`encode_fixed_patch(coastal=True)`, now the default, and both
variants are reported). The fair baseline gets **0.622**, not 0.518. So our advantage is
**+6.3 points of recall for ~7× the bytes**, not +16.7. Say it this way; a reviewer will check.

## WP8 done (24 Sept) — the encoder and the scheduler became one loop
`scripts/wp8_queue_aware.py`. Write-up: **`code/results/wp8_report.md`**.

* `simulate_online` encodes each tile *knowing the current buffer*, the way the satellite
  actually works (`encode_lod` and it share one `encode_tile`; a test pins them identical when
  the coastal policy is fixed, so WP8 measures the idea and not the refactor).
* `coast_mode="queue"`: pressure = queued bytes / link capacity in the next 12 h (both onboard
  quantities) slides the escalation threshold between `esc_min` and `esc_max`.
* **No fixed coastal policy is right in more than one regime** — the best one changes three
  times between 20k and 320k tiles/day. Queue-aware is within **1.1 points** of the best at
  every load without being told the load. Worst-case regret **0.0075 vs 0.0168** for the best
  fixed policy; mean regret **0.0018 vs 0.0115** (refreshed 15:31; was 0.011/0.022 and 0.004/0.013).
* **Held out**: the 4 fitted parameters were re-run unchanged at link share 0.15 and 0.40
  (never tuned on). Lowest mean regret in all three, lowest worst case in two, tied in one.
* Honest size of the claim: ~1 point of recall over a *well-chosen* fixed threshold. It is a
  **robustness** result, not a higher peak. Say it that way.

## WP5 optimality gap done (24 Sept) — a promised deliverable that didn't exist
`scripts/wp5_optimality_gap.py` + `sat7/optimum.py`. Write-up: **`code/results/wp5_optimality_gap_report.md`**.
The plan ticks "optimality gap on small passes" under *must have*; there was no solver, only prose.

* Exact DP for the knapsack-with-concave-divisible-class, **verified against brute-force
  enumeration** on random instances and bounded above by a fractional relaxation on every run.
* **Greedy is essentially optimal at our scale** (**0.069%** mean, refreshed): ~13,800 items of ~1 kB in a
  25–60 MB window is nearly a fractional knapsack, and greedy is exactly optimal for those.
* **The gap is real only on lumpy windows**: **4.62%** mean, 34.7% worst over saturated small
  instances (refreshed 15:29; the 8.95% below is pre-WP9). FIFO loses 72.9% on the same queues.
* Convergence check: the gap grows with truncation resolution and settles by 16 steps — coarse
  settings *understate* it. The gap lives in **truncation choice**: the optimum cuts several
  progressive items well, we only truncate whichever one straddles the end of the pass.
  → concrete Phase-3 improvement, and the only place the *scheduler* (not the detector) loses.
* Found and fixed a bug in our own validation: `a <= b+tol <= c+tol` chains in Python and
  cancels the tolerance, flagging 1e-16 rounding as violations.

## WP5b joint/whole-day bound done (24 Sept) — the hardest open item
`scripts/wp5_joint_bound.py` + `sat7/optimum.py`. Write-up: **`code/results/wp5_joint_bound_report.md`**.
Closes the limitation WP5 flagged itself ("the per-window gap does not bound the end-to-end
loss") and the plan's unstarted nice-to-have "offline LP upper bound".

* Joint scheduling is NP-hard (multi-knapsack + release times), so it is **bracketed**:
  `online <= clairvoyant <= true optimum <= relaxation bound`. The bound is verified against
  brute-force enumeration of every item→window assignment on 120 random instances.
* **Up to 40k tiles/day our whole-day schedule is provably optimal** — all three quantities
  equal to float precision. At 320k the true optimum is between 140,554 and 159,721, so we are
  between **0.75% and 12.7%** below best achievable. FIFO is 78.7% below.
* **Clairvoyance is worth at most 0.7%.** Perfect foresight over the entire day gains 0.13% at
  160k and 0.66% at 320k. **So do not build arrival prediction / traffic priors** — against a
  perfect oracle the ceiling is two thirds of one percent. The remaining loss is greediness
  under lumpiness, which WP5 already localised to *truncation choice*.
* Also tightened the single-window bound: each item's credit now saturates at its own value
  (it could previously earn several times it). Slack 59% → 29% on mixed instances, 0% on whole.

## WP9 done (24 Sept) — the FMEA table is now executable, and it failed in places
`tests/test_failure_modes.py` (14 fault-injection tests). Write-up: **`code/results/wp9_failure_modes_report.md`**.
The report's 8 failure modes were all prose; none of our tests injected a fault. Now they do.

* **⚠ "AIS gap = priority, not accusation" was FALSE for large vessels — now FIXED.**
  Marking a ship dark changed *two* things: value ×5 **and** payload L1→L2 (px^0.57 → px^1.14).
  Above **128 px** the payload outgrew the weight, so a dark vessel ranked **below** the same
  ship with a clean AIS match — on **13.6% of real ships (18.7% of detected)**, the biggest ones.
  **Fix (`LoDConfig.dark_mode="decoupled"`, now the default):** a dark vessel gets the *same*
  base chip as anyone else carrying the 5× weight, and the ROI+wake becomes a separate
  lower-priority follow-up. The ranking ratio is now exactly `dark_weight` at every size, by
  construction. Old behaviour kept as `dark_mode="replace"` so the defect stays reproducible.
  **Strictly better, not a trade** — at 320k tiles/day: large-dark recall **+24.1 pts**
  (0.686→0.927), all-dark +5.2, **overall ship recall +1.8**, dark latency **−2.0 h**, bytes +1.2%.
  (Bumping `dark_weight` to 9.5 instead would have cost −2.9 pts of overall recall.)
  WP6 headline: 511× data reduction (was 517×), ship recall unchanged at 0.685.
* "Thumbnail + **raw ring buffer**, ground re-request": the thumbnail exists but carries **no
  ship ids** (so it never counts as delivery), and **there is no ring buffer in the code**.
* "Audit sampling: 1% random **raw tiles**": implemented as a 2% sample of *thumbnails* on
  gated-empty tiles. Fires, but is not what the report says.
* "Watchdog + safe mode": safe mode works; **no watchdog exists**.
* Holding as claimed: pass shortfall (progressive truncation), storage full (evicts lowest
  value-per-byte, dark vessels survive), lost pass recovered by the next one when buffering.

## WP10 done (24 Sept) — sensitivity of every invented constant
`scripts/wp10_sensitivity.py`. Write-up: **`code/results/wp10_sensitivity_report.md`**.
Seven unmeasured constants swept one at a time, at 40k (nothing binds) and 160k (link binds).

* **Conclusions hold in 69 of 70 settings.** "value-greedy ≥ FIFO" never fails. "ours ≥ fair
  fixed-patch" fails once: `thumb_value = 0.1` (10× baseline) → we fall 0.006 behind, because
  thumbnails outnumber ship chips 4:1 and start crowding out the cargo.
  **Constraint to state in the paper: `thumb_value` must stay below ~0.05.** Baseline 0.01 has 5× margin.
* **Cloud fraction dominates absolute recall** (range **0.42**, from 0.82 at 0% to 0.40 at 50%)
  — but it is a *denominator* effect: our margin over the fair baseline stays positive at every
  level, and greedy still beats FIFO everywhere. **Quote every recall number with the assumed
  cloud fraction attached.** The Airbus set can't settle it (0.4% cloud); needs a climatology [LIT].
* **`dark_weight` is now inert for overall recall** (range 7e-5 across 1→20) and dark recall
  **saturates at 5.0** — so the baseline sits on the plateau. Direct confirmation the WP9
  decoupling fix worked: before it, raising the weight cost 2.9 points of overall recall.
* **`aging_per_hour` does essentially nothing** (range 2.4e-4, below the ±0.005 seed noise).
  Nothing is starved because the LoD payload is small relative to a pass. **Either drop it or
  stop claiming it does anything** — a mechanism that provably does nothing is worse than none.
* Ranking at 160k: cloud 0.365 ≫ thumb_value 0.033 > p_dark 0.020 > conf_high 0.018 >
  dark_wake_value 0.011 ≫ aging 0.0002 > dark_weight 0.00007.

## NEXT STEP (highest value)
Updated 24 Sept ~17:00. The three "cheap wins" are **done and measured**, not just recommended.

**Done today (beyond WP1 INT8 / WP2 / WP3 / WP11):**
* **`conf_high` 0.9 → 0.670 ADOPTED** as the `LoDConfig` default. Full re-run of WP6/7/8/5/5b/10/11
  (`results/rerun_conf_high_670.log`). At 40k: recall unchanged **0.685**, bytes **117.9 → 108.0 MB
  (−8.4%)**, data reduction **511× → 557×**. At 160k: **+1.5 pts** recall, margin over the fair
  baseline **+2.6 → +4.1 pts**. Honest cost: the optimality gap grew (0.069% → 0.251% at scale)
  because the cheap rung makes items lumpier, and the FIFO ratio slipped 2.4× → 2.26× because
  cheaper encoding helps FIFO too.
* **`conf_low` swept for the first time — it is the 2nd most influential constant** (range 0.157).
  ⚠ **WP2's "lower it to 0.329" advice was WRONG and is withdrawn**: every script ties `conf_low`
  to the 0.25 detection threshold, not the 0.4 dataclass default, so 0.329 is *stricter* and
  measured at **−3.2 points** of recall. Read the call site, not the dataclass.
* **Cloud stated as a band**: 0.68 at the assumed 15%, **0.40–0.82 across 0–50%**. Never quote bare.
* **Gate test added** (`tests/test_gate.py`, 10 tests) — the last untested artefact. Torch tests
  skip cleanly in `.venv` and run in `.venv312` (pytest installed there).
* **Gate scores for all 5,320 tiles** (`wp3_gate_scores_test.csv`), ready to re-key WP7/WP8.

**Left, in order:**
1. ⚠ **The paper and the video — the only hard deadline (26 Sept).** Now 14 result documents,
   none in the paper. This is the binding constraint; the engineering backlog is not.
2. **Re-key WP7 thumbnail gating and WP8's control law on the learned gate** — both still use the
   classic filter's 0.645-recall signal; the gate is 0.988 and 80× cheaper. Scores are ready.
3. **q30 coastal tiles** — needs the detector-on-recompressed-tiles check. −18% of coastal bytes.
4. **2-D sensitivity** (`scripts/wp10b_interaction.py`, written, not yet run): cloud × thumb_value.
5. ARCHITECTURE §E leftovers: SAHI/seams, **flight hardware** (all timings are a laptop RTX 5060),
   multi-station, bursty arrivals.
6. Then small-ship recall — still the largest single loss, and INT8 made it worse.

## Environment — FIXED 24 Sept (the old notes here were wrong)
* **`.venv312` is not corrupted.** It runs torch/ultralytics directly, CUDA and all. The
  PYTHONPATH + uv-interpreter workaround in the old notes was never needed — just run:
  ```bash
  cd "C:\Users\Mega-PC\Desktop\IASTAM_Problem7\code"
  .venv312/Scripts/python.exe -W ignore scripts\<script>.py
  ```
  Verified: torch 2.11.0+cu128 (CUDA True, RTX 5060), ultralytics 8.4.160, Python 3.12.14.
* **The real problem was no `pip`** in that venv, which is why `onnx` could never be installed
  and INT8 export "failed". Fixed with `python -m ensurepip --default-pip`; then
  `python -m pip install --timeout 120 --retries 5 onnx onnxruntime onnxslim`
  (the default pip timeout is too short for the 14 MB onnxruntime wheel — it will time out
  mid-download and look like a proxy problem; it is not, just retry with a longer timeout).
  Now installed: onnx 1.23.0, onnxruntime 1.30.0, onnxslim 0.1.96.
* `.venv` (Python 3.14) still has **no working torch** — use it for everything non-torch
  (all of WP5–WP8 runs there; `.venv/Scripts/python.exe -m pytest tests -q`).
* Ultralytics `int8=True` is for TFLite/TensorRT/OpenVINO, **not ONNX**. INT8 ONNX comes from
  `onnxruntime.quantization.quantize_dynamic` — see `scripts/wp1_export_int8.py`.
* GPU: RTX 5060 8 GB (batch 8 @768). Kaggle T4 works, notebook in `kaggle/` (preflight-tested).

## Files
* `code/` — package `sat7/` (rle, prefilter, orbit, scheduler, **real_workload**, dedup, synthetic)
  + `scripts/` + **86 tests** (optimum DP + whole-day bounds vs brute force, calibration maths,
  14 FMEA fault-injection tests)
* `code/data/yolo_ships/` — leakage-free dataset (split.csv, images, labels)
* `code/results/` — all measurements + charts; `code/runs/ships/weights/best.pt` — detector
* WP6 outputs: `wp6_report.md`, `wp6_tiles.csv`, `wp6_ships.csv`, `wp6_pred_flags.csv`,
  `wp6_catalogue.json`, `wp6_size_model.json/.png`, `wp6_real_table.csv`, `wp6_real_recall.png`,
  `wp6_real_sweep.png`, `wp6_thr_and_compare.png` (variants tagged `*_datasetmix`, `*_nosizemodel`)
* WP7 outputs: `wp7_report.md`, `wp7_cheap_products.csv|.json|.png`, `wp7_budget_sweep.csv|.png`
* WP8 outputs: `wp8_report.md`, `wp8_queue_aware.csv|.png`, `wp8_control_trace.png`
* WP5 gap: `wp5_optimality_gap_report.md`, `wp5_optimality_gap.csv|.png`, `wp5_gap_stdout.txt`
* WP5b joint bound: `wp5_joint_bound_report.md`, `wp5_joint_bound.csv|.png`
* WP9 failure modes: `wp9_failure_modes_report.md` (+ `tests/test_failure_modes.py`)
* WP10 sensitivity: `wp10_sensitivity_report.md`, `wp10_sensitivity.csv`, `wp10_tornado.png`
* WP1 exports (24 Sept): `wp1_int8_report.md`, `wp1_export.json`, `wp1_export_run.log`,
  `runs/ships/weights/best.onnx` + `best_int8.onnx`
* WP2 calibration (24 Sept): `wp2_calibration_report.md`, `wp2_calibration.json`,
  `wp2_thresholds.csv`, `wp2_reliability.png`, `wp2_val_predictions.csv`, `wp2_val_gt_matched.csv`
* WP3 gate (24 Sept): `wp3_gate_report.md`, `wp3_gate.json`, `wp3_gate_val_selected.json`,
  `wp3_gate_tradeoff.csv|.png`, weights `models/gate.pt`
  (⚠ `wp3_gate*_SMOKETEST_ONLY_1epoch.*` is a 1-epoch verification run — not a result)
* `report/` — 3 PDFs: Technical Report, Project Overview (from zero), Phase 2 & Poster Guide
* `GLOSSARY.md`, `architecture.png`, `simple_steps.png`
* Rebuild PDFs: `scripts/build_report.py`, `build_overview.py`, `build_phase2_guide.py`

## Honesty rules we follow (keep them)
Label every figure REAL / SIM / LIT / TARGET. We rejected 3 versions of the duplicate finder after
looking at evidence (hash-only, haze, no-data bars). Report negative results (classic pre-filter).
