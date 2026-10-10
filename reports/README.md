# Reports

Twenty work packages — each a short write-up of what was measured, what it changed, and what it
cost us to believe — plus one planning spec (18) and an onboard-feasibility reframe (22). They are
meant to be read in order, but each stands alone.

Numbers live in `../code/results/` as `.csv` / `.json`; these documents narrate them. Where a
report carries a **REFRESHED** banner, the banner is current and the tables below it are not —
the `.csv` is always authoritative.

> ### ⚠ REFRESHED 10 Oct 2026 — two canonical changes; reports 05–22 narrate the earlier estimate
>
> 1. **The coastal context tile is now measured.** Every result up to report 22 charged a coastal
>    tile a flat 32.4 kB, taken from a size ladder measured on open-sea tiles. Each tile is now
>    charged its own JPEG q40 size (median 66.5 kB, `wp28_coast_tile_model.json`), and every result
>    that depends on it was regenerated. Thumbnails and ship crops are still modeled sizes.
> 2. **B1 / B2 are now the same-input run** (`wp26_b0_b4.py`, 150 scenes, 1,164 ships): B1 is one
>    detector call on the whole scene, B2 is SAHI on those scenes.
>
> The one-sentence summaries below, and the bodies of reports 05–22, still quote the **earlier
> all-modeled estimate**. It is kept, not deleted: `wp6_real_table_modeledcoast.csv` reproduces it.
> Current values, from the regenerated files:
>
> | quantity | reports 05–22 say (earlier estimate) | current | file |
> |---|---|---|---|
> | data reduction vs a bent pipe | 557× (108.0 MB/day) | **341×** (176.3 MB/day) — coastal tiles measured, thumbnails and crops modeled | `wp6_real_table.csv` |
> | B1 / B2 recall | 0.782 / 0.745 (262- and 47-ship samples, different inputs) | **0.339 / 0.717** (same scenes) | `wp26_b0_b4.json` |
> | detector on native tiles (not the paper's B1) | 0.782 (262 ships) | 0.766 (8,173 ships) | `wp24_detection_eval.json` |
> | vs FIFO at 160k tiles/day | 2.26× | 3.59× (1.88× at 80k) | `wp6_real_sweep.csv` |
> | greedy vs exact optimum, operational scale | 0.251% | 0.884% | `wp5_optimality_gap.csv` |
> | value of whole-day foresight | ≤0.2% | ≤0.3% | `wp5_joint_bound.csv` |
> | queue-aware worst-case regret | 1.1 points | 0.2 points | `wp8_queue_aware.csv` |
> | thumbnail gating re-key | 3.8% of the downlink | 2.3% | `wp14_gate_signal.json` |
> | "ours ≥ fair baseline" under the sweep | 69 of 70 settings | 74 of 80 (all 6 failures at 160k) | `wp10_sensitivity.csv` |
> | compute : radio energy | 8.9 : 1 | 5.5 : 1 | `wp17_energy_model.json` |
> | relay (B4): worst-case latency, share, energy | 11.6→6.2 h, 44%, +34% | 35.2→10.9 h, 66%, +52% | `wp18_campaign.json` |
> | P0–P3 vs LoD, same day | 49.9 vs 108.5 MB | 90.5 vs 178.5 MB | `wp23_semantic_compare.json` |
>
> Unchanged: ship recall 0.685 at 15% cloud, the +6.3 points over the fair baseline at nominal
> load, detector mAP50 0.804, ES_proc 57.6%. Not regenerated: `wp7_budget_sweep.csv` (it sweeps
> coastal sizes from the open-sea ladder itself and needs a new coastal measurement, not a rerun).

## The story in order

| # | report | the one sentence |
|---|---|---|
| 01 | [Data integrity](01-data-integrity.md) | 19.1% of Airbus tiles have a near-duplicate twin; a naive split would have leaked 1,617 groups into the test set, ours leaks 0 |
| 02 | [Detector and quantisation](02-detector-and-quantisation.md) | mAP50 0.804. ONNX FP32 is lossless and 1.21× faster on CPU; **INT8 is 8.5× slower and costs 2.2 points on small ships** — rejected |
| 03 | [Confidence calibration](03-confidence-calibration.md) | the detector is *underconfident* (says 0.352, right 0.408 of the time); isotonic cuts ECE 8.9×, and the cheap rung was cut at the wrong place |
| 04 | [The onboard gate](04-onboard-gate.md) | the classic CV pre-filter fails as a gate (0.645 recall); a 47k-parameter CNN keeps **98.8% of ships while skipping 40% of empty tiles** |
| 05 | [Scheduler on real detections](05-scheduler-on-real-detections.md) | replacing the synthetic workload with real detector output moved recall 97.1% → **0.685**, and showed the downlink is no longer the bottleneck |
| 06 | [The byte budget](06-byte-budget.md) | **86% of our "semantic downlink" was not semantic** — coastal tiles and blanket thumbnails. Also: our baseline had been unfair, so the margin is +6.3 points, not +16.7 |
| 07 | [Queue-aware encoding](07-queue-aware-encoding.md) | no fixed coastal policy is best in more than one regime; letting the buffer set the detail level is within 1.1 points of the best at every load |
| 08 | [Optimality gap](08-optimality-gap.md) | an exact DP, verified against brute force, says greedy is **0.251% from optimal** at operational scale — the plan had ticked this as done with no solver in the repo |
| 09 | [Whole-day bound](09-whole-day-bound.md) | perfect foresight over an entire day is worth **≤0.2%**, so *do not build arrival prediction* |
| 10 | [Failure modes](10-failure-modes.md) | the FMEA table became 14 fault-injection tests, and **it failed in places** — a dark-vessel rule did the opposite of what we claimed on 13.6% of ships |
| 11 | [Sensitivity](11-sensitivity.md) | every invented constant swept; conclusions hold in 69 of 70 settings, and **cloud fraction dominates everything** |
| 12 | [End-to-end integration](12-end-to-end-integration.md) | real JPEGs through the real chain reproduce the simulation to four decimals — and 52% of reported ship loss turns out to rest on 21 real tiles |
| 13 | [Tile seams and SAHI](13-tile-seams-and-sahi.md) | on a real 10k swath a 380 px ship is cut 74.5% of the time, but **a cut ship is usually still detected** — so blanket SAHI is a poor trade ⚠ *§5 corrected by [report 16](16-swath-policies.md)* |
| 14 | [Coastal recompression](14-coastal-recompression.md) | the q40 coastal setting we adopted as "no recall cost" **actually costs 4.0 points**, all on small ships — and q30 is rejected on measurement |
| 15 | [Gate signal re-key](15-gate-signal-rekey.md) | moving thumbnail gating onto the learned gate returns **3.8% of the whole downlink at zero measured recall cost** |
| 16 | [Swath tiling policies](16-swath-policies.md) | on real multi-tile swaths with global fusion, **SAHI costs 1.56× the regular tiling (down from 2.25×) and beats edge-triggered** on both seam recall and compute — reversing report 13 §5 |
| 17 | [Energy model](17-energy-model.md) | the per-stage `E_proc`/`E_comm` model: the learned gate saves **≥57.6% of processing energy** (power-free for the CPU onboard build), and **once you stop sending pixels, compute outweighs the radio 8.9 : 1** — vs a bent pipe, ES ≈ 98% |
| 18 | [Campaign-runner spec](18-campaign-runner-spec.md) | *planning doc, not code:* how B0–B4 compose from today's modules (B2 done, B1/B3 = the wp11 chain, B4 absent) — **the "end-to-end B0–B4 run" is blocked on the relay; B0–B3 is the executable scope today** |
| 19 | [Relay energy](19-relay-energy.md) | `E_relay = E_ISL + E_GS` for B4: **relay always costs more energy than direct (1.8×), so it's a latency buy, not a power saving** — our energy injected into report 20's `route()` gives weighted ES **0.9816** (f=0.44), but ES is ~flat in the relay fraction (0.14-pt spread) |
| 20 | [Relay path choice](20-relay-path-choice.md) | B4 windows + `min(J_direct, J_relay)`: a complementary-coverage relay **halves worst-case latency (11.5 h → 5.8 h)** — SIM from real orbit propagation; the energy half is TARGET (report 19), and direct-only reproduces the ground-only baseline exactly |
| 21 | [B0–B4 campaign](21-b0-b4-campaign.md) | the full campaign end to end (closes START_HERE §5.4), B3 pinned to the 108 MB/557× headline row: **B4 is a reroute, not extra capacity** — recall/MB = B3 verbatim (0.680 at 15% cloud / 108.5 MB), relay cuts worst-case latency **11.6 h → 6.2 h** for +34% comm energy (TARGET); all sanity gates pass |
| 23 | [Paper-faithful modules](23-paper-faithful-modules.md) | first-class SAHI mode + isolated overlap/fusion ablations, the **P0–P3 scheme** (Table I) as a drop-in for the LoD ladder, the joint program **`min αE+βD+γT`** *solved as one program* with a validity bracket, a calibratable energy model, and large-scene (DOTA/HRSID) loaders — infra, no new headline; LoD still reproduces the headline row, P0–P3 is leaner (90.5 vs 178.5 MB with measured coastal tiles; 49.1 vs 108.5 MB on the earlier flat size) at −6.3 pts recall |
| 22 | [Onboard feasibility](22-onboard-feasibility.md) | **demo, not prototype:** every timing is a laptop, so "runs onboard" is a TARGET — but a 3.15M-param detector + 47k-param gate is the class our baseline **Φ-sat-2 already runs onboard** (Myriad 2, ~1 W), it's a **few-% duty cycle** at 40k tiles/day even on a VPU 6× slower, and the ES_proc energy saving is a hardware-independent time ratio; states what a prototype still owes |

## If you only read three

**[05](05-scheduler-on-real-detections.md)** for what the system does, **[12](12-end-to-end-integration.md)** for
proof that it does it on real data, and **[10](10-failure-modes.md)** for how we found out where it
breaks.

## Reading the labels

Every figure in every report is marked:

- **REAL** — measured on Airbus imagery with our trained models
- **SIM** — produced by the orbit/link simulator over real detections
- **LIT** — taken from published work, cited
- **TARGET** — a design goal, not an achievement
- **ASSUMPTION** — a number we invented; all of these are swept in [report 11](11-sensitivity.md)
