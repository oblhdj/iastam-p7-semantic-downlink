# Reports

Twelve work packages, each a short write-up of what was measured, what it changed, and what it
cost us to believe. They are meant to be read in order, but each stands alone.

Numbers live in `../code/results/` as `.csv` / `.json`; these documents narrate them. Where a
report carries a **REFRESHED** banner, the banner is current and the tables below it are not —
the `.csv` is always authoritative.

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
| 13 | [Tile seams and SAHI](13-tile-seams-and-sahi.md) | on a real 10k swath a 380 px ship is cut 74.5% of the time, but **a cut ship is usually still detected** — so blanket SAHI is a poor trade |
| 14 | [Coastal recompression](14-coastal-recompression.md) | the q40 coastal setting we adopted as "no recall cost" **actually costs 4.0 points**, all on small ships — and q30 is rejected on measurement |
| 15 | [Gate signal re-key](15-gate-signal-rekey.md) | moving thumbnail gating onto the learned gate returns **3.8% of the whole downlink at zero measured recall cost** |

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
