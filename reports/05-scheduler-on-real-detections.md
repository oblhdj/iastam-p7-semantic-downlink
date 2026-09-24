# WP6 -- The scheduler on REAL detections

> ### ⚠ REFRESHED 24 Sept 16:30 — THIS DOCUMENT'S INNER NUMBERS ARE SUPERSEDED
> **Latest re-run adopts `conf_high` 0.670 (WP2):** recall unchanged at **0.685**, bytes
> **117.9 → 108.0 MB (-8.4%)**, latency 4.50 → 4.38 h, data reduction **511x → 557x**.
> The prose and tables below predate the WP7 byte-budget defaults (q40 coastal + gated
> thumbnails) and the WP9 decoupled dark-vessel encoder. **`wp6_real_table.csv` and
> `wp6_real_sweep.csv` (both re-run 14:03, post-fix) are authoritative.** What changed:
>
> | claim in this document | superseded by |
> |---|---|
> | "Data reduction: **421x**" (§ below) | **557x** (60,124 MB offered → 108.0 MB sent) |
> | "we send **142.6 MB**" | **108.0 MB** |
> | "**2.7x** more ships than FIFO" | **2.4x** (orbit mix, 160k tiles/day: 0.648 vs 0.269) |
> | — same, dataset mix | **2.2x** (40k tiles/day: 0.769 vs 0.349) |
> | baseline table shows only "Phi-sat-2 style" **0.518** | that baseline was **unfair** (ocean-only by construction). The fair one scores **0.622**, so our margin is **+6.3 points**, not +16.7 — see WP7 §4 |
>
> Ship recall **0.685** and "100% of what the satellite still knows" are unchanged.


The Contribution-A experiment used to run on a synthetic day: contexts drawn from assumed
probabilities, detector confidence drawn from a Beta(5,2). WP6 replaces those draws with the
trained detector's real output on the 5,320 held-out test tiles, and reports what changes.

Reproduce:

```
python scripts/wp6_build_catalogue.py          # ~2 min, 8 workers  -> wp6_tiles/ships/pred_flags.csv
python scripts/wp6_fit_size_model.py           # -> wp6_size_model.json
python scripts/wp6_simulate_real.py            # -> wp6_real_table.csv + 3 charts
```

## 1. What is now measured, and what is still assumed

| REAL (measured on the held-out split) | ASSUMPTION (stated on every chart) |
|---|---|
| ships per tile, their lengths | when each tile is captured (no timestamps in the data) |
| detector confidence on every ship, and every miss | which ships are dark (no AIS in the data) -- 10% |
| false alarms, per tile, at any threshold | tiles imaged per day -- 40,000 |
| tile context (classic pre-filter, same code as onboard) | the context mix, unless `--mix dataset` |
| payload bytes per ship (WP4 power law) | link budget and pass geometry (WP: orbit) |

The Airbus set is a ship-*finding* benchmark: 59% of its tiles contain a ship and 27% touch a
coast, which no real orbit sees. So the default run keeps the operational context mix as a stated
assumption and draws a real tile from inside each context; `--mix dataset` uses the file's own mix.
Either way everything *inside* a tile is measured.

## 2. The catalogue (`wp6_catalogue.json`)

5,320 tiles, 8,173 ships. Pre-filter contexts: 3,707 candidates, 1,415 land/coast, 177 empty sea,
21 cloud (82 ms/tile median).

| assumed in WP5 | measured in WP6 |
|---|---|
| ships per ship-tile 1.8 | **1.83** (the one assumption that held) |
| ships per coast-tile 3.0 | **1.73** |
| false positives 2% of empty tiles | **0.6%** at conf>=0.4, **1.6%** at 0.25, **7.7%** at 0.05 |
| confidence ~ Beta(5,2), ~28% above 0.9 | median **0.68**, p90 **0.89**, only **7.5%** above 0.9 |

Consequences of the last row: the cheap 40-byte L0 rung of the level-of-detail ladder almost never
fires on real data, so the ladder collapses to mostly L1 chips. The thresholds were calibrated for a
confidence distribution the detector does not produce -- this is WP2's job.

Two more numbers that decided design questions:

* **Coastal tiles hold 30% of all ships** (2,454/8,173) and detector recall there is **0.784** vs
  **0.895** on open sea. Sending the whole coastal tile is not a hedge, it is where a third of the
  ships live.
* **12 ships sit on the 21 tiles the cloud gate discards.** They are kept in the recall denominator
  on purpose, so that loss is visible instead of free.

## 3. Payload cost per ship (`wp6_size_model.json`)

WP4 reported medians, but its own p90 is 3-6x the median. Fitting the 551 real crops:

| level | product | fit | R^2 |
|---|---|---|---|
| L1 chip | crop tight q40 | bytes = 137.5 * px^0.57 | 0.76 |
| L2 ROI + wake | crop wake q80 | bytes = 43.3 * px^1.14 | 0.89 |

A 300 px cargo ship costs **6.3x** a 12 px fishing boat as an L1 chip (3,550 B vs 567 B) and **39x**
as an L2 ROI with wake (28.9 kB vs 736 B). The scheduler ranks by value per byte, so each real ship
is now charged for its own real length instead of one median for all.

## 4. Result: 40,000 tiles/day, orbit mix (`wp6_real_table.csv`)

21,137 real ships (14,785 above conf 0.25), 2,908 real false alarms, 6 passes, 210 MB of capacity.

| strategy | ships delivered | onboard ceiling | of the ceiling | dark | median latency | MB sent |
|---|---|---|---|---|---|---|
| Bent pipe (raw, FIFO) | 0.002 | 0.841 | 0.3% | 0.002 | 19.5 h | 201.7 (capped) |
| Phi-sat-2 style (patch, FIFO) | 0.518 | 0.518 | 100% | 0.522 | 4.4 h | 13.5 |
| Ours: LoD + no buffer | 0.533 | 0.685 | 78% | 0.529 | 5.8 h | 110.6 |
| Ours: LoD + FIFO | 0.685 | 0.685 | 100% | 0.679 | 6.0 h | 142.7 |
| **Ours: LoD + value-greedy** | **0.685** | 0.685 | **100%** | 0.679 | **4.4 h** | 142.6 |

*Onboard ceiling* = ships the onboard software still knows about after the cloud gate and the
detector, i.e. the best an infinite downlink could deliver for that encoder.

**Data reduction: 421x** -- a bent pipe would have to offer 60,124 MB for the same day; we send 142.6 MB.
(The old synthetic figure was 442x, so that headline survives contact with real data.)

The honest reading of this table: at this load **the downlink is no longer the bottleneck**. Our
scheduler delivers 100% of what the satellite knows; the gap from 0.685 to 1.0 is the detector and
the cloud gate, not the link. Scheduling and FIFO tie here, and value-greedy's only win is latency
(4.4 h vs 6.0 h). Phi-sat-2-style patches are 10x cheaper in bytes; our extra 129 MB buys +0.17
recall, and it buys it through *coverage* (coastal tiles, graded detail), not through scheduling.

Compared with the synthetic workload, same orbit and link: 0.685 vs **0.998**. The interim report's
97-99% was largely a property of the invented confidence distribution.

## 5. Where the scheduler actually earns its place

Scheduling only matters once the offered load exceeds the link. Two real ways to get there:

**(a) Higher duty cycle** (`wp6_real_sweep.csv`, orbit mix):

| tiles/day | value-greedy | FIFO | no buffer |
|---|---|---|---|
| 40,000 | 0.680 | 0.680 | 0.528 |
| 80,000 | **0.687** | 0.443 | 0.294 |
| 160,000 | **0.637** | 0.221 | 0.173 |

**(b) Dense scenes** -- the dataset's own ship-heavy mix (`--mix dataset`, 61,938 ships/day):

| strategy | ships | dark vessels | of the ceiling | median latency |
|---|---|---|---|---|
| Ours: LoD + value-greedy | **0.743** | **0.840** | 88% | 5.5 h |
| Ours: LoD + FIFO | 0.280 | 0.280 | 33% | 18.4 h |
| Ours: LoD + no buffer | 0.214 | 0.221 | 25% | 8.8 h |

**2.7x more ships and 3.0x more dark vessels than FIFO, with latency cut from 18.4 h to 5.5 h.**
This is the Contribution-A claim, now on real detections.

## 6. Where to put the onboard threshold (`wp6_thr_sweep.csv`)

Only possible with real confidences:

| threshold | ships delivered | false alarms/day | MB sent |
|---|---|---|---|
| 0.05 | **0.768** | 16,576 | 155.0 |
| 0.25 (current) | 0.680 | 2,908 | 143.2 |
| 0.40 | 0.611 | 1,190 | 140.8 |
| 0.70 | 0.464 | 200 | 136.8 |

Dropping the cut-off from 0.25 to 0.05 delivers **+8.8 points of ships for +8% of bytes**, because the
byte budget is dominated by coastal tiles and thumbnails, not by ship chips. The cost is not
bandwidth, it is **5.7x more false alarms** for the ground-side analyst. That is a ground-segment
decision, and it should be made with this table rather than by keeping the detector's default.

## 7. Robustness

* **3 resampled days** (different tiles, same settings): ship recall spread [0.680, 0.690].
* **Payload model ablation** (`--no-size-model`, every ship charged the WP4 median): 0.680 vs 0.685
  at 40k, 0.648 vs 0.637 at 160k. The per-ship power law is the more faithful model but the
  conclusions do not depend on it -- worth saying out loud.
* Wilson 95% CI on 21,137 ships: +/-0.006, so the differences above are far outside the noise.

## 8. What this changes for the rest of the project

1. The remaining loss is the **detector**, not the downlink. Small-ship recall (0.739 for <32 px) and
   coastal recall (0.784) are now the highest-value work, not more scheduler tuning.
2. **WP2 calibration is now clearly scoped**: the LoD ladder's thresholds (0.4 / 0.9) were set for a
   confidence distribution the detector does not produce. Calibrate, then re-cut the rungs.
3. Report both operating points. "421x less data, 100% of what the satellite knows" at nominal load;
   "2.7x more ships than FIFO" under saturation. Neither alone is the whole story.
