# Known limitations and future improvements

One page for everything this project does **not** show. Sections 1 and 2 restate limitations that
`README.md` and `paper/PHASE3_RESULTS.md` already declare, with the same numbers; section 3 covers
the demo; section 4 lists three points the audit raised that those documents do not yet state.
Every number is read from `code/results/` and a test compares this page with those files
(`code/tests/test_docs.py`). Labels are the repository's: REAL, SIM, SIM-over-REAL, ASSUMPTION.

---

## 1. The five to say out loud

### 1.1 One day fits; consecutive days do not

The headline day is simulated alone and given the 36 h of passes that follow it. It sends
**176 MB**. The 25 % link share sustains **134.5 MB per 24 h**, so a nominal day offers **130 %** of
what a day can carry [SIM-over-REAL, `wp29_validation.json` → `steady_state`].

Run for six consecutive days, the value-greedy scheduler recovers **0.677–0.688** of the ships on
every day, with a ship's first report in a median 4.7–5.6 h. It does so by shedding bytes: by day
six it delivers 82 % of the bytes offered. FIFO on the same days falls behind (median latency
**9.4 h → 42.9 h**) and delivers 0.424 of day six's ships.

Two conditions sit under that result. Recall counts a progressively truncated image as delivered
once it clears its minimum fraction (10 % of its bytes for a modeled item): an assumption, not a
measurement. And "100 % of what the satellite still knew" is a single-day statement. A 35 % share
carries the day outright.

### 1.2 24 % of the transmitted bytes are still model sizes

The data reduction is **341×** (60,124 MB → 176.3 MB a day, mean of three days). Of those bytes,
76 % are coastal context tiles charged at their own measured JPEG size; the other **24 %** —
safety thumbnails at a flat 1,000 B and ship crops from a fitted power law — are modeled
[`wp28_coast_tile_model.json`]. On real packets the crop model was within 0.94–1.26× of the
measured size. Measuring the coastal tiles alone moved the figure from the earlier all-modeled
557× to 341×; measuring the rest would move it again, by less.

### 1.3 No power was ever measured

Every energy figure is an **estimate**: an assumed power (CPU 28 W, GPU 60 W, transmitter 15 W,
inter-satellite link 12 W; all swept) multiplied by a stage time measured on a laptop or a
transmit time from the simulated link. Per day that gives 45.6 kJ of processing and 8.4 kJ of
transmission, 54.0 kJ in all [`wp17_energy_model.json`].

* ES_proc, 57.6 %, is a ratio of measured times, so the assumed powers cancel.
* ES_total, about 98 %, is against a bent pipe that sends every raw byte, which would need 92.6
  days of contact per day of imaging. A bent pipe limited to this link spends at most
  30.8 kJ/day, less than this pipeline, and delivers 0.2 % of the ships. Read ES_total as energy per
  unit of imagery accounted for, not as a smaller daily energy bill.

### 1.4 The lead over the fair baseline fails in six of 80 swept settings

Over the 80 settings of the invented constants, value-greedy ≥ FIFO holds in all 80, and "ours ≥
the fair Phi-sat-2-style baseline" in **74 of 80** [`wp10_sensitivity.csv`]. All six failures are
at 160,000 tiles/day, where the lead is 1.3 points before anything is changed (6.3 points at
40,000 tiles/day, where nothing fails): `thumb_value` 0.05 and 0.10, `conf_high` 0.90 and 0.99,
`conf_low` 0.05 and 0.10. Each makes the encoder spend more bytes on a saturated link. The
6.3-point lead is a nominal-load claim.

### 1.5 The energy model costs one detector call per tile; SAHI is a labelled variant

The default energy figures cost **one detector call per tile**, which is what the simulated day
runs. A pipeline that slices scenes with SAHI makes 1.5625 calls per tile. That case is recorded as
a labelled variant, not as the default [`wp17_energy_model.json` → `sahi_variant`]:

| | default | SAHI variant |
|---|---|---|
| processing per day | 45.6 kJ | 70.0 kJ |
| ES_proc | 57.6 % | 47.0 % |
| compute : radio | 5.5 : 1 | 8.4 : 1 |

The variant runs the gate once per tile, as it is trained and run, and assumes the fusion step
costs 0 ms, because it has never been timed. **Status:** of the judge-facing documents only
`PHASE3_RESULTS.md` §6 quotes the variant; the README, the QA card and the slides quote the
default alone.

## 2. Declared in the README and PHASE3_RESULTS as well

| Limitation | What it means for a claim |
|---|---|
| **Onboard execution is a target, not a prototype.** Every timing is a laptop | "runs onboard" is argued against flown hardware of the same class, not demonstrated |
| **Cloud.** The Airbus set is 0.4 % cloud; the assumed 15 % share is resampled from 21 real cloud tiles | recall 0.685 is quoted with its cloud fraction, as a band of 0.40–0.82 over 0–50 % cloud |
| **Small ships.** At the 0.25 cut recall on small ships is 0.50–0.61 | the detector, not the downlink, is the residual loss |
| **Scenes are stitched from independent tiles**, because an Airbus tile is already 768 px | B1 (0.339) and B2 (0.717) are measured on synthetic scenes; no ship crosses a seam |
| **SAHI does not beat plain tiling on those scenes**: 0.717 against 0.730, at 1.56× the calls | B2's gain is over one call on the whole scene, not over a tile-by-tile pass |
| **Window 768, not the paper's 512** | the ranking reverses at 512 |
| **Two inputs.** B1/B2 use 150 scenes; B3/B4 a simulated 40,000-tile day | B2's detection recall and B3's delivered recall are not steps of one curve |
| **The relay buys latency, not energy**: about 1.8× the communication energy per relayed item | the canonical B4 row's relay latency is a window estimate; the capacity-aware simulation is `wp27` |
| **One ground station; tiles resampled independently; capture times uniform over the day** | no orbital strips, no station network in the headline |
| **Coastal recompression is not free**: quality 40 costs 4.0 points of recall, on small ships | byte savings on coastal tiles are a recall trade |
| **No AIS data**: the dark-vessel flag is random | the prioritisation is implemented and fault-tested, not demonstrated |

## 3. Limits of the demo

* The bundled tiles are synthetic stand-ins (the Airbus rules forbid redistributing real ones).
  Everything computed on them is labelled SYNTH: it shows the chain runs, it is not a measurement.
* Accuracy appears only when a label file is uploaded with an image, and one image is a spot check.
* Per-image energy is an assumed 28 W multiplied by a time measured on the presenting laptop.
* The link panel sends one image down an otherwise empty link: no queue, no contention. The
  multi-day result beside it is the committed one, not a live run.
* Each tile's context comes from the classic pre-filter, which was tuned on Airbus tiles and can
  call open sea "cloud" on an unfamiliar image. A manual override exists and is shown as manual.
* The learned gate is a torch model and is not in the CPU demos.
* The static fallback covers four samples at default settings.
* On Windows a background process is throttled; the demo opts out. That opt-out was verified
  against a forced throttle, not against Windows applying it by itself.

## 4. Raised by the audit, not yet stated in the judge-facing documents

Reported here so that they are in one place. No number was changed.

1. **Coastal ships the detector missed are counted as found on the ground.** In the level-of-detail
   scheme a delivered coastal tile credits every ship on it, including those the onboard detector
   did not report (the audit put them at 6.3 % of the ships in the simulated day). No ground-side
   detector is run, and no lower bound without that credit is published beside 0.685.
2. **ES_proc compares against a pipeline stage that the proposed pipeline still needs.** The 57.6 %
   removes the classic pre-filter from the proposed arm, while the encoder still takes each tile's
   context from it.
3. **"0 decisions changed."** `wp11_integration.json` records one ship in 8,173 changing side at
   the 0.25 cut (counted under `det_thr` and `conf_low`, the same threshold) and none at
   `conf_high`, which the script checks at 0.9 rather than the 0.670 in use. `PHASE3_RESULTS.md` §5
   was corrected on 10 Oct; `demo/PAPER_COVERAGE.md` §2 still carries the old wording.

## 5. Future improvements, in the order they would change a claim

1. **Publish the lower bound** on recall without ground-side recovery of coastal misses (the
   scheduler already has the mode), and state the assumption beside the headline.
2. **Measure power** on a development board of the target class, with wall-clock time and quantised
   accuracy. This turns every estimate in §1.3 into a measurement.
3. **Measure the remaining 24 % of the bytes** as real serializations, as was done for coastal tiles.
4. **Make the multi-day run the canonical one**, and replace the truncation assumption with a
   measured curve of detectability against delivered fraction.
5. **Evaluate SAHI on real large scenes.** Loaders for DOTA and HRSID exist; the detector needs
   fine-tuning on the new domain first.
6. **Restate ES_proc** with the pre-filter in both arms, or take the context from the learned gate;
   time the fusion step so the SAHI variant no longer assumes it.
7. **Put the learned gate into the canonical pipeline**, and export it to ONNX so the CPU demo runs it.
8. **Use the capacity-aware relay simulation for the canonical B4 row.**
9. **A cloud climatology** in place of 21 tiles.
10. **Give the joint program a value term**, so that different weightings choose different levels.
11. **Housekeeping:** continuous integration; the decision on the Airbus galleries in git history;
    remove the stale `sim_*` artefacts; rewrite `code/README.md`.
