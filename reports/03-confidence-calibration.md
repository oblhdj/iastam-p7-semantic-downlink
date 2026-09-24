# WP2 — is the detector's confidence a probability? (24 Sept 2026)

Closes the last unstarted must-have and NEXT-STEP #1. Scripts: `wp2_predict_split.py` (val
inference, 3.12 torch env) then `wp2_calibrate.py` (pure numpy, `.venv`). Data:
`wp2_calibration.json`, `wp2_thresholds.csv`, `wp2_reliability.png`. All figures REAL.

**Fitted on val, evaluated on test, never the reverse.** The val split had never been used for
anything; it is now dumped (`wp2_val_predictions.csv`, 17,201 boxes, 8,173 ships, recall 0.871
at conf 0.05 vs 0.861 on test). Verified disjoint from test: **0 overlapping images.**

## 1. The raw confidence is not a probability — it is systematically underconfident
| mapping | ECE | Brier | mean confidence | actual accuracy |
|---|---|---|---|---|
| raw | **0.0878** | 0.1132 | 0.352 | 0.408 |
| Platt (a=1.663, b=0.889) | 0.0143 | 0.1000 | 0.406 | 0.408 |
| **isotonic** (recommended) | **0.0099** | 0.1000 | 0.406 | 0.408 |

The detector says 0.352 on average and is right 0.408 of the time: it **understates** its own
reliability by 5.6 points. Isotonic regression cuts calibration error **8.9x** (0.0878 -> 0.0099)
and Platt 6.1x; both reach the same Brier, so the gain is pure calibration, not discrimination
(a monotone map cannot change ranking, hence cannot change mAP or recall-at-rank).

## 2. Consequence: the cheap rung was cut at the wrong place, and it is 7x more usable than we thought
`LoDConfig.conf_high` (0.9 at the time of writing, **0.670 since**) gates the 40-byte L0 report (metadata only, no chip); everything
below gets a ~900 B L1 chip. What the rungs should compare against, after calibration:

| intended probability | raw score that means it | share of predictions (calibrated) | share (raw cut) |
|---|---|---|---|
| 0.25 | 0.278 | 45.2% | 48.1% |
| 0.40 | **0.329** | 40.1% | 34.8% |
| 0.75 | 0.513 | 28.6% | 18.0% |
| **0.90** | **0.670** | **21.7%** | **3.1%** |
| 0.95 | 0.725 | 19.2% | 0.2% |

**"0.9" should have been 0.670.** Because the network is underconfident, raw 0.9 is not a
90%-certain ship — it is far more certain than that, which is why the rung fired on only **3.1%**
of predictions (WP6 measured the same thing per-ship: 7.5%). Cut where 0.9 actually means 0.9 and
the cheap rung serves **21.7%** of predictions — **7x more**. Be honest about the direction: this
*accepts more risk than today*, because today's rung is stricter than advertised. The point is
that the threshold now means what it says.

## 3. The re-cut is already measured, and it is strictly better
No new simulation was needed: WP10 swept `conf_high`, and the calibrated 0.670 sits just below
its 0.70 sample (`wp10_sensitivity.csv`; the trend is monotone from 0.99 down to 0.70, so 0.670
is marginally better again).

| load | conf_high | ship recall | MB offered | median latency h | margin over fair baseline |
|---|---|---|---|---|---|
| 40k (link idle) | 0.90 now | 0.6799 | 117.9 | 4.498 | +0.0633 |
| 40k | **0.70** | 0.6799 | **108.9** | 4.490 | +0.0633 |
| 160k (link binds) | 0.90 now | 0.6476 | 467.1 | 4.996 | +0.0265 |
| 160k | **0.70** | **0.6618** | **429.4** | **4.871** | **+0.0406** |

* At 40k: **-7.6% bytes for zero recall cost.**
* At 160k: **+1.4 points of ship recall AND -8.1% bytes AND -0.13 h latency** — strictly better
  on every axis, and the margin over the fair Phi-sat-2 baseline grows from +2.6 to +4.1 points.
* Mechanism: every ship moved from a 900 B chip to a 40 B report frees ~860 B, and under
  congestion those bytes become other ships. Nothing is given up because the ships being
  down-rung are the ones we are most certain about.
* This closes the gap WP10 left open: it had shown 0.70 dominates 0.90 but had **no principled
  reason** to prefer any value. Calibration supplies it — 0.670 is the score that means 0.9.

## 4. ⚠ CORRECTION — the "conf_low 0.4 → 0.329" recommendation was wrong

The first version of this report said `conf_low = 0.4` discards detections whose true
probability is still ≥ 0.4, and recommended lowering it to the calibrated equivalent **0.329**
for free recall. **That was wrong, and in the wrong direction.**

`0.4` is only the *dataclass default*. **Every script overrides it** with the onboard detection
threshold: `LoDConfig(conf_low=args.det_thr)` with `det_thr = 0.25`, in `wp6_simulate_real.py`,
`wp7_budget_sweep.py`, `wp8_queue_aware.py`, `wp10_sensitivity.py` and `wp11_integration_demo.py`
alike. So production already runs at **0.25**, which is *more* permissive than 0.329 — moving to
0.329 would **tighten** the bottom rung and lose ships, not gain them. The 0.4 default is dead
code; it is now commented as such in `../code/sat7/scheduler.py`.

The real structure is that **`conf_low` and `det_thr` are the same knob doing two jobs**: what
the detector is willing to report, and what the encoder bothers to send. `wp6_simulate_real.py`
now takes `--conf-low` / `--conf-high` so the two can be separated, and `conf_low` is swept
independently in WP10 over {0.05, 0.10, 0.25, 0.329, 0.40} at fixed `det_thr = 0.25`.

The genuinely useful version of this question was already answered by WP6 from the other side:
lowering the **detection** threshold 0.25 → 0.05 buys **+8.8 points of ships for +8% bytes**, at
5.7× more false alarms — a ground-segment decision, not a calibration one.

**Lesson worth keeping:** a calibrated threshold is only meaningful next to the value actually
in use. Read the call site, not the dataclass.

## 5. Recommendation
1. Adopt **isotonic** and report calibrated probabilities to the ground segment, not raw scores.
   Platt is the fallback if two parameters are easier to fly than a step function; it costs
   0.004 ECE.
2. ~~Re-cut `conf_high` **0.9 -> 0.670**~~ — **ADOPTED 24 Sept** as the `LoDConfig` default.
   Measured effect at 40k after the full re-run: recall unchanged at **0.685**, bytes
   **117.9 -> 108.0 MB (-8.4%)**, latency 4.50 -> 4.38 h, data reduction **511x -> 557x**.
3. ~~Sweep `conf_low` 0.4 -> 0.329~~ — withdrawn, see §4: production is already at 0.25.
4. A monotone map changes no ranking, so **no WP5-WP10 conclusion is at risk** from items 1-2
   beyond the byte/recall shifts tabulated above.
