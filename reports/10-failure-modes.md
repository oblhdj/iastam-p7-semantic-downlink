# WP9 — The FMEA table, made executable

The technical report carries a "Failure modes and mitigations" table with eight entries.
Every one of them was prose: of 72 tests, none injected a fault. These now exist as
`../code/tests/test_failure_modes.py` (14 tests), which inject each fault and check whether the
claimed mitigation actually holds.

Where a mitigation does not hold, the test pins the **real** behaviour and says so in its
name, rather than being weakened until it passes.

```
.venv/Scripts/python.exe -m pytest tests/test_failure_modes.py -q
```

## Scorecard

| # | Failure mode | Claimed mitigation | Verdict |
|---|---|---|---|
| 1 | Detector misses a ship | Thumbnail + raw ring buffer, ground re-request | **partly false** |
| 2 | Miscalibrated confidence | Calibration, recalibration from ground | **not implemented** (WP2 built, unrun) |
| 3 | Domain shift | Audit sampling: 1% random **raw tiles** | **partly false** |
| 4 | AIS gap or spoofing | Priority only; a human confirms | **FALSE for large vessels** |
| 5 | Pass capacity below prediction | Progressive items; re-plan every pass | holds |
| 6 | Storage full | Drop lowest value-per-byte first | holds |
| 7 | Software crash / bit flip | Watchdog + safe mode (thumbnails + FIFO) | holds (safe mode only; no watchdog) |
| 8 | Ground-station outage (a lost pass) | — | holds for buffering policies |

## The one that matters: "priority, not accusation" is false for big ships

The report states that a missing AIS match raises a vessel's **priority** and nothing else —
deliberately, so that a spoofed or absent AIS track cannot cause harm. The code does not do
that. Marking a vessel dark changes two things at once:

* its **value** goes up 5x, and
* its **payload** changes from L1 (a chip, bytes ~ px^0.57) to L2 (ROI + wake, ~ px^1.14).

The scheduler ranks by value per byte, and above a certain size the payload grows faster than
the weight compensates:

| ship length | L1 bytes | L2 bytes | rank if normal | rank if dark | |
|---|---|---|---|---|---|
| 12 px | 567 | 736 | 0.00176 | 0.00680 | dark wins |
| 50 px | 1,279 | 3,744 | 0.00078 | 0.00134 | dark wins |
| **128 px** | — | — | — | — | **crossover** |
| 200 px | 2,818 | 18,183 | 0.00036 | 0.00028 | **dark loses** |
| 392 px | 4,135 | 39,159 | 0.00024 | 0.00013 | **dark loses** |

**13.6% of real ships (18.7% of detected ones) sit above the crossover**, and they are the
largest vessels — precisely the ones an operator would least want deprioritised.

### Does it actually cost anything?

Only under heavy congestion. Re-running the real workload with `dark_weight` raised to 9.5
(the value that restores the claim at the largest real ship, 392 px):

| load | change to big-dark recall | all-dark recall | total ship recall |
|---|---|---|---|
| 80,000 tiles/day | none | none | none |
| 160,000 | none | none | none |
| 320,000 | **+14.1 points** (0.686 → 0.827) | +3.3 | **−2.9** |

Below saturation the ordering never binds, so the bug is invisible. At 320,000 tiles/day it
costs 14 points of recall on exactly the vessels the system exists to find.

### FIXED — by decoupling, not by tuning the weight

The obvious patch is a bigger `dark_weight`. Raising it to 9.5 does buy back 14 points of
large-dark recall, but it costs **2.9 points of overall ship recall**, because dark vessels
then crowd out ordinary traffic. It treats the symptom.

The actual defect is that **priority and payload were coupled**: one flag changed both what
an item is worth *and* what it costs. "Priority only" cannot be true while the flag also
selects a more expensive product.

`LoDConfig.dark_mode` now defaults to **`"decoupled"`**:

* a dark vessel gets **exactly the same base product** as any other ship of that size and
  confidence (L0 or L1), carrying the 5x weight;
* the ROI + wake becomes a **separate, lower-priority follow-up** item (value
  `dark_wake_value`, truncatable), sent when there is capacity for it.

The claim is now true *by construction* — the ranking ratio between a dark vessel and its
clean-track twin is exactly `dark_weight`, at every size, pinned by a test up to 1000 px.
The old behaviour is kept as `dark_mode="replace"` so the defect stays reproducible.

**And it is strictly better, not a trade.** Real workload, 320,000 tiles/day:

| metric | replace (old) | decoupled (new) | change |
|---|---|---|---|
| large-dark recall (>128 px) | 0.686 | **0.927** | **+24.1 points** |
| all-dark recall | 0.586 | 0.638 | +5.2 points |
| **overall ship recall** | 0.574 | **0.592** | **+1.8 points** |
| dark-vessel median latency | 7.41 h | **5.40 h** | **−2.0 hours** |
| bytes generated | — | — | +1.2% |

Everything improves at once, including total recall, which the weight-bump fix had damaged.
The reason is that the base chip is cheap (1–4 kB) and carries 5x value, so it has enormous
value-per-byte and goes out first; the expensive 18–39 kB ROI+wake is now optional instead of
being the dark vessel's *only* representation. Under congestion the old design let a big dark
vessel be crowded out entirely; now its report always gets through and only the wake imagery
is at risk.

At 40,000–160,000 tiles/day recall is unchanged (nothing binds) and the cost is +1.1% bytes;
dark-vessel latency still improves by 0.4 h at 160,000. The WP6 headline moves from 517x to
**511x** data reduction, with ship recall unchanged at 0.685.

## Other findings

**#1 — the thumbnail safety net is a hint, not a delivery path.** A tile whose ship the
detector missed does still get a thumbnail, so a human could spot it and re-request. But the
thumbnail `Item` carries no ship ids, so it contributes nothing to delivered recall, and
there is **no raw ring buffer anywhere in the code**. The report should say "a human may
notice and re-request", not imply the ship was reported.

**#3 — audit sampling exists, but not as described.** The report promises "1% random raw
tiles". What is implemented is a 2% sample of *thumbnails* on tiles the gate called empty.
It does fire, so silent gating is detectable, but thumbnails are not raw tiles and the
sample is not over all tiles.

**#7 — safe mode works, watchdog does not exist.** A degraded configuration (thumbnails
only, FIFO) still encodes and delivers without crashing. There is no watchdog.

**#5, #6, #8 hold as claimed.** A pass smaller than planned truncates progressive items
rather than dropping them; storage pressure evicts the lowest value-per-byte first and dark
vessels survive it; a fully lost pass is recovered by the next one for buffering policies
(and genuinely lost by the no-buffer variant, which is the documented behaviour).

## What to change in the report

1. Fix the AIS row, or fix the code — currently the table states something untrue.
2. Soften the "detector misses a ship" row: no raw ring buffer exists.
3. Correct "1% random raw tiles" to what is implemented.
4. Drop "watchdog" or build one.
