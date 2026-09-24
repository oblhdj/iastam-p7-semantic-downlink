# WP10 — Which results depend on numbers we invented?

> ### ⚠ REFRESHED 24 Sept 16:39 — re-run after adopting `conf_high` 0.670, and `conf_low` added
> The tables below were computed with `conf_high = 0.9` and **did not sweep `conf_low` at all**.
> `wp10_sensitivity.csv` is authoritative. Three changes:
>
> **1. `conf_low` is the second most influential constant in the project** — and it had never
> been swept. Recall range **0.157** at 40k, **0.146** at 160k, behind only cloud:
>
> | `conf_low` | recall @40k | recall @160k | |
> |---|---|---|---|
> | 0.05 | 0.768 | 0.743 | |
> | 0.10 | 0.743 | 0.719 | |
> | **0.25** | **0.680** | **0.662** | ← production (tied to the detection threshold) |
> | 0.329 | 0.648 | 0.629 | the "calibrated P=0.4" value WP2 first recommended |
> | 0.40 | 0.611 | 0.597 | the dead dataclass default |
>
> **WP2's original "lower `conf_low` to 0.329" advice would have cost 3.2 points of recall.**
> It is withdrawn — see WP2 report §4. Production is already at 0.25, *below* 0.329.
>
> **2. `conf_high` 0.670 adopted, and the sweep confirms it.** At 40k recall is flat at 0.680 for
> every value (only bytes move: 108.5 MB at 0.670 vs 120.6 at 0.99). At 160k: 0.6622 at 0.670
> vs **0.6476 at the old 0.9** — **+1.5 points**, and the margin over the fair baseline grows
> from +2.6 to **+4.1** points. `conf_high` drops to 5th place in the ranking.
>
> **3. The cloud band is unchanged and still dominates** — see § below, now stated as a band.

## ⚠ The cloud fraction is an assumption, and it sets the headline

Every recall figure in this project is quoted at an **assumed 15% cloud fraction**. It is the
single most influential number we invented, and WP11 showed it rests on **21 real tiles**
(52% of all reported ship loss comes from resampling them). The honest statement is a band:

| cloud fraction | ship recall @40k | @160k |
|---|---|---|
| 0% | **0.819** | 0.775 |
| 5% | 0.771 | 0.736 |
| **15% (what we quote)** | **0.680** | **0.662** |
| 30% | 0.562 | 0.552 |
| 50% | **0.396** | 0.392 |

**Say "0.68 at an assumed 15% cloud fraction; 0.40–0.82 across 0–50%", never "0.68" alone.**
It is a *denominator* effect — our margin over the fair baseline stays positive at every level
and greedy still beats FIFO everywhere — but the absolute number is not ours to claim without
the assumption attached. The Airbus set is 0.4% cloud and cannot settle it; a climatology is
needed [LIT].


Seven constants underpin every headline in this project and none of them were measured:

| constant | baseline | where it came from |
|---|---|---|
| `dark_weight` | 5.0 | invented |
| `dark_wake_value` | 1.0 | invented (added in WP9) |
| `thumb_value` | 0.01 | invented |
| `p_dark` | 0.10 | invented — there is no AIS in the Airbus data |
| cloud fraction | 0.15 | invented — the dataset itself is 0.4% cloud |
| `conf_high` | 0.9 | inherited from the original ladder |
| `aging_per_hour` | 0.5 | invented |

Each is swept alone around the baseline, at a load where nothing binds (40,000 tiles/day) and
one where the link does (160,000), asking two separate questions: how far does the outcome
move, and **does any conclusion flip?**

```
.venv/Scripts/python.exe scripts/wp10_sensitivity.py
```

## 1. The conclusions hold — with one exception

Across **70 settings**:

* `value-greedy >= FIFO` — **violated 0 times**. The core scheduling claim is robust to
  every assumption in the table.
* `ours >= fair fixed-patch baseline` — **violated once**: at `thumb_value = 0.1` (ten times
  baseline) under congestion, we fall 0.006 behind.

That single exception is informative rather than embarrassing. Thumbnails outnumber ship
chips roughly four to one, so once a thumbnail is worth 10% of a ship the scheduler starts
spending the link on the safety net instead of on ships:

| `thumb_value` | ships delivered (160k) | vs fair baseline |
|---|---|---|
| 0.001 – 0.01 | 0.648 | +0.027 |
| 0.05 | 0.634 | +0.013 |
| **0.10** | **0.615** | **−0.007** |

The baseline of 0.01 sits a factor of five below where the harm starts. That is a comfortable
margin, but it is a margin, not an accident — and it should be stated as a constraint:
**the thumbnail must stay worth less than ~5% of a ship or it competes with the cargo.**

## 2. Cloud fraction dominates — but it is a denominator, not a weakness

| cloud | ships delivered (40k) | ours − fair baseline | greedy − FIFO (160k) |
|---|---|---|---|
| 0.00 | 0.819 | +0.073 | +0.480 |
| 0.15 (baseline) | 0.680 | +0.063 | +0.378 |
| 0.50 | 0.396 | +0.035 | +0.123 |

Recall swings by **0.42** — more than every other constant combined. But ships on cloud tiles
are kept in the denominator on purpose (WP6), and the cloud gate discards those tiles for
*every* strategy, so this moves the absolute number and not the comparison: our margin over
the fair baseline stays positive at every cloud level, and greedy still beats FIFO everywhere.

The honest reading: **the most load-bearing number in the project is one we invented, and it
sets the absolute recall we can claim.** Any headline recall figure must be quoted with the
assumed cloud fraction attached. The Airbus set cannot settle it — it is a ship-finding
benchmark, 0.4% cloud — so a cloud climatology for the target latitudes is the right source,
and that is a literature value, not something we can measure here.

## 3. `dark_weight` is now inert for overall recall — the WP9 fix worked

| `dark_weight` | ships delivered (160k) | dark vessels delivered |
|---|---|---|
| 1.0 | 0.6476 | 0.641 |
| 2.0 | 0.6477 | 0.668 |
| **5.0** | 0.6476 | **0.675** |
| 10.0 | 0.6476 | 0.675 |
| 20.0 | 0.6476 | 0.675 |

Overall recall moves by 7e-5 across a twentyfold change in the weight, while dark-vessel
recall rises and then **saturates at 5.0**. So the baseline sits exactly at the plateau: high
enough to get the full dark-vessel benefit, with nothing to gain from going higher and no
cost to overall recall.

This is a direct check on the WP9 fix. Before decoupling, raising the weight traded overall
recall away (−2.9 points at 9.5). Now the weight only reorders dark vessels among themselves,
which is what "priority, not accusation" was supposed to mean all along.

`dark_wake_value`, the constant WP9 introduced, moves recall by 0.011 at worst — it is a
minor knob, and 1.0 is unremarkable within it.

## 4. `aging_per_hour` is doing almost nothing

| constant | recall range at 160k |
|---|---|
| cloud | 0.365 |
| thumb_value | 0.033 |
| p_dark | 0.020 |
| conf_high | 0.018 |
| dark_wake_value | 0.011 |
| **aging_per_hour** | **0.00024** |
| dark_weight | 0.00007 |

Sweeping aging from 0 (off) to 2.0 per hour changes delivered ships by 2.4e-4 — within noise.
The aging term exists to stop old items being starved, and on this workload nothing is
starved long enough for it to matter, because the level-of-detail payload is small relative
to a pass. It is not wrong, it is simply not earning its place. Either drop it, or justify it
on a workload where starvation actually occurs (that would be the dense-scene mix, unswept
here). Reporting a mechanism that provably does nothing is worse than not having it.

## 5. What to say in the paper

1. Quote absolute recall **with the assumed cloud fraction attached**; it dominates.
2. State that the comparative claims were swept over 70 settings and held in 69.
3. State the constraint `thumb_value < ~0.05`, and why.
4. `dark_weight = 5` is at a saturation plateau, not a guess that happened to work — and,
   after WP9, it costs nothing in overall recall.
5. Either remove `aging_per_hour` or stop claiming it does anything.

## Limits

* One-at-a-time sweep: interactions are not explored. `thumb_value` and cloud fraction plainly
  interact (more cloud means fewer thumbnails), and that pair is worth a 2-D sweep.
* One seed per point. WP6 measured the rep-to-rep spread at ±0.005, so movements smaller than
  that — `aging`, `dark_weight` on overall recall — are at the noise floor, which is exactly
  the conclusion drawn about them.
* `conf_low` is not swept here; WP6 swept it separately as the onboard detection threshold.
