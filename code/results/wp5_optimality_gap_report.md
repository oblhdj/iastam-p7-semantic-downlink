# WP5 -- How far from optimal is the greedy scheduler?

> **REFRESHED 24 Sept 16:33** (post-WP9, and post-`conf_high` 0.670). Saturated small windows
> **6.67%** mean / 54.8% worst; operational scale **0.251%**. FIFO 67.3% mean / 93.0% worst.
> ⚠ **The gap grew** (was 4.62% / 0.069%) and that is expected: the cheap 40 B L0 rung now
> fires 7x more often, so item sizes are more heterogeneous and the knapsack is lumpier.
> **Buying an 8.4% byte saving made the scheduling problem slightly harder.** The conclusion
> is unchanged — 0.251% at operational scale is negligible and FIFO still loses two thirds —
> but say the trade out loud. See `wp5_optimality_gap.csv`.


The technical plan lists **"optimality gap on small passes"** under *must have* and ticks it.
The technical report states that greedy by value-per-byte "is optimal for divisible items
with linear value, but not for indivisible items, so we will measure the optimality gap".

Neither had been done: there was no solver anywhere in the code, only the promise. This
closes it.

```
python scripts/wp5_optimality_gap.py --frac-steps 16
```

## 1. The problem being solved

Per contact window:

    maximise   sum over whole items       v_i * x_i         x_i in {0, 1}
             + sum over progressive ones  v_j * sqrt(f_j)   f_j in {0} u [0.1, 1]
    subject to sum s_i x_i + sum s_j f_j  <=  capacity

A knapsack with a concave-divisible class, so greedy is not optimal in general.
`sat7/optimum.py` solves it exactly by dynamic programming over byte buckets, with each
progressive item discretised into truncation steps.

**Three things keep this honest:**

* **Verified against brute force**, which is the real check. `best_value` is checked against
  exhaustive enumeration on 25 random whole-item instances *and* on 15 mixed instances
  containing progressive items, where the brute force enumerates every truncation choice.
  On the identical discretised instance the DP must match the enumerated optimum exactly,
  and it does. Plus the textbook counterexample: greedy takes a density-7/6 item worth 7.0,
  the optimum takes two items worth 10.0.
* **Bounded from above, but loosely.** A fractional relaxation credits every byte at the
  best rate its item could ever achieve, and the DP never exceeds it. Progressive items are
  credited at `v / (s * sqrt(0.1))`, not the linear rate: sqrt is concave, so the *first*
  bytes of a progressive item are the richest, and a linear bound would not be a bound at
  all. **This bound is tight only for whole items** -- measured slack is 0% there but ~50%
  on progressive-heavy instances, so it catches gross errors and nothing subtler. It is a
  guard rail, not the verification; the brute-force cross-check is.
* **The measured gap is a lower bound.** Byte bucketing charges rounded-up weights, so the
  DP explores a subset of feasible solutions. It can only *hide* a gap, never invent one.

## 2. The answer depends entirely on how lumpy the window is

**At operational scale the gap is negligible.** A typical window holds ~13,800 items, almost
all of them 0.9-2.5 kB chips inside a 25-60 MB pass. Nothing is large relative to the
capacity, so the instance is nearly fractional -- and greedy *is* optimal for fractional
knapsack. Measured: **0.09% mean**, worst single window 2.5%.

| tiles/day | value-greedy | FIFO | windows saturated |
|---|---|---|---|
| 20,000 | 0.000% | 0.000% | 0% |
| 40,000 | 0.000% | 2.1% | 17% |
| 80,000 | 0.05% | 28.3% | 83% |
| 160,000 | 0.32% | 52.6% | 83% |

**On small, lumpy windows it is real.** Miniature instances (items subsampled, capacity
scaled by the same byte fraction, bucket size chosen so the smallest item spans at least two
buckets -- so the DP resolves everything exactly):

| tiles/day | value-greedy | worst window | FIFO |
|---|---|---|---|
| 20,000 | 0.000% | 0.000% | 0.000% |
| 40,000 | 0.000% | 0.000% | 5.1% |
| 80,000 | 4.6% | 33.4% | 57.9% |
| 160,000 | 11.8% | 26.2% | 70.8% |

**Headline, over the 22 saturated small windows where the choice actually binds: value-greedy
leaves 8.95% of the optimal value on the table on average, 33.4% in the worst window. FIFO
leaves 72.9% on average.**

So the report's claim was right in both directions, and can now be stated with numbers:
greedy is essentially optimal at the scale we operate at, and the ~9% it does lose appears
only when few, large items compete for one window.

## 3. Where the gap actually comes from -- a convergence check

Sweeping how finely a progressive item may be truncated, at 160,000 tiles/day:

| truncation steps | mean gap | worst |
|---|---|---|
| 2 | 6.1% | 12.1% |
| 4 | 10.2% | 18.6% |
| 8 | 14.1% | 25.9% |
| 16 | 14.7% | 26.2% |

The estimate rises and then converges by 16 steps, which says two useful things. The coarse
settings **understate** the gap (as predicted -- fewer options can only weaken the DP), so 4
steps would have been an under-report. And the gap lives mostly in *truncation choice*: the
optimum picks how far to cut several progressive items, while our policy truncates only the
one item that happens to straddle the end of the pass.

**That is a concrete improvement, not just a measurement.** The greedy policy could choose
truncation fractions across several coastal tiles instead of filling with whatever lands
last. Worth doing in Phase 3 -- it is the only place we have found where the scheduler
itself, rather than the detector, is leaving value behind.

## 4. Limits

* This is the **per-window** gap: given the queue greedy got itself into, how much did it
  leave behind in that window. The joint problem -- scheduling all six passes at once with
  perfect knowledge of future arrivals -- is harder and is not solved here. The per-window
  gap does not bound it.
* Miniature instances preserve the size, value and saturation mix, but they are samples; at
  20k-40k the subsample shrinks to ~25 items, where nothing binds and the gap is trivially 0.
* Value is the paper's own value model (dark vessels weighted 5x, sqrt for partial sends).
  A different value model would give a different gap; that model is itself an assumption
  that has never been swept.
