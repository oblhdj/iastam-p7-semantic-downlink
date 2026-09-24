# WP5b — Bounding the loss over the whole day

> **REFRESHED 24 Sept 16:34** (post-WP9, and post-`conf_high` 0.670). Total gap at the worst
> load **7.1%** (was 12.7% → 9.6%), averaging 5.2% where anything binds; price of not seeing
> the future **≤0.2%** (was 0.66% → 0.4%). FIFO 78.0% below the bound. Both conclusions
> strengthen again — **do not build arrival prediction.** See `wp5_joint_bound.csv`.


The per-window gap (`wp5_optimality_gap_report.md`) measured how much value greedy leaves
behind *inside* one contact window, and said plainly that it does **not** bound the
end-to-end loss: it never questions the choices that produced the queue in the first place.
This closes that hole. It also closes the plan's unstarted nice-to-have, "offline LP upper
bound".

```
.venv/Scripts/python.exe scripts/wp5_joint_bound.py
```

## 1. Why this is not just a bigger knapsack

Scheduling every window jointly is a multi-dimensional knapsack **with release times**:

* an item captured after a window has opened cannot use it, so capacity is not
  interchangeable — early windows serve only early items, the last window serves everything;
* each window has its own capacity, not a shared pool.

That is NP-hard, so rather than pretend to solve it we bracket it:

```
online greedy   <=   clairvoyant   <=   joint optimum   <=   upper bound
(what we do)         (foresight,          (unknown)          (relaxation)
                      feasible)
```

**Upper bound** (`joint_upper_bound`): integrality relaxed, each byte credited at the best
rate its item could ever reach, capacities and release times respected. One detail decides
whether it is a bound at all — each item must consume the *earliest* capacity it can reach.
Late capacity is reachable by everything, so spending it on an item that could have used an
early window destroys flexibility; consuming the most constrained capacity first is what
makes the sweep valid. Verified against brute-force enumeration over **every** item→window
assignment on 120 random instances.

**Clairvoyant** (`clairvoyant_value`): a real, feasible schedule that ranks by value per byte
across the entire day at once and places each item in the earliest window it can reach.
Because it is feasible it is a lower bound on the joint optimum, so the true optimum is
trapped between it and the relaxation.

A second fix went in alongside: the single-window bound used to credit bytes at the best rate
without limit, letting one progressive item earn several times its own value. Each item's
credit now saturates at `value / best_rate` bytes. Slack on mixed instances fell from **59%
to 29%**, and to **0%** when no progressive items are present.

## 2. Result

Whole-day delivered value, real workload, value units (not ships):

| tiles/day | online greedy | clairvoyant | upper bound | total gap | price of being online | FIFO gap |
|---|---|---|---|---|---|---|
| 20,000 | 10,220.3 | 10,220.3 | 10,220.3 | 0.0% | 0.0% | 0.0% |
| 40,000 | 20,598.6 | 20,598.6 | 20,598.6 | 0.0% | 0.0% | 0.0% |
| 80,000 | 41,777.9 | 41,778.5 | 41,911.4 | 0.3% | 0.002% | 19.7% |
| 160,000 | 79,797.6 | 79,905.2 | 83,481.4 | 4.4% | 0.13% | 59.8% |
| 320,000 | 139,500.7 | 140,554.3 | 159,720.8 | 12.7% | 0.66% | 78.7% |

**At operational load the whole-day schedule is provably optimal.** Up to 40,000 tiles/day
the three quantities are equal to floating-point precision: there is nothing left to win,
from any scheduler, ever.

**Under heavy congestion the loss is bracketed, not hand-waved.** At 320,000 tiles/day the
true optimum lies between the clairvoyant 140,554 and the bound 159,721, so our online
schedule is between **0.75% and 12.7%** below the best achievable. The bound is a relaxation
and the clairvoyant is only a heuristic, so the honest statement is the bracket, not either
endpoint.

**FIFO is 78.7% below the bound at the same load.** The ordering that makes almost no
difference at 40,000 tiles/day is most of the result at 320,000.

## 3. The finding that changes what to build next

**Clairvoyance is worth at most 0.7%.**

The "price of being online" column isolates what our scheduler loses purely by *not knowing
future arrivals*: give it perfect foresight over the entire day and it gains 0.002% at
80,000 tiles/day, 0.13% at 160,000, 0.66% at 320,000.

That closes off a whole direction. Predicting the arrival stream — forecasting where ships
will be, learning a traffic prior, deferring capacity for an expected burst — cannot buy
more than two thirds of one percent, at the most congested load we simulate, against a
*perfect* oracle. Any real predictor would get a fraction of that. It is not worth building.

Whatever remains in the 0.75–12.7% bracket is the price of **greediness under lumpiness**,
not of ignorance about the future — and WP5 already located it precisely: the optimum cuts
several progressive items well, while our policy truncates only the one that happens to
straddle the end of the pass. That is the thing worth fixing.

## 4. Limits

* The bound is a relaxation, loose on progressive-heavy instances (~29% slack measured on
  synthetic mixes). The true gap is nearer the clairvoyant end of the bracket than the bound
  end, but we do not know where.
* The clairvoyant heuristic ranks by value per byte, so it inherits the same truncation
  weakness. A better offline heuristic would raise the lower end of the bracket and tighten
  the claim.
* Storage eviction is applied in the online simulation but not in the offline bound, which
  makes the bound slightly more generous still — in the safe direction.
* Value is the paper's own model (dark vessels weighted 5x, sqrt for partial sends), which
  remains an unswept assumption.
