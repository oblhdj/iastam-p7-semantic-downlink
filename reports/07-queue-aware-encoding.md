# WP8 -- Let the buffer decide how much detail a coastal tile deserves

> **REFRESHED 24 Sept 16:32** (post-WP9, and post-`conf_high` 0.670). Queue-aware worst-case
> regret **0.0058** (was 0.0101 → 0.0075), mean **0.0025**; the best fixed policy is now
> "adaptive, fixed >=3" at **0.0123** worst / 0.0072 mean. Ours is still lowest on **both**,
> and the worst case improved again. See `wp8_queue_aware.csv`.


WP7 left an unfinished idea. Adaptive coastal detail (escalate to a whole tile only where
the classic stage saw objects the network did not confirm) **wins under congestion and loses
at nominal load** -- because a fixed threshold cannot know which regime it is in.

The satellite can. The buffer is right there when the compressor runs, and the pass schedule
comes off its own ephemeris. WP8 uses both.

```
python scripts/wp8_queue_aware.py
python scripts/wp8_queue_aware.py --link-share 0.15     # held-out check
```

## 1. What changed in the code

`simulate` took a finished list of items: every tile had been compressed before anything was
known about the downlink. That is an artefact of offline simulation, not of the satellite.
`simulate_online` walks the tiles in capture order, runs the passes as they fall due, and
encodes each tile knowing the current queue:

    pressure = bytes queued / link capacity in the next 12 h

Both terms are onboard quantities. `coast_mode="queue"` slides the escalation threshold
linearly between `esc_min` (plenty of room: send every coastal tile whole, keep the safety
net) and `esc_max` (hopeless backlog: whole tiles would only crowd out ship reports).

`encode_lod` and `simulate_online` share one per-tile encoder (`encode_tile`), and a test
asserts the online path reproduces the batch path exactly whenever the coastal policy is
fixed -- otherwise WP8 would be measuring the refactor rather than the idea.

## 2. No fixed policy is right in more than one regime

Ship recall, real detections, value-greedy, link share 0.25:

| tiles/day | whole tile always | fixed >=3 | fixed >=11 | fixed >=25 | mosaic only | **queue-aware** |
|---|---|---|---|---|---|---|
| 20,000 | **0.686** | 0.679 | 0.671 | 0.661 | 0.621 | **0.686** |
| 40,000 | **0.682** | 0.675 | 0.666 | 0.655 | 0.621 | **0.682** |
| 80,000 | **0.687** | 0.679 | 0.671 | 0.661 | 0.624 | 0.679 |
| 160,000 | 0.649 | 0.658 | **0.665** | 0.658 | 0.623 | 0.664 |
| 320,000 | 0.574 | 0.587 | 0.601 | 0.613 | **0.623** | 0.613 |

The best fixed policy moves three times across the range: whole tiles up to 80k, escalate-at-11
at 160k, mosaic-only at 320k. Committing to one in advance is a bet on the duty cycle.

**Regret** -- how much recall a fixed choice costs when the regime turns out differently:

| policy | worst case | mean |
|---|---|---|
| **queue-aware (ours)** | **0.011** | **0.004** |
| adaptive, fixed >= 11 | 0.022 | 0.014 |
| adaptive, fixed >= 25 | 0.027 | 0.019 |
| adaptive, fixed >= 3 | 0.036 | 0.013 |
| whole tile always | 0.049 | 0.013 |
| mosaic only | 0.066 | 0.046 |

Queue-aware is within 1.1 points of the best policy at every load, without being told which
load it is running at. The nearest fixed policy is twice as bad in the worst case and three
times as bad on average.

## 3. Held-out check (this matters -- the law has four fitted parameters)

`esc_min`, `esc_max`, `pressure_lo` and `pressure_hi` were fitted to five measured points at
link share 0.25, which is exactly the setup where a control law can flatter itself. So the
same, unchanged parameters were run against two link budgets they were never tuned on --
changing the capacity changes the pressure scale, which is what the law keys on:

| link share | queue-aware worst case | best fixed policy | worst case |
|---|---|---|---|
| 0.25 (tuned) | **0.011** | fixed >= 11 | 0.022 |
| 0.15 (held out) | **0.027** | fixed >= 25 | 0.030 |
| 0.40 (held out) | 0.011 | fixed >= 3 | 0.010 |

It keeps the lowest **mean** regret in all three (0.004 / 0.007 / 0.003) and the lowest worst
case in two, tying at 0.40. Note that the best *fixed* policy is a different one in each row,
which is the point: you cannot pick it in advance, and queue-aware never has to.

This is not a large win in absolute recall -- about 1 point over a well-chosen fixed threshold.
It is a robustness win, and it should be claimed as one.

## 4. Why this is the right shape for the project

The two contributions were separate mechanisms: a level-of-detail encoder and a value-aware
scheduler, evaluated one after the other. They are now one control loop -- the scheduler's
state determines the encoder's decisions, and the encoder's output is what the scheduler
ranks. The project's own thesis, "spend bytes where the model is unsure", now also answers
"...and only as far as the link can carry them".

It also costs nothing to run: pressure is a division, the threshold is a clamp, and the
uncertainty signal (unconfirmed candidates) was already computed by the pre-filter that runs
on every tile anyway.

## 5. Limits, stated plainly

* The law is linear in one signal. A queue-length controller would normally be tuned against
  a model of the arrival process; ours is fitted to five points and checked on two more.
* Pressure uses a 12 h horizon; that was not swept.
* The gain over a well-chosen fixed threshold is ~1 point of recall. The honest claim is
  robustness across regimes, not a higher peak.
* i.i.d. tile resampling (WP6 limit) still applies: real passes see contiguous strips, so
  real pressure would be burstier than modelled, which if anything favours adapting.
