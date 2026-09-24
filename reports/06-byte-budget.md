# WP7 -- Spending the byte budget where it buys ships

> ### ⚠ CORRECTION 24 Sept — "q60 → q40, no recall cost" is wrong
> That claim was measured inside the scheduler simulation, where each ship's recall comes from a
> confidence computed on the **original** file — so recompression could not possibly show up in
> it. [Report 14](14-coastal-recompression.md) re-runs the **detector** on recompressed tiles:
> **q40 costs 4.0 points of ship recall on coastal tiles**, essentially all of it on small ships
> (0.476 → 0.398). The byte saving is real; the "no cost" was an artefact of the measurement.
> Say it as a trade. q30 is rejected there too, on measurement rather than caution.
>
> Thumbnail gating has also been re-keyed onto the learned gate: **−3.8% of total bytes at zero
> measured recall cost** — see [report 15](15-gate-signal-rekey.md).

> **REFRESHED 24 Sept 15:30 (post-WP9).** Byte-budget tables re-run with the decoupled
> dark-vessel encoder; see `wp7_budget_sweep.csv` for current values.
> ⚠ The **"421x to 517x"** data-reduction figure in §below is superseded: the current
> value is **557x** (WP9 added a dark-vessel L2 follow-up; WP2's `conf_high` 0.670 then cut
> 8.4% of bytes). `wp6_real_table.csv` is authoritative. Re-run 24 Sept 16:31.


WP6 put the scheduler on real detections. The first thing that fell out of it was
uncomfortable: **86% of our "semantic downlink" is not semantic.**

| item | count/day | MB | share |
|---|---|---|---|
| coastal tiles | 2,060 | 88.4 | **61.7%** |
| thumbnails | 33,975 | 34.0 | **23.7%** |
| L1 chips | 8,721 | 11.7 | 8.2% |
| L2 ROIs | 1,110 | 7.0 | 4.9% |
| false alarms | 2,393 | 2.2 | 1.5% |
| L0 metadata | 986 | 0.04 | 0.03% |

Ship information is 13%. WP7 measures the cheaper alternatives on real tiles and puts them
through the real workload, so the design is chosen on evidence rather than on instinct.

```
python scripts/wp7_measure_cheap_products.py     # -> wp7_cheap_products.csv|.json|.png
python scripts/wp7_budget_sweep.py               # -> wp7_budget_sweep.csv|.png
python scripts/wp7_budget_sweep.py --tiles 160000
```

## 1. What the cheaper products actually cost (400 real tiles)

**Thumbnails have a floor.** JPEG headers dominate at this size, so shrinking barely helps
and costs a lot of what the thumbnail is *for* -- being able to see a ship the detector missed:

| thumbnail | bytes (median) | ships still visible |
|---|---|---|
| 48 px q60 | 727 | 0.476 |
| 64 px q60 | 793 | 0.538 |
| 96 px q60 (current) | 984 | 0.615 |
| **128 px q30** | **1,037** | **0.643** |
| 128 px q60 | 1,278 | 0.683 |

Going *down* from 96 px to 48 px saves 26% of the bytes and loses 23% of the visible ships --
a bad trade. Going *up* to 128 px q30 costs the same as 96 px q60 and shows more ships:
at this scale **resolution is cheaper than quality**. The only real lever on the 24% is
therefore sending fewer thumbnails, not smaller ones.

**Coastal tiles compress well, and a mosaic is far cheaper still:**

| coastal product | bytes | PSNR |
|---|---|---|
| whole tile q80 | 76.9 kB | 46.7 dB |
| whole tile q60 (interim) | 41.8 kB | 40.3 dB |
| whole tile q50 | 39.4 kB | 39.3 dB |
| **whole tile q40 (adopted)** | **31.4 kB** | **37.9 dB** |
| whole tile q30 | 25.2 kB | 36.8 dB |
| ROI mosaic x1.5 q60 | 2.3 kB | -- (detected ships only) |

(bytes and PSNR measured on the same 400 tiles, medians)

## 2. Two changes adopted (measured, no recall cost)

1. **Coastal tiles q60 -> q40**: -24% on the largest item in the budget. PSNR 40.3 -> 37.9 dB.
2. **Gated thumbnails**: skip the thumbnail only where two independent opinions agree there
   is nothing -- the classic pre-filter sees no candidate *and* the detector fired nothing --
   keeping a 2% audit sample so silent domain shift is still visible. On the real catalogue
   this clears 21% of the empty tiles and puts 5 of the 1,134 missed ships at risk.

Together: **140.9 -> 114.6 MB/day (-19%) with identical recall (0.682)**, and the data
reduction against a bent pipe improves from 421x to **517x**. The saving is modest because
the classic pre-filter's high false-candidate rate (WP3's negative result) is what limits
the gate -- a learned gate would widen it, which is the strongest argument yet for WP3.

q30 (25.2 kB, 36.8 dB) is measured and tempting, but adopting it needs a check that the
detector still finds ships in a recompressed tile. That needs the torch environment.

## 3. Adaptive coastal detail -- the level-of-detail idea applied to the tile

Our own thesis is "spend bytes where the model is unsure", but coastal tiles were
all-or-nothing: every one downlinked whole, at 62% of the budget. The uncertainty signal was
already onboard and free -- **the classic pre-filter and the network disagree**. Bright
objects the cheap stage found and the network did not confirm ("unconfirmed candidates")
predict where the network missed a ship, measured on the 1,415 real coastal tiles:

| unconfirmed candidates | tiles | P(tile hides a missed ship) | missed ships there |
|---|---|---|---|
| 0 | 249 | 0.173 | 49 |
| 1-2 | 155 | 0.290 | 59 |
| 3-5 | 152 | 0.329 | 75 |
| 6-10 | 150 | 0.300 | 52 |
| **11+** | **709** | **0.413** | **699** |

Half the coastal tiles hold three quarters of the missed ships. So: send the cheap mosaic
normally, escalate to the whole tile only above a threshold.

**At nominal load (40,000 tiles/day) escalation is not worth it** -- the link is not the
bottleneck, so giving up any recall to save bytes is a bad deal:

| coastal policy | ships delivered | MB offered |
|---|---|---|
| whole tile q40 | **0.682** | 114.6 |
| whole tile q30 | **0.682** | 101.9 |
| adaptive, escalate >= 11 | 0.666 | 84.0 |
| mosaic only | 0.621 | 53.0 |

**Under congestion (160,000 tiles/day) it wins outright** -- bytes convert into ships, and
adaptive beats every fixed policy on *both* axes at once:

| coastal policy | ships delivered | MB offered | median latency |
|---|---|---|---|
| whole tile q60 (interim) | 0.638 | 538.7 | 5.23 h |
| whole tile q40 | 0.650 | 455.0 | 5.14 h |
| whole tile q30 | 0.661 | 404.9 | 5.10 h |
| **adaptive, escalate >= 11** | **0.666** | **333.3** | **4.85 h** |
| adaptive, escalate >= 25 | 0.658 | 294.0 | 4.64 h |
| mosaic only | 0.623 | 212.8 | 4.39 h |

More ships than sending every tile whole, for 62% of the bytes and shorter latency. It also
beats escalating at random: a coin-flip on half the tiles would land near 0.65, the signal
buys the extra ~1.6 points.

**Not adopted as the default**, because at the nominal design point it costs 1.6 points of
recall for bytes we do not need. It is the right policy exactly when the buffer is filling,
which points at the obvious next step: **make the escalation threshold a function of queue
depth**, closing the loop between the level-of-detail encoder and the scheduler. The
satellite knows its own buffer state, so this costs nothing to implement.

## 4. Honest correction: the fixed-patch baseline was unfair, and fixing it costs us

The interim report's own self-criticism table listed as weakness #2 that the Phi-sat-2-style
baseline ignores coastal ships by construction. It did, and WP6 inherited it. Fixed: the
baseline now images coastal scenes too (and pays for coastal false alarms).

| baseline | ships delivered | MB sent |
|---|---|---|
| Phi-sat-2 style, ocean only (as reported at interim) | 0.518 | 13.5 |
| **Phi-sat-2 style, fair (+ coastal)** | **0.622** | 16.3 |
| Ours: LoD + value-greedy | 0.685 | 116.3 |

So our advantage over a *fair* fixed-patch baseline is **+6.3 points of ship recall for about
7x the bytes**, not the +16.7 points the unfair version suggested. Those 6.3 points are
exactly the ships the detector never saw, recovered by downlinking coastal tiles -- which is
a real contribution, but a much narrower one than the interim figure implied. It must be
stated this way.

Under congestion the picture is better for us, because the baseline cannot use spare
capacity: at 160,000 tiles/day ours delivers 0.648 against the fair baseline's 0.621, and
against FIFO on our own payload (0.272) it is still **2.4x**.

## 5. What is now worth doing next

1. **Queue-aware escalation** (above) -- the two contributions become one mechanism.
2. **WP3 learned gate** -- it is the limiter on thumbnail gating *and* it would sharpen the
   unconfirmed-candidate signal that drives adaptive escalation. Needs torch.
3. **Detector-on-recompressed-tiles check**, which unlocks q30 (a further -18% on coastal).
4. The remaining loss is still the detector: 0.685 of ships delivered against a 0.685 ceiling.
