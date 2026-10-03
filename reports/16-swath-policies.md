# 16 — The four tiling policies on real swaths: SAHI wins at scale

Closes the scale caveat [report 13](13-tile-seams-and-sahi.md) raised against itself, and builds
the **global-coordinate transform + detection fusion** that `../docs/START_HERE.md` §3 lists as
*not implemented* (the paper's B2 centrepiece). Script: `../code/scripts/wp16_swath_policies.py`.
Data: `../code/results/wp16_swath_policies.json|.csv`, built on the real swaths from
[the swath-stitching step](../code/results/wp15_manifest.json) (`wp15_build_swaths.py`).
**REAL** — real Airbus test tiles, real ground truth, the trained detector, real cross-slice fusion.

## Why this was reopened

Report 13 answered the seam question by taking a **single 768 px tile and cutting it into 384 px
quarters**. It said plainly that this was a harsher regime than a real sensor — the slices are
small relative to the ships — and told the reader to *"read the ordering of the policies as the
result; read the magnitudes as an upper bound."* There was also no fusion step: each cut slice was
scored on its own, so the experiment never had to transform a detection back into a wide frame or
merge detections across a seam.

The operational system does exactly those two things it skipped. A real sensor produces a wide
swath; something onboard slices it at the detector's native 768 px; and the detections must be
lifted back to swath coordinates and fused before anything downstream sees them. So the question is
re-run here on the real swaths, at native 768 px, with the fusion actually built.

The four policies are unchanged from report 13, now slicing the 3072 px swath (its seams are the
half-tile-offset grid from the stitching step, the grid an onboard slicer lands on because it
cannot see where the source tiles met):

  * **whole** — the regular 4×4 grid aligned to the source tiles: 16 intact tiles, nothing cut.
    This is an **oracle** — a real slicer does not know where the tiles met, so it cannot align —
    but it is the right *reference* for "what if no ship were ever cut".
  * **naive** — the slicer's regular 768 grid, phase-shifted half a tile and clamped at the border
    (25 cells: 3 full + 2 partial per axis). Every cell boundary is a seam, so straddlers are cut.
  * **sahi** — overlapping 768 tiles, 20% overlap (25 slices).
  * **edge** — the naive grid plus a re-run of the intact source tile wherever the classic
    pre-filter sees a blob touching that tile's midline (report 13's own proposed alternative).

## The method bug the sanity gate caught — report it, don't bury it

The `whole` policy slices the swath back into its intact source tiles, so its recall *must* equal
running the detector on those tiles directly. The script checks this before trusting any other
number. The first run **failed** it: whole-policy recall 0.8081 vs direct per-tile 0.8283, a 2.0
point gap on the same 99 ships.

The cause was real, not cosmetic. Cross-slice NMS — the fusion step — was being applied to the
`whole` tiling, whose 16 tiles do **not** overlap. With nothing to de-duplicate, the NMS could only
do harm, and it did: it merged two *distinct* ships sitting close on either side of an abutting
cell boundary into one, dropping a true positive. The fix is to fuse with NMS **only where slices
overlap** (sahi, edge); for non-overlapping tilings (whole, naive) the fusion is a plain
concatenation, since each slice is already de-duplicated internally by the detector. This is one
refinement over report 13, which ran NMS across its non-overlapping naive grid. After the fix the
gate passes to the digit: **0.8283 == 0.8283, Δ = 0.0000**, and the global-coordinate transform is
trustworthy. (This is the same class of bug as [report 12](12-end-to-end-integration.md)'s matching
divergence and [report 08](08-optimality-gap.md)'s tolerance bug: a measurement artefact in our own
code that a cross-check, not caution, exposed.)

## Compute: SAHI's overhead collapses, edge-triggered's does not

24 swaths, **716 ships**, 77 (10.8%) straddle a seam. Compute is slices/swath relative to the
**regular 4×4 tiling** (16 cells) — report 13's "slices relative to the naive grid" convention,
mapped to swath scale. (At report 13's single-tile scale the naive 2×2 *was* the regular tiling, so
the denominators correspond; here the regular tiling is the whole/never-cut grid, and the
seam-cutting naive grid is a separate, costlier 25-cell grid.)

| policy | slices/swath | compute (×regular) | report 13 (×regular) |
|---|---|---|---|
| whole (never cut, oracle) | 16.0 | **1.00×** | 0.25× |
| naive grid | 25.0 | **1.56×** | 1.00× |
| SAHI 20% | 25.0 | **1.56×** | **2.25×** |
| edge-triggered | 34.0 | **2.13×** | **1.38×** |

The one number that needs no denominator, and so compares cleanly across the two experiments:
**edge-triggered costs 1.36× SAHI's compute here, versus 0.61× in report 13.** The ordering flips.
The reason is geometric and not about the detector: SAHI's 20% overlap adds a fixed fractional ring
of slices, which is *asymptotically cheap* on a wide swath (2.25× → 1.56×), whereas edge-triggered
pays the full naive grid — itself already 1.56× here — and then adds re-inference on top.

## Recall: cuts hurt, SAHI recovers most, edge recovers less

| policy | overall | on seam | off seam | small <32 | medium | large >96 |
|---|---|---|---|---|---|---|
| whole (oracle) | 0.774 | **0.909** | 0.757 | 0.601 | 0.916 | 0.993 |
| naive grid | 0.728 | **0.675** | 0.734 | 0.562 | 0.878 | 0.917 |
| SAHI 20% | 0.772 | **0.922** | 0.754 | 0.612 | 0.892 | 0.993 |
| edge-triggered | 0.772 | **0.883** | 0.759 | 0.606 | 0.911 | 0.979 |

* **The seam problem is real.** Cutting at the seams costs **23.4 points** on the ships that
  straddle them (naive 0.675 vs the oracle's 0.909) — close to report 13's 28.6 points, and a
  little gentler, exactly as it predicted for the larger native-scale slices.
* **SAHI recovers the seam loss fully** (0.922, even edging past the oracle, because its overlap
  also gives second looks away from seams) **at 1.56× the regular tiling**.
* **Edge-triggered recovers most but not all of it** (0.883) **and costs more** (2.13×).
* As in report 13 and [report 02](02-detector-and-quantisation.md), **small ships are the ones no
  tiling policy saves** (0.56–0.61 everywhere): the answer there is a better detector, not a better
  slicer.

## This reverses report 13 §5 — and here is why the scale mattered

Report 13 §5 recommended **against** blanket SAHI and **for** edge-triggered re-inference, on the
strength of "edge ties SAHI on seam recall at 61% of the compute." On real swaths with real fusion,
every leg of that recommendation turns over:

* edge no longer **ties** SAHI on seam recall (0.883 vs 0.922, a 3.9 pt gap; report 13 had them
  exactly equal at 0.875);
* edge is no longer **cheaper** — it costs 1.36× SAHI, not 0.61×;
* so **SAHI dominates edge-triggered**: higher seam recall *and* lower compute.

What report 13 got right: the **ordering heuristics at small scale** (seams hurt; a cut ship is
often still found; small ships are unrecoverable) all survive. What it **could not see** from a
single cut tile:

1. **Asymptotic overlap density.** SAHI's cost is a fixed-fraction ring of extra slices. On one
   768 tile cut into 384 quarters that ring is enormous relative to the frame (9 vs 4, 2.25×); on a
   3072 swath it is a thin margin (25 vs 16, 1.56×) and keeps shrinking as the swath grows. Report
   13's 2.25× was the worst case for SAHI, not the operating point.
2. **Real global fusion.** The edge trigger's value is recovering a ship split across a seam. With
   true cross-tile fusion, SAHI's overlapping slice already *contains* that ship intact and fuses it
   for free, so the targeted re-inference has less left to add — while still paying for itself
   everywhere the pre-filter fires.

**Recommendation for Phase 3 (B2): implement blanket SAHI with global-coordinate fusion, at the
detector's native 768 px window.** It is the paper's promised B2 anyway; at the native window it is
the cheaper *and* more accurate seam policy, and the fusion step it needs is now built and
sanity-checked. Edge-triggered is not worth the extra machinery here. This supersedes report 13 §5
— **but the win is specific to the 768 px window; it does not hold at the paper's literal 512 px
spec.** The next section shows the check at both sizes and why B2 should target 768.

## Window size: checked at 512 px (paper) and 768 px (native) — and the winner changes

The accepted paper (START_HERE §3) specifies SAHI at **512×512, 20% overlap**; the numbers above
use **768 px** (the detector's native input, no upscaling). This is not a free choice, so both were
run on the same swaths, same detector, same fusion (`--sahi-window 512`,
`../code/results/wp16_swath_policies_sahi512.json`). A 512 window is upscaled 1.5× to the detector's
768 px input, so a resize effect rides along with the seam story and is separated out below.

| SAHI window | seam recall | compute (×regular) | slices/swath | small | medium | large |
|---|---|---|---|---|---|---|
| **768 (native)** | **0.922** | **1.56×** | 25.0 | 0.612 | 0.892 | 0.993 |
| 512 (paper) | 0.896 | 4.00× | 64.0 | 0.637 | 0.859 | 0.972 |
| *edge-triggered (768)* | *0.883* | *2.13×* | *34.0* | *0.606* | *0.911* | *0.979* |

Edge-triggered takes **no window-size argument** — it is always the native 768 px naive grid plus
intact-tile re-runs — so its 2.13× (34.0 slices/swath) and 0.883 seam recall are **identical in
both runs** (verified across `wp16_swath_policies.json` and `wp16_swath_policies_sahi512.json`). It
is shown once as a fixed reference, not re-measured per window: the only variable moving between the
two SAHI rows is the SAHI window itself, so the comparison is controlled.

**The conclusion does not hold at 512, and it is not averaged.** At 512:

* SAHI's seam recall **falls to 0.896**, now within 0.03 of edge-triggered (0.883) — i.e. they
  **tie**, which is report 13's original finding.
* SAHI's compute **explodes to 4.00×** the regular tiling (64 slices, because a 512 stride of 410 px
  needs 8 windows per axis), so edge-triggered (2.13×) is now **cheaper** — edge/SAHI = 0.53,
  essentially report 13's 0.61.
* So at the paper's literal 512, **report 13's recommendation (prefer edge-triggered) actually
  holds.** The reversal in this report is specific to the native 768 px window.

**The resize effect, stated separately** (512→768 is a 1.5× upscale): it **helps small ships**
(+2.5 pts, magnification makes them easier) but **hurts medium (−3.3) and large (−2.1)**. The large
drop is the key one: a 512 px window **cannot contain a ship longer than 512 px intact**, and large
vessels are exactly the population that dominates seam straddlers (report 13: a 380 px ship is cut
74.5% of the time). So shrinking the window re-introduces the seam problem it was meant to solve,
for the biggest, highest-value targets.

**B2 should target 768 px, not the paper's 512 — and here is why.** It is not that 768 "wins the
bracket": at 512 the detector is forced to re-slice its own native field of view into upscaled
sub-windows, which (a) triples the compute (4.0× vs 1.56×), (b) loses large ships it could otherwise
hold whole, and (c) only buys small-ship recall that reports [02](02-detector-and-quantisation.md)
and [14](14-coastal-recompression.md) say is a *detector* problem, not a slicing one — and buys it
at the expense of the big ships. The 512 default is a generic SAHI setting that predates knowing
this detector's native resolution; for a 768-native detector the correct SAHI window is 768. The
B2 module (`../code/sat7/b2_sahi_fusion.py`) is parameterised by window size so the paper-literal
512 remains reproducible, but its default and recommended setting is 768. **This is a point for the
paper-integration track: the §III SAHI spec should be revised from 512 to 768 for our detector, with
the 512 numbers kept as the published-default comparison.**

## Limits

* **24 swaths / 716 ships is a sample, not the population.** The source tiles are a seeded random
  draw of 384 of the 5,320 real test tiles (ship-bearing, to keep swaths dense and the straddle
  rate comparable to report 13). The **recall magnitudes** carry this sampling error; the straddle
  rate here (10.8%) sits just under the full-population 12.4% measured over all 4,234 test
  ship-tiles in the stitching step, so the sample is not obviously skewed on the quantity that
  drives the result.
* **The compute inversion does not depend on the sample at all.** Slices/swath is fixed geometry
  (16 / 25 / 25 / 34), so edge costing more than SAHI is exact, not estimated. Only the recall gap
  (SAHI 0.922 vs edge 0.883 on seam ships) would move with more swaths. For the recommendation to
  flip back, that 3.9-point seam-recall gap would have to **reverse** at full scale *and* the fixed
  compute ordering would have to stop mattering — neither has been observed, and the geometry makes
  the second impossible. Running all 5,320 tiles (~330 swaths) would tighten the recall gap's
  confidence interval; it has **not** been run here.
* **The `whole` oracle is not operationally achievable** (a real slicer cannot align to invisible
  source-tile edges); it bounds "no ship ever cut" and is not a deployable policy. The real choice
  is among naive / SAHI / edge, and SAHI dominates that set.
* One detector, one IoU-matching rule (0.3 for GT matching inside the slice, 0.5 for cross-slice
  fusion), conf 0.25, 20% SAHI overlap — the same settings as report 13 so the two are comparable.
  The swaths are built from ship-bearing tiles; a realistic mostly-empty swath would change the
  per-swath *counts* but not the per-ship seam behaviour.
