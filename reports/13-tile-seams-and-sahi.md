# 13 — Tile seams and SAHI, measured instead of derived

Closes `../docs/ARCHITECTURE.md` §E item 2: *"`sahi_cost.png` is a geometric formula, not an
experiment."* Scripts: `../code/scripts/wp12_seams.py`, `../code/scripts/wp12b_cut_recall.py`.
Data: `../code/results/wp12_seams.json|.csv|.png`, `../code/results/wp12b_cut_recall.json|.csv|.png`.
All figures **REAL** — the trained detector on real Airbus tiles.

## Why this was parked, and why it had to be reopened

The Airbus tiles are **already 768 px and already cut**, so SAHI changes nothing for any number
in this project, and the Phase-2 guide said so. That was correct — but it quietly assumes the
operational system also receives pre-cut tiles. It will not. A real sensor produces a wide swath,
perhaps 10,000 px across, and something onboard has to slice it before the 768 px detector sees
it. At that point every seam is a chance to cut a ship in half.

So the question is real, and it had only ever been answered with arithmetic on box sizes.

## 1. The geometry: big ships are the ones that get cut

For a 10,000 × 10,000 px swath tiled at 768 px with no overlap — 196 tiles:

| ship length | P(cut by a seam) |
|---|---|
| median, 43 px | 10.9% |
| 100 px | 24.3% |
| 200 px | **45.3%** |
| largest in Airbus, 380 px | **74.5%** |

This is the part that looks alarming: three quarters of the largest vessels are split, and large
vessels are the highest-value targets in the whole system.

## 2. But cut is not lost — and that is the finding

`wp12b_cut_recall.py` crops a real tile along a line through a real ship so that only a fraction
of its hull stays inside, then runs the detector on that crop — exactly what a tile boundary
does. Matching is against the **clipped** ground-truth box, because finding the visible half
still means the satellite knows a ship is there. 400 ships × 9 fractions = 3,600 detector runs.

| % of ship left inside the tile | overall | small (<32 px) | medium | **large (>96 px)** |
|---|---|---|---|---|
| 20% | 0.115 | 0.000 | 0.124 | 0.211 |
| 40% | 0.407 | 0.016 | 0.419 | 0.754 |
| **50%** (worst case for a seam) | 0.557 | 0.101 | 0.620 | **0.915** |
| 70% | 0.723 | 0.341 | 0.798 | **1.000** |
| 100% (intact) | **0.812** | 0.527 | 0.891 | 1.000 |

**Large ships barely notice being cut in half** (0.915 at 50% visible, 1.000 by 70%). Small ships
are destroyed by it (0.101 at 50%) — but small ships are only cut 10.9% of the time, and large
ships 74.5%.

**The two effects cancel.** The ships most likely to be cut are the ones most robust to it.
Expected loss on a 10k swath with no overlap at all:

| ship size | P(cut) | ships actually lost |
|---|---|---|
| median 43 px | 10.9% | **1.1%** |
| 100 px | 24.3% | **2.4%** |
| 200 px | 45.3% | **4.5%** |
| 380 px | 74.5% | **7.5%** |

So the honest size of the problem is **1–7.5% of ships**, not the 10–75% the geometry alone
suggests.

## 3. Four policies, measured end to end

`wp12_seams.py` takes each real 768 px tile as a miniature "frame" and cuts it itself, so the
straddling ships are real and the detector is the real one. 200 tiles, 373 ships, **15.0% of them
straddle the 2×2 seam**. Every slice is inferred at the model's native 768 px, so compute is
proportional to the number of slices.

| policy | slices/frame | compute | **recall on seam ships** | recall off seam | overall |
|---|---|---|---|---|---|
| whole tile (never cut) | 1.00 | 0.25× | 0.9643 | 0.7729 | 0.8016 |
| naive 2×2 grid | 4.00 | 1.00× | **0.6786** | 0.6940 | 0.6917 |
| SAHI, 20% overlap | 9.00 | 2.25× | **0.8750** | 0.7634 | 0.7802 |
| **edge-triggered** | 5.51 | **1.38×** | **0.8750** | 0.7003 | 0.7265 |

* Cutting a tile costs **28.6 points of recall on the ships that straddle the cut** — the seam
  problem is real once you actually tile.
* **Edge-triggered ties SAHI exactly on seam recall (0.8750 both) at 61% of the compute.** This
  is the project's own proposed alternative, and it holds up: spend the extra inference only
  where the cheap classic stage already sees a blob touching a border.
* SAHI wins on *overall* recall (0.780 vs 0.727) because its extra slices also give second looks
  at ships nowhere near a seam. That is a genuine benefit, but it is not what the policy is for,
  and it is paid for everywhere.

> **Scale caveat, stated plainly.** Cutting a 768 px tile into 384 px quarters is a harsher
> regime than tiling a 10k swath at 768 px: the slices are smaller relative to the ships, so both
> the seam frequency (15.0% here vs 10.9% for a median ship at real tiling) and the SAHI overhead
> (2.25× here vs 1.47× on a real swath) are exaggerated. Read the *ordering* of the policies as
> the result; read the magnitudes as an upper bound on the problem.

## 4. And the compute objection has evaporated anyway

SAHI was rejected in the Phase-2 guide because "1.51× compute is expensive where energy is the
constraint." [Report 12](12-end-to-end-integration.md) measured the onboard budget for the first
time, and it is not the detector that dominates it:

| configuration | ms/tile |
|---|---|
| today: classic pre-filter 56.3 + detector 11.1 | **67.4** |
| learned gate 0.15 + detector 11.1 | **11.3** (6.0× cheaper) |
| learned gate + SAHI at 1.47× detector | **16.5** (4.1× cheaper than today) |

**Dropping the classic pre-filter frees 56 ms/tile; SAHI on a real swath would spend 5.2.** The
swap pays for SAHI ten times over — and the swap is worth making on its own, since the learned
gate is also 43 points more accurate ([report 04](04-onboard-gate.md)).

## 5. Recommendation

1. **Do not implement blanket SAHI.** It buys at most 7.5% of ships on the largest vessels and
   pays for it on every tile of every frame.
2. **Implement edge-triggered re-inference** when the system moves to wide swaths. It matched
   SAHI on the only metric that matters here, at 61% of the cost, and it reuses two stages that
   already run.
3. **Do it after the interim deadline.** Tiling touches the whole pipeline and none of the
   current results depend on it. Present this as measured Phase-3 work: the geometry, the
   cut-recall curve, the policy comparison and the compute budget are all here.
4. Note the one thing this does *not* cover: **small ships on seams are unrecoverable by any
   tiling policy** (0.101 at 50% visible). If small-ship recall matters more later, the answer is
   a better detector, not a better tiling.
