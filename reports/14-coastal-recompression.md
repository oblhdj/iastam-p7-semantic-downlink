# 14 — Does the detector survive a recompressed coastal tile?

Closes the check `../docs/ARCHITECTURE.md` named as blocking q30 coastal tiles. Script:
`../code/scripts/wp13_recompress_check.py`. Data: `../code/results/wp13_recompress.json|.csv|.png`.
**REAL** — 400 real coastal tiles, 748 ground-truth ships, the trained detector.

## Why this check exists

Coastal tiles are the single largest item in the downlink budget ([report 06](06-byte-budget.md)),
and we send them *whole* for one specific reason: the onboard detector may have **missed** a ship
in them, so the ground segment re-runs detection on what arrives. That makes the quality setting
a detection question, not an image-quality question. PSNR says a q40 tile looks fine. It does not
say a detector still finds the same ships in it.

WP7 moved coastal tiles from q60 to q40 and recorded **"no recall cost"**. That measurement came
from the scheduler simulation, where each ship's recall comes from the *catalogue's* stored
confidence — a number computed on the **original** file and completely unaffected by
recompression. The check was therefore structurally incapable of detecting a cost.

## The result: q40 was not free

| quality | kB/tile | vs q60 | PSNR dB | recall | Δ vs original | small | medium | large |
|---|---|---|---|---|---|---|---|---|
| original file | 222.5 | 2.72× | — | **0.6711** | — | 0.4759 | 0.8462 | 0.9906 |
| q60 (interim) | 81.9 | 1.00× | 36.0 | 0.6551 | **−1.60 pts** | 0.4456 | 0.8502 | 0.9811 |
| **q40 (adopted)** | 65.5 | 0.80× | 34.0 | **0.6310** | **−4.01 pts** | **0.3975** | 0.8502 | 0.9906 |
| q30 (proposed) | 56.6 | 0.69× | 32.8 | 0.6176 | −5.35 pts | 0.3924 | 0.8178 | 0.9906 |
| q20 | 43.3 | 0.53× | 31.0 | 0.5642 | −10.70 pts | 0.2886 | 0.8300 | 0.9717 |

⚠ **The adopted q40 setting costs 4.0 points of ship recall on coastal tiles**, not zero. The
damage is almost entirely on **small ships** (0.476 → 0.398, a 16% relative loss); medium and
large vessels are untouched at every quality down to q20.

That pattern is the same one INT8 showed ([report 02](02-detector-and-quantisation.md)): any
lossy step in this pipeline is paid for by the smallest targets, which are already the weakest
bin. Two independent mechanisms, one victim.

## Verdict on q30: **keep q40**

q30 saves a further **13.6% of coastal bytes for another −1.34 points** of recall. Coastal tiles
are ~62% of the budget, so that is ~8% of total downlink for 1.3 points of ships. Given that the
scheduler already has byte headroom at nominal load (the link is not the bottleneck —
[report 05](05-scheduler-on-real-detections.md)), buying bytes with ships is the wrong direction.

**q30 is rejected, and the rejection is now measured rather than deferred.**

## What this means for WP7's adopted change

[Report 06](06-byte-budget.md) should be read with this correction: the q60 → q40 move saves 20%
of the largest budget item and costs **4.0 points of coastal small-ship recall**. It may still be
the right trade — coastal tiles are a safety net for ships the detector already missed, and a
smaller net that reaches further may beat a perfect one that does not — but it is a **trade**, not
a free win, and the report said free.

The honest summary line: *"coastal q60 → q40, −20% of the biggest item, at a cost of 4 points of
small-ship recall on those tiles."*

## Limits of this check
* 400 of the 1,415 real coastal tiles, one detector, one IoU rule.
* It measures re-detection on the **ground**, which is what these tiles are for. It says nothing
  about a human analyst looking at the same tile — a 34 dB tile is still visually fine.
* Re-encoding a JPEG that was already JPEG compounds artefacts. The original Airbus files are
  themselves compressed, so these numbers are the *realistic* double-compression case, which is
  what would actually happen onboard.
