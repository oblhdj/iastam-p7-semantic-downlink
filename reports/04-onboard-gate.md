# WP3 — the tiny learned gate, trained and measured (24 Sept 2026)

Replaces WP3's negative result. The classic CV pre-filter was measured unusable as a gate
(recall 0.645 while dropping 17.7% of empty tiles). A 47k-parameter CNN replaces it.
Script `../code/scripts/wp3_train_gate.py` (3 bugs fixed 24 Sept; had never been run). Data:
`wp3_gate.json`, `wp3_gate_tradeoff.csv|.png`, weights `../code/models/gate.pt`. All figures REAL,
test split, 5,320 tiles / 8,173 ships. 4 epochs, 128 px input, loss 0.2100 -> 0.1458.

## 1. It dominates the classic filter at every operating point — by 35 to 64 points
Same axis for both: x = share of *empty* tiles dropped (compute saved), y = share of ships kept.

| empty tiles dropped | classic CV filter | **learned gate** | gain |
|---|---|---|---|
| 17.7% | 64.5% | **99.6%** | **+35.1 pts** |
| 41.9% | 55.5% | **98.7%** | **+43.2 pts** |
| 58.1% | 33.3% | **97.6%** | **+64.3 pts** |

Read the other way: at **99.0%** ship recall the gate drops **35.6%** of empty tiles
(threshold 0.110) — twice what the classic filter managed while losing a third of the ships.
At **99.5%** recall it still drops 18.4%. The classic filter never reaches either.

## 2. ⚠ Correction: the gate's cost was overstated ~80x by its own script
`wp3_train_gate.py` timed the whole eval loop — 768x768 JPEG decode, resize, transfer, forward —
and wrote it out as `infer_ms_per_tile` = **4.65 ms**. That is dataloader-bound and is **not**
comparable with the detector's 10.71 ms/tile of pure GPU inference. Re-measured against the
saved `../code/models/gate.pt`, forward pass only, batch already on the device (median of 50):

| device | batch 1 | batch 32 | batch 128 |
|---|---|---|---|
| RTX 5060 | 0.570 ms/tile | **0.052 ms/tile** | 0.060 ms/tile |
| CPU | 1.624 ms/tile | **0.940 ms/tile** | — |

The script has been patched to report both (`infer_ms_per_tile` now pure forward,
`eval_loop_ms_per_tile_dataloader_bound` keeps the old figure) and `wp3_gate.json` corrected in
place. **Had the 4.65 ms figure been believed, the gate would have looked like a net loss:**
break-even needs the gate to cost less than 0.904 ms/tile on GPU, and 4.65 exceeds it 5x.

## 3. Break-even: the gate pays, but the ceiling is the dataset, not the gate
Only **20.4%** of test tiles are empty (train 19.8%, val 20.8%), so a *perfect* gate could skip
at most 20.4% of detector calls. Using the val-selected threshold (section 5):

| path | detector alone | gate + detector | net saving |
|---|---|---|---|
| GPU (detector 10.71 ms/tile) | 10.71 ms | 0.052 + 0.9034x10.71 = **9.73 ms** | **9.2%** |
| onboard CPU (ONNX FP32 38.6 ms/tile) | 38.60 ms | 0.94 + 0.9034x38.60 = **35.81 ms** | **7.2%** |

At the safer 99.5% setting the saving is 4.6% (GPU) / 2.7% (CPU). The gate's own cost is **0.5%
of the detector** on GPU and 2.4% on CPU — effectively free.

**State the ceiling honestly.** The 7-9% saving is set by this dataset being 20.4% empty, which
is an artefact of an Airbus *ship-detection* competition set: tiles were selected because they
are interesting. A real swath is mostly empty ocean, where the same gate at the same threshold
would skip most of it. So **7-9% is a floor, not the expected operational figure** — and the
honest way to say it is that the gate's *safety* (98.8% of ships at 40% of empties dropped) is
what was measured here; its *saving* cannot be measured on this data at all.

## 4. What this unlocks downstream
* **WP7 thumbnail gating** used the classic filter's "no candidate" signal to skip thumbnails
  (24% of the byte budget). That signal was 64.5%-recall; it can now be 99.6%-recall at the same
  compute, so the gating can be more aggressive at less risk.
* **WP8's control law** keys on unconfirmed candidates (classic stage saw it, network did not).
  A better-calibrated candidate signal should sharpen the coastal escalation decision.
* Both are follow-on measurements, **not yet run** — this report only establishes the gate.

## 5. Threshold selected on val, quoted on test — the caveat is closed
The first pass read the 99%/99.5% thresholds off the test split, which is marking its own
homework. Redone properly: pick the threshold on **val** (never trained on, never used for
anything until WP2), then quote it on **test**. `wp3_gate_val_selected.json`.

| target | threshold from val | val recall / empties dropped | **test recall / empties dropped** | generalisation gap |
|---|---|---|---|---|
| 99.0% | 0.126 | 0.9901 / 39.1% | **0.9878 / 40.2%** | **-0.23 pts** |
| 99.5% | 0.076 | 0.9954 / 23.4% | **0.9947 / 21.8%** | **-0.07 pts** |

**The threshold transfers.** The gap is 0.23 points at the 99% target and 0.07 at 99.5% — well
inside anything that would change a decision, and the val-chosen threshold happens to drop *more*
empties on test (40.2%) than the test-chosen one did (35.6%). Quote the honest numbers:
**98.8% of ships kept while skipping 40.2% of empty tiles**, against the classic filter's 55.5%
at a comparable 41.9% — **+43.3 points of safety at matched compute.**
