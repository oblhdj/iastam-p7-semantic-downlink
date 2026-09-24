# WP11 — end-to-end integration: real image bytes through the real chain (24 Sept 2026)

Closes ARCHITECTURE.md §E item 1, the largest "never started" gap: *"nothing runs tile →
prefilter → detector → LoD → scheduler → ground"*. Script: `scripts/wp11_integration_demo.py`.
Data: `wp11_integration.json`, `wp11_per_tile.csv`, `wp11_funnel.png`, log `wp11_run.log`.
All figures REAL — the **full 5,320-tile held-out test split**, actual JPEGs, actual models,
on the PC (RTX 5060).

Every stage had been measured alone, and WP6–WP10 **replay a stored catalogue**
(`wp6_tiles.csv` / `wp6_ships.csv`) instead of running the models. Nothing had ever checked
that the replay matched the thing it replays. This does.

## 1. The chain runs at FULL SCALE, and reproduces WP6 to four decimals

Re-run 24 Sept ~19:35 over the **entire test split (5,320 tiles, 8,173 ships)**, after adopting
`conf_high` 0.670. The day-scale result is identical to the catalogue pipeline it was checked
against — from real image bytes, not from a stored CSV:

| quantity | WP11 (live, from JPEGs) | WP6 (catalogue replay) |
|---|---|---|
| ships in the simulated day | **21,137** | 21,137 |
| MB offered | **108.5** | 108.5 |
| ship recall | **0.6798** | 0.6799 |
| dark recall | 0.6793 | 0.6793 |
| median latency | 4.49 h | 4.50 h |

Tile-by-tile cross-check over all 5,320:

| field | agreement | differing |
|---|---|---|
| `context`, `ctx3`, `n_candidates`, `n_ships` | **1.0000** | 0 |
| `n_fp_0.25` | 0.9998 | **1 tile** |
| ship pairing (8,173 ships) | exact | max \|size diff\| 5.7e-14 |
| ship confidence | — | max \|diff\| **2.5e-01**, 6,858 differ bitwise |
| **decisions changed** at `det_thr` / `conf_low` / `conf_high` | — | **1 / 1 / 0** |

⚠ **Correction to the 400-tile run, which reported "0 decisions changed".** At full scale it is
**1 ship in 8,173 (0.012%)**, not zero. The max confidence difference is 0.25, far too large for
float drift: on one tile two predictions sat at nearly equal IoU against the same ground-truth
ship, and last-digit FP32 differences flipped which one the greedy matcher consumed. The ship is
still found either way — a different box is credited with it. That is the honest number, and it
is small enough to leave the conclusion intact: **the catalogue WP6–WP10 replay is faithful.**

> Method note: the first version of this cross-check reported a **30% disagreement** on false
> alarms. That was entirely self-inflicted — it reimplemented the matching rule as "greedy,
> exclusive, IoU ≥ 0.5" while the catalogue uses "overlaps any GT box at IoU ≥ 0.3". The script
> now **imports** `flag_predictions` / `count_false_alarms` from `wp6_build_catalogue` instead
> of restating them. A second artefact — pairing ships by `(image, size_px)` — cross-produced
> whenever a tile held two ships of equal length, reporting 56 "matched ships" out of 49 and
> inventing confidence differences of 0.38. Both were bugs in the test, not the pipeline.

## 2. ⚠ The finding: 52% of all reported ship loss is **12 real ships**, stamped out 287 times

The orbit mix draws a *fixed share* of the simulated day from each context regardless of how
many real tiles back that context. Measured against the full 5,320-tile catalogue:

| context | drawn into WP6's day | real tiles behind it | reuse | ships/tile | ships in the day |
|---|---|---|---|---|---|
| empty | 24,225 | 766 | ×32 | 0.000 | 0 |
| ships | 7,690 | 3,118 | ×2 | 1.830 | 14,075 |
| coast | 2,060 | 1,415 | ×1 | 1.734 | 3,573 |
| **cloud** | **6,025** | **21** | **×287** | 0.571 | **3,443** |

WP6's day holds 21,137 ships at ceiling 0.685, so **6,657 ships are reported lost — and 3,443 of
them (52%) are cloud-gated.** Those 3,443 are **12 real ships on 21 real tiles**, resampled 287
times each.

This matters because of what it collides with: **WP10 already found cloud fraction is the single
dominant parameter** (recall range **0.42**, from 0.82 at 0% cloud to 0.40 at 50% — five times
the next-largest effect). So the most influential quantity in the project is backed by the
thinnest evidence in the dataset. The Airbus set is 0.4% cloud; it cannot settle this, and WP10
already flagged that a climatology is needed [LIT].

**How to say it in the paper.** Not "we deliver 68.5% of ships and the rest is the detector's
fault" — over half that shortfall is one cloud assumption resting on 12 ships. Say: *"68.5% at
an assumed 15% cloud fraction; the cloud term dominates and rests on 21 real tiles, so quote it
as a band, not a point."* That is the honest framing and a reviewer will respect it more than
the point estimate.

The script now prints the backing pool for every context and **warns automatically** when one
is under 30 real tiles, so this cannot silently recur.

## 3. ⚠ The classic pre-filter costs more than the detector it is supposed to gate

Measured in the same run, single-threaded, each stage on its own timer:

| stage | runs on | cost per tile |
|---|---|---|
| classic pre-filter | every tile, CPU | **56.3 ms** |
| learned gate (47k params) | every tile, GPU | **0.154 ms** |
| detector (YOLO, 768 px) | surviving tiles, GPU | **11.1 ms** |

The pre-filter is **5.1× more expensive than the detector** on GPU, and still 1.5× more
expensive than the detector's honest CPU figure (38.6 ms, ONNX FP32 — WP1). **A gate that costs
more than the thing it gates can never save compute, at any accuracy.** WP3 had already
rejected the classic filter on *accuracy* (0.645 recall while dropping 17.7% of empties); this
retires it on *cost* as well, independently. The learned gate costs **0.154 ms** at full scale — **365× cheaper**
than the classic stage and 1.4% of the detector. Swapping one for the other frees **56 ms/tile**,
which is the compute budget that makes SAHI affordable — see WP12.

(The catalogue records 88.8 ms/tile for the pre-filter; that run used 8 worker processes
contending for cores. 56.3 ms single-threaded is the cleaner figure.)

## 4. What the demo does *not* establish
* ~~Absolute recall is not comparable with WP6~~ — **resolved by running the full split**: it
  now matches to 1e-4. The earlier 400-tile run gave 0.829 because its cloud pool was a single
  tile holding zero ships. The thin-pool warning remains in the script, and still fires for
  `cloud` even at full scale (21 real tiles, x287 reuse) — that one is the dataset's limit, not
  the demo's, and §2 is why it matters.
* ⚠ A first attempt at the full split **stalled**: the script held every decoded 768x768 tile in
  memory (9.4 GB at 5,320 tiles). It now keeps only the 128 px gate copies (261 MB).
* Still a laptop RTX 5060, not flight hardware — §E item 3 stands untouched.
* The chain runs tile-at-a-time from disk; there is no real-time or thermal budget.
* `ctx3` splits open sea into "ships"/"empty" using **ground-truth** `n_ships`, which an onboard
  system would not have. It is cosmetic for the encoder (a tile with no GT ships has no ship
  items to emit either way) but it **does** set the sampling pools, so it is not free.

## 5. Recommended next step
Re-run WP6 with the cloud fraction swept as a **band** rather than fixed at 15%, and quote every
recall number with its cloud assumption attached — WP10 §2 asked for this and §2 above shows why
it is now the top honesty risk in the paper.
