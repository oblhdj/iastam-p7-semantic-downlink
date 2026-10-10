# Transmitting Information Rather Than Raw Data

**IASTAM 6.0 — Track 4, Problem 7.** An onboard pipeline that lets a small satellite downlink
*what it learned about ships* instead of the pictures it took.

A satellite images far more ocean than it can ever transmit. The usual answer is to compress the
pictures harder. Ours is to stop sending pictures: detect ships onboard (**YOLO**, then **SAHI**
with global-coordinate fusion so ships that straddle a tile seam are recovered), then spend the
scarce downlink on **information about them**, graded by how sure the detector is and scheduled by
how much each item is worth per byte.

That is the paper's B0→B4 chain. Three pieces of the onboard stack:

1. **SAHI + fusion (paper B2)** — sliced inference over a large scene, boxes mapped back to global
   coordinates and fused. On the same 150 scenes, one detector call on the whole scene (paper B1)
   finds **0.339** of the ships and SAHI + fusion finds **0.717**, at **1.56×** the detector calls
   of a plain tiling ([`wp26_b0_b4.json`](code/results/wp26_b0_b4.json)). We use it **selectively**:
   a seam-cut ship is usually still detected, so blanket SAHI is a poor trade
   ([report 13](reports/13-tile-seams-and-sahi.md)) — and on these scenes it does not beat plain
   tiling at all (see [Limitations](#limitations-stated-up-front)).
2. **Confidence-aware level of detail** — a confident detection costs 40 bytes of metadata; an
   uncertain one earns an image chip; a coastal tile the detector may have misread is sent whole.
3. **A value-aware multi-pass scheduler** — the downlink is a knapsack that refills every orbit,
   so items are chosen by value per byte and truncated progressively when a pass runs short.

Measured on **53,195 real Airbus tiles**, with a detector we trained, over **real orbit passes**
computed from an SGP4 propagator for a ground station at Sfax.

---

## Headline results

| | measured | how to read it |
|---|---|---|
| **Data reduction vs a bent pipe** | **341×** — *coastal tiles measured; thumbnails and crops still modeled* | 60,124 MB offered → 176.3 MB sent per day, mean of three days. Each coastal tile is charged its own measured JPEG size ([`wp28`](code/results/wp28_coast_tile_model.json)); the earlier all-modeled estimate was 557× ([report 05](reports/05-scheduler-on-real-detections.md)) |
| **Ships delivered** | **0.685** ⚠ *at an assumed 15% cloud* | band **0.40–0.82** across 0–50% cloud — never quote it bare |
| Fraction of what the satellite still knew | **100%** — *for a single day* | at nominal load the downlink is *not* the bottleneck — the detector is. That is one day's traffic given 36 h of passes. The link share sustains 134.5 MB per 24 h and a day offers 130% of that; over six consecutive days recall still holds (0.677–0.688) because the scheduler truncates low-value products, and FIFO does not hold ([`wp29`](code/results/wp29_validation.json)). Recall counts a progressively truncated item as delivered once it clears `min_fraction` (10% of its bytes) |
| vs FIFO under congestion | **3.59×** | at 160k tiles/day; the link saturates from about 80k (1.88× there) |
| vs a *fair* Phi-sat-2 style baseline | **+6.3 points** for ~11× the bytes | our own earlier "+16.7" used an unfair baseline — see [report 06](reports/06-byte-budget.md) |
| Greedy vs the exact optimum | **0.884%** | exact DP, verified against brute force |
| Detector | **mAP50 0.804** | 11 ms/tile GPU; recall 0.739 small / 0.958 medium / 0.994 large *at conf ≥ 0.05* |
| **Whole scene (B1) → SAHI + fusion (B2)** | recall **0.339 → 0.717** @ **1.56×** compute | the same 150 scenes of 4×4 tiles, 1,164 ships; precision 0.68 → 0.70; window 768, 20% overlap — [`wp26_b0_b4.json`](code/results/wp26_b0_b4.json) |
| Detector on native 768 px tiles — *reference, not the paper's B1* | recall **0.766**, precision 0.754 | one call per tile on all 5,320 test tiles (8,173 ships) at the 0.25 cut — [`wp24_detection_eval.json`](code/results/wp24_detection_eval.json) |

Every figure above is reproducible from this repository. The `.csv` files under
`code/results/` are authoritative; the reports narrate them.

> **Phase-3 status & paper→evidence map.** The accepted Phase-2 paper reported no numbers by
> design; **Phase 3 is the measured counterpart and is complete** — B0→B4, the five metrics,
> the energy model, the optional relay, and the ablations all ran. Authoritative write-ups:
> [`paper/PHASE3_RESULTS.md`](paper/PHASE3_RESULTS.md) and [`demo/PAPER_COVERAGE.md`](demo/PAPER_COVERAGE.md)
> (paper claim → evidence). SAHI + global-coordinate fusion, the per-stage energy model and the
> inter-satellite relay are **implemented and measured**. `docs/ARCHITECTURE.md` and
> `docs/STATUS.md` predate Phase 3 and may still mark built items as "missing" — trust the two
> files named here.

## How the pipeline fits together

```
   scene ──► gate ──► YOLO ──► SAHI + fusion ──► level-of-detail ──► scheduler ──► ground
             47k CNN   768px    global coords      encoder            value-greedy
             0.15 ms   11 ms    1.56× vs tiling    40 B / chip / tile
```

B1 is YOLO on the original image — one call on the whole scene, no slicing; **B2 is SAHI + YOLO +
fusion** (`sat7.perception`, `sat7.b2_sahi_fusion`). Both are measured on the same scenes by
`scripts/wp26_b0_b4.py`. The rest of the chain (LoD + scheduler) is B3. See
[report 16](reports/16-swath-policies.md) and [`demo/PAPER_COVERAGE.md`](demo/PAPER_COVERAGE.md).

`scripts/wp11_integration_demo.py` runs this whole chain on real JPEGs and reproduces the
catalogue-driven simulation to four decimal places — see
[report 12](reports/12-end-to-end-integration.md).

## Repository layout

| path | what is in it |
|---|---|
| **[`paper/PHASE3_RESULTS.md`](paper/PHASE3_RESULTS.md)** | **Start here for judges.** The measured Phase-3 write-up, section-for-section against the paper |
| **[`demo/PAPER_COVERAGE.md`](demo/PAPER_COVERAGE.md)** | paper claim → Phase-3 evidence, line by line |
| **[`reports/`](reports/)** | 23 narrative write-ups, one per work package |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | pre-Phase-3 build audit — **superseded for status** by `PHASE3_RESULTS.md` |
| [`docs/STATUS.md`](docs/STATUS.md) | pre-Phase-3 state — **superseded for status** by `PHASE3_RESULTS.md` |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | every term and symbol used in the reports |
| [`paper/`](paper/) | Phase-3 measured results, slide deck (`slides/`) and the 2-min video script |
| [`demo/quickstart/`](demo/quickstart/) | the chain live on a laptop CPU in ~2 s: no GPU, no torch, no dataset (synthetic stand-in tiles) |
| `code/sat7/` | the package: perception (YOLO / SAHI), encoder, scheduler, exact optimum, orbit, pre-filter, dedup |
| `code/scripts/` | one script per work package, each with a docstring saying *why* it exists |
| `code/tests/` | 253 tests (5 skip in `.venv`: 2 need torch, 3 need onnxruntime) — 14 fault-injection, exact-solver checks, and the gate's claims pinned |
| `code/results/` | generated artefacts — CSV, PNG, JSON, logs. Machine-written, not prose |

## Running it

**Quickest — no GPU, no torch, no dataset (~2 s on a laptop CPU):**

```bash
python -m pip install -r demo/quickstart/requirements.txt
python demo/quickstart/run_demo.py
```

This runs the real `sat7` chain (pre-filter → YOLO ONNX → SAHI + fusion → P0–P3 packet →
scheduler) on 9 synthetic stand-in tiles and ends on the measured headline read from
`code/results/`. See [`demo/quickstart/README.md`](demo/quickstart/README.md) for why the tiles
are synthetic and how to run it on the real ones.

**The full pipeline** needs two environments, because torch and the rest disagree about Python versions:

```bash
cd code
.venv/Scripts/python.exe    -m pytest tests -q        # 248 pass, 5 skip (need torch / onnxruntime)
.venv312/Scripts/python.exe scripts/wp11_integration_demo.py --tiles 400
```

`.venv312` (Python 3.12) has torch + CUDA and runs anything touching the detector.
`.venv` (Python 3.14) runs everything else, which is most of the analysis.
The dataset and weights are not in git; `code/scripts/wp0_build_dataset.py` rebuilds the
leakage-free split from the Airbus archive.

## How we tried to keep ourselves honest

This mattered more than any single result, so it is stated plainly:

- **Every figure is labelled REAL / SIM / LIT / TARGET / ASSUMPTION.** No simulated number is quoted as measured.
- **We report our own negative results.** The classic pre-filter failed as a gate (0.645 recall);
  INT8 quantisation is 8.5× *slower* on CPU and costs 2.2 points on small ships; `aging_per_hour`
  provably does nothing; blanket SAHI is not worth its compute, and on our stitched scenes it does
  not beat plain tiling (0.717 vs 0.730 — see Limitations). All of these are in the reports.
- **When a later check contradicted an earlier claim, the later check won.** We had adopted a
  coastal JPEG setting as "no recall cost"; re-running the *detector* on recompressed tiles showed
  it costs 4.0 points on small ships ([report 14](reports/14-coastal-recompression.md)). The
  original measurement could not have seen it, and we say so rather than quietly restating it.
- **We corrected our own claims rather than defending them.** The Phi-sat-2 baseline was unfair by
  construction, so our advantage is +6.3 points, not +16.7. A dark-vessel rule did the opposite of
  what we claimed on 13.6% of ships until [report 10](reports/10-failure-modes.md) caught it.
- **Assumptions are swept, not assumed.** [Report 11](reports/11-sensitivity.md) sweeps every
  invented constant over 80 settings. The scheduler beats FIFO in all 80; our lead over the fair
  baseline holds in 74, and the six failures are named (see Limitations).
- **The thing we are least sure of is the loudest.** 52% of all reported ship loss traces to
  cloud, which rests on **21 real tiles**. That is why recall is quoted as a band.
- **Tooling is disclosed.** Development was assisted by Claude (Anthropic). Every experiment,
  number and conclusion in this repository was specified, run and reviewed by the team.

## Limitations, stated up front

Timings are a laptop RTX 5060, not flight hardware — "runs onboard" is **not** demonstrated; it is
a **TARGET** mapped onto flown Myriad-class hardware ([report 22](reports/22-onboard-feasibility.md)).
The Airbus set is 0.4% cloud, so the cloud assumption cannot be settled from this data. Tiles are
resampled i.i.d. rather than as orbital strips. The headline campaign models a single ground station.
SAHI needs a large scene; Airbus tiles are already 768×768, so B1 and B2 are measured on **synthetic
scenes stitched from tiles**. Window is **768**, not the paper's 512 — the ranking reverses at 512
([report 16](reports/16-swath-policies.md)).

**A negative result, reported as one.** On those scenes SAHI + fusion (recall **0.717**) does not
beat a plain tile-by-tile pass with no overlap (**0.730**), and it costs 1.56× the detector calls
([`wp26_b0_b4.json`](code/results/wp26_b0_b4.json), same 150 scenes, 1,164 ships). The reason is the
seams: the scenes are stitched from independent tiles, so no ship ever crosses a seam and the overlap
has nothing to recover — it only re-detects the same ships in windows that straddle two unrelated
tiles. What B2 is measured against is the whole-scene call (B1, 0.339), where slicing is worth 38
points; whether overlap beats plain tiling needs real large scenes, which we do not have.

**A second negative result: our lead over the fair baseline does not survive every setting.** The
sensitivity sweep ([`wp10_sensitivity.csv`](code/results/wp10_sensitivity.csv), 80 settings of the
invented constants) keeps "value-greedy ≥ FIFO" in all 80, but "ours ≥ the fair Phi-sat-2-style
baseline" in only **74 of 80**. All six failures are at 160,000 tiles/day, where the lead is just
**1.3 points** before anything is changed (it is 6.3 points at 40,000 tiles/day, where nothing
fails): `thumb_value` 0.05 (−0.5 points) and 0.10 (−1.9), `conf_high` 0.90 (−0.02) and 0.99 (−0.4), `conf_low` 0.05 (−1.6) and 0.10 (−0.8). Each of those settings makes the encoder spend more bytes — on thumbnails, on crops
for confident ships, or on low-confidence detections — which a saturated link cannot afford.

**What a check of our own numbers found.** `code/scripts/wp29_validation.py` runs 37 checks on units,
data sizes, energy, latency and fairness against the committed results. None fails; five gaps are
stated rather than fixed:
(1) the headline day is simulated alone, with 36 h of passes — it offers 130% of what the 25% link
share sustains per 24 h (134.5 MB), so consecutive days force the scheduler to truncate;
(2) a truncated progressive product still counts as delivering its ships, which is an assumption,
not a measurement, and is what keeps recall flat under load;
(3) "ES_total ≈ 98%" compares us with a bent pipe that sends every raw byte, which would need 92.6
days of contact per day — a bent pipe limited to this link spends at most 30.8 kJ/day, less than our
54 kJ/day, and delivers 0.2% of the ships;
(4) the B4 row's relay latency is a window estimate (no volume, no capacity);
(5) thumbnails and ship crops, 24% of the headline's bytes, are still model sizes (next paragraph).
**No power was measured in this project: every energy figure is an estimate from assumed powers.**

The headline's byte count mixes two kinds of number: coastal tiles are **measured** (each tile's own
JPEG), thumbnails and ship crops are still **modeled** sizes. B1/B2 (150 scenes) and B3/B4 (a
simulated 40,000-tile day) run on different inputs, so B2's detection recall and B3's delivered
recall are not steps of one curve.

The numbered reports under [`reports/`](reports/) and [`paper/PHASE3_RESULTS.md`](paper/PHASE3_RESULTS.md)
are the current and authoritative account of this project.

## Licence and data

Code and reports: [MIT](LICENSE).

The imagery is **not** ours to relicense. It comes from the
[Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection) and remains
subject to that competition's terms. None of it is redistributed here;
`code/scripts/wp0_build_dataset.py` rebuilds the leakage-free split from the original archive,
which you must obtain yourself.
