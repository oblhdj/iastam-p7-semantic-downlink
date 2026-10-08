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
   coordinates and fused. Measured recall **0.745** at **1.56×** the compute of regular tiling
   ([report 16](reports/16-swath-policies.md)). We use it **selectively**: a seam-cut ship is
   usually still detected, so blanket SAHI is a poor trade ([report 13](reports/13-tile-seams-and-sahi.md)).
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
| **Data reduction vs a bent pipe** | **557×** | 60,124 MB offered → 108.0 MB sent, per day |
| **Ships delivered** | **0.685** ⚠ *at an assumed 15% cloud* | band **0.40–0.82** across 0–50% cloud — never quote it bare |
| Fraction of what the satellite still knew | **100%** | at nominal load the downlink is *not* the bottleneck — the detector is |
| vs FIFO under congestion | **2.26×** | at 160k tiles/day, where the link saturates |
| vs a *fair* Phi-sat-2 style baseline | **+6.3 points** for ~7× the bytes | our own earlier "+16.7" used an unfair baseline — see [report 06](reports/06-byte-budget.md) |
| Greedy vs the exact optimum | **0.251%** | exact DP, verified against brute force |
| Detector | **mAP50 0.804** | 11 ms/tile GPU; recall 0.739 small / 0.958 medium / 0.994 large *at conf ≥ 0.05* |
| **SAHI + fusion (B2)** | **0.745** recall @ **1.56×** compute | window 768, 20% overlap; selective, not blanket — [report 16](reports/16-swath-policies.md) |

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

B1 is YOLO on the original image; **B2 is SAHI + YOLO + fusion** (`sat7.perception`,
`sat7.b2_sahi_fusion`). The rest of the chain (LoD + scheduler) is B3. See
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
| `code/sat7/` | the package: perception (YOLO / SAHI), encoder, scheduler, exact optimum, orbit, pre-filter, dedup |
| `code/scripts/` | one script per work package, each with a docstring saying *why* it exists |
| `code/tests/` | 137 tests (2 skip, need torch) — 14 fault-injection, exact-solver checks, and the gate's claims pinned |
| `code/results/` | generated artefacts — CSV, PNG, JSON, logs. Machine-written, not prose |

## Running it

Two environments, because torch and the rest disagree about Python versions:

```bash
cd code
.venv/Scripts/python.exe    -m pytest tests -q        # 137 pass, 2 skip (need torch)
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
  provably does nothing; blanket SAHI is not worth its compute. All four are in the reports.
- **When a later check contradicted an earlier claim, the later check won.** We had adopted a
  coastal JPEG setting as "no recall cost"; re-running the *detector* on recompressed tiles showed
  it costs 4.0 points on small ships ([report 14](reports/14-coastal-recompression.md)). The
  original measurement could not have seen it, and we say so rather than quietly restating it.
- **We corrected our own claims rather than defending them.** The Phi-sat-2 baseline was unfair by
  construction, so our advantage is +6.3 points, not +16.7. A dark-vessel rule did the opposite of
  what we claimed on 13.6% of ships until [report 10](reports/10-failure-modes.md) caught it.
- **Assumptions are swept, not assumed.** [Report 11](reports/11-sensitivity.md) sweeps every
  invented constant; conclusions hold in 69 of 70 settings, and the one failure is named.
- **The thing we are least sure of is the loudest.** 52% of all reported ship loss traces to
  cloud, which rests on **21 real tiles**. That is why recall is quoted as a band.
- **Tooling is disclosed.** Development was assisted by Claude (Anthropic). Every experiment,
  number and conclusion in this repository was specified, run and reviewed by the team.

## Limitations, stated up front

Timings are a laptop RTX 5060, not flight hardware — "runs onboard" is **not** demonstrated; it is
a **TARGET** mapped onto flown Myriad-class hardware ([report 22](reports/22-onboard-feasibility.md)).
The Airbus set is 0.4% cloud, so the cloud assumption cannot be settled from this data. Tiles are
resampled i.i.d. rather than as orbital strips. The headline campaign models a single ground station.
SAHI needs a large scene; Airbus tiles are already 768×768, so B2 is measured on **synthetic swaths
stitched from tiles**. Window is **768**, not the paper's 512 — the ranking reverses at 512
([report 16](reports/16-swath-policies.md)).
The numbered reports under [`reports/`](reports/) and [`paper/PHASE3_RESULTS.md`](paper/PHASE3_RESULTS.md)
are the current and authoritative account of this project.

## Licence and data

Code and reports: [MIT](LICENSE).

The imagery is **not** ours to relicense. It comes from the
[Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection) and remains
subject to that competition's terms. None of it is redistributed here;
`code/scripts/wp0_build_dataset.py` rebuilds the leakage-free split from the original archive,
which you must obtain yourself.
