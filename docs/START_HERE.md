# IASTAM 6.0 — Problem 7 — START HERE

> **⚠ STATUS UPDATE (7 Oct 2026).** This file was written mid-Phase-3 and parts of §3/§5 below are
> now out of date: **SAHI + global-coordinate fusion, the per-stage energy model, and the optional
> inter-satellite relay are all implemented and measured**, and the B0→B4 campaign runs end to end.
> For current status trust **[`../paper/PHASE3_RESULTS.md`](../paper/PHASE3_RESULTS.md)** and
> **[`../demo/PAPER_COVERAGE.md`](../demo/PAPER_COVERAGE.md)**. The §7 *superseded-numbers* list and
> the project rules (§8) remain correct and useful.

**Read this file alone and you can help with the project.** It is self-contained: no other file
is needed. Written 2 Oct 2026; status banner updated 7 Oct 2026.

Everything is in the repo `github.com/oblhdj/iastam-p7-semantic-downlink`, locally at
`Desktop/IASTAM_Problem7`. Deeper detail lives in `docs/STATUS.md`, `docs/ARCHITECTURE.md` and
the 15 write-ups in `reports/` — but you do not need them to be useful.

---

## 1. The challenge

IASTAM 6.0, **Track 4, Problem 7: "Transmitting Information Rather Than Raw Data."**
Team of four at ENIS, Sfax, Tunisia: Ahmed Makni, Mohamed Achraf Ben Saad, Mohamed Ketata,
Mariem Megdiche.

* **Interim paper: submitted and ACCEPTED.**
* **Phase 3 runs 1–14 October 2026** — this is the measurement/validation phase. We are in it now.

## 2. The idea in one page

A satellite images far more ocean than it can transmit. The usual answer is to compress the
pictures harder. Ours is to **stop sending pictures**: detect ships onboard, then spend the scarce
downlink on *information about them*.

Two contributions:

1. **Confidence-aware level of detail.** A confident detection costs ~40 bytes of metadata; an
   uncertain one earns an image chip; a coastal tile the detector may have misread is sent whole.
2. **A value-aware multi-pass scheduler.** The downlink is a knapsack that refills every orbit, so
   items are ordered by value-per-byte and truncated progressively when a pass runs short.

Measured on **53,195 real Airbus Ship Detection tiles** (768×768), with a detector we trained
ourselves, over **real orbit passes** from an SGP4 propagator for a ground station at Sfax.

## 3. What the accepted paper commits us to — and the gap

The accepted paper is **"Energy-Aware Semantic Downlink for Maritime Object Detection with
Optional Inter-Satellite Relay."** Crucially it reports **zero measured numbers by design**
("No numerical performance value is reported without a corresponding experimental measurement"),
so nothing wrong got submitted. It is a methodology + roadmap paper. **Phase 3 is the contract.**

It promises a **B0→B4 progression**:

| config | onboard | communication |
|---|---|---|
| B0 | none | full image (raw reference) |
| B1 | YOLO on original image | detection output |
| B2 | **SAHI + YOLO** | detection output |
| B3 | SAHI + YOLO + semantic policy | metadata / ROI |
| B4 | + adaptive comms | direct **or optional inter-satellite relay** |

and priority levels **P0** discard · **P1** metadata · **P2** metadata+ROI · **P3** metadata+ROI+context.

### ✅ What the paper promised — now built and measured (updated 7 Oct 2026)

| promised | status |
|---|---|
| **SAHI sliced inference** (window + 20% overlap) | **implemented & measured** — `sat7/b2_sahi_fusion.py`, `sat7/perception.py`; B2 ran (recall 0.745 @ 1.56× compute). Window **768**, not 512 — the result reverses at 512 (report 16); see the trade in §7 |
| **Global-coordinate transform + detection fusion** | **implemented & measured** — `sat7/b2_sahi_fusion.fuse`; validated to 0.0000 recall error vs per-tile (report 16) |
| **Per-stage energy model** `E_proc = Σ P_k·T_k` | **implemented & measured** — `wp17_energy_model.py` / `sat7/energy.py`; ES_proc 57.6%, ES_total ≈ 98%, proc:comm 8.9:1 (report 17) |
| **Optional inter-satellite relay**, `J_direct` vs `J_relay` | **implemented & measured** — `sat7/relay.py`; worst-case latency 11.6→6.2 h (reports 19–21) |
| Joint objective `min αE + βD + γT` | the online scheduler uses value-per-byte (near-optimal, report 08); the full program is also solved directly in `sat7/joint.py` (report 23) |

And the reverse — **built, but absent from the paper**: dark-vessel/AIS prioritisation, the
multi-pass scheduler itself, queue-aware adaptive detail, the optimality bounds, the leakage-free
dataset split. These are among our strongest results and currently have no home in the accepted
narrative. They fit naturally under the paper's §IV-B "adaptive downlink policy".

## 4. What is already built and measured

86 tests pass. Headline numbers (all reproducible from `code/results/*.csv`, which are
authoritative over any prose):

| | measured | how to read it |
|---|---|---|
| Data reduction vs bent pipe | **557×** | 60,124 MB offered → 108.0 MB sent per day |
| Ships delivered | **0.685** | ⚠ *at an assumed 15% cloud*; band **0.40–0.82** over 0–50% cloud — **never quote bare** |
| Of what the satellite still knew | **100%** | at nominal load the downlink is **not** the bottleneck — the detector is |
| vs FIFO under congestion | **2.26×** | at 160k tiles/day, where the link saturates |
| vs a *fair* Phi-sat-2 baseline | **+6.3 points** for ~7× bytes | our earlier "+16.7" used an unfair baseline |
| Greedy vs exact optimum | **0.251%** | exact DP, verified against brute force |
| Detector | **mAP50 0.804** | 11 ms/tile GPU; recall 0.739 small / 0.958 med / 0.994 large (conf ≥ 0.05) |

The 15 work packages, one line each:

| # | finding |
|---|---|
| 01 | 19.1% of Airbus tiles have a near-duplicate twin; a naive split leaks 1,617 groups, ours leaks 0 |
| 02 | ONNX FP32 is lossless and 1.21× faster on CPU. **INT8 is 8.5× slower and costs 2.2 pts on small ships — rejected** |
| 03 | The detector is **underconfident** (says 0.352, right 0.408). Isotonic cuts ECE 8.9×. **`conf_high` should be 0.670, not 0.9** |
| 04 | Classic CV gate fails (0.645 recall). A **47k-param CNN keeps 98.8% of ships while skipping 40.2% of empty tiles**; trains in 2m21s |
| 05 | Replacing the synthetic workload with real detections moved recall 97.1% → **0.685**; the downlink stopped being the bottleneck |
| 06 | **86% of our "semantic downlink" was not semantic** (coastal tiles + blanket thumbnails). Our baseline had been unfair |
| 07 | No fixed coastal policy wins in more than one regime; letting the buffer set detail is within 1.1 pts of best at every load |
| 08 | Exact DP: greedy is **0.251% from optimal**. The plan had ticked this as done with no solver in the repo |
| 09 | Perfect foresight over a whole day is worth **≤0.2%** → **do not build arrival prediction** |
| 10 | FMEA became 14 fault-injection tests and **failed in places**: a dark-vessel rule did the opposite of its claim on 13.6% of ships |
| 11 | Every invented constant swept; conclusions hold in 69/70 settings; **cloud fraction dominates everything** |
| 12 | Real JPEGs through the real chain reproduce the simulation to 4 decimals; 52% of reported ship loss rests on 21 real tiles |
| 13 | On a real 10k swath a 380 px ship is cut 74.5% of the time, **but a cut ship is usually still detected** → blanket SAHI is a poor trade |
| 14 | The q40 coastal setting adopted as "no recall cost" **actually costs 4.0 points**, all on small ships; q30 rejected |
| 15 | Moving thumbnail gating onto the learned gate returns **3.8% of the whole downlink at zero measured recall cost** |

## 5. What Phase 3 produced (all delivered — see `../paper/PHASE3_RESULTS.md`)

Ranked by "the paper promised it":

1. ✅ **SAHI + global-coordinate fusion (B2)** — `sat7/b2_sahi_fusion.py` + `sat7/perception.py`;
   B2 ran, recall 0.745 @ 1.56× compute (reports 13, 16). Window 768, the measured trade in §7.
2. ✅ **Per-stage energy model** — `wp17_energy_model.py` / `sat7/energy.py`: `E_proc = Σ P_k·T_k`,
   `E_comm = P_tx·D_tx/R_tx`, `ES = 1 − E_prop/E_base`. ES_proc 57.6%, ES_total ≈ 98% (report 17).
3. ✅ **Optional inter-satellite relay (B4)** — `sat7/relay.py`: `E_relay = E_ISL + E_GS`,
   `min(J_direct, J_relay)` with `J = λ_E·E + λ_T·T`, real ISL windows. Worst latency 11.6→6.2 h
   (reports 19–21). Energy half is TARGET pending real P_isl/R_isl.
4. ✅ **B0–B4 campaign** end to end — `wp18_campaign_runner.py`, sanity-gated (report 21).
5. 🟡 **Ablations**: without SAHI / adaptive / ROI / relay are reported (reports 06, 07, 16, 21);
   without-overlap / without-fusion are now isolable as flags (`sat7/perception.py`, report 23) but
   not yet run as their own measured rows.
6. ✅ **Unreported work folded in** — scheduler, queue-aware LoD, optimality bounds, leakage-free
   split — under §IV-B in `../paper/PHASE3_RESULTS.md`.

Still open (stated as limitations, not claims): edge-hardware wall-clock (every figure is a laptop
RTX 5060, so "runs onboard" is a **TARGET**, report 22); multi-ground-station (the capability exists,
`sat7.orbit.find_passes_multi`, but the headline campaign is single-station); burst/correlated
arrivals (tiles resampled i.i.d.); CI.

## 6. How to run things

Two Python environments, on purpose:

```bash
cd Desktop/IASTAM_Problem7/code

# everything non-torch: simulation, sweeps, analysis, tests
.venv/Scripts/python.exe -m pytest tests -q          # expect 86 passed
.venv/Scripts/python.exe scripts/<name>.py

# anything needing torch / ultralytics / onnx (CUDA works here)
.venv312/Scripts/python.exe -W ignore scripts/<name>.py
```

* `.venv` is Python 3.14 and has **no working torch** — that is fine, most scripts don't need it.
* `.venv312` is Python 3.12 with torch 2.11+cu128, ultralytics, onnx. It is **not** corrupted
  (an old note said so). It simply had no `pip`; fixed with `python -m ensurepip --default-pip`.
* pip needs a long timeout for big wheels: `--timeout 120 --retries 5`.
* **Nothing needs Kaggle.** The one job everyone assumed was heavy (the learned gate) trains in
  **2m21s** locally. The dataset is 7.7 GB — do not upload it anywhere.

## 7. Traps — things that have already bitten us

* **Never quote ship recall without the cloud assumption.** 0.685 is at an assumed 15% cloud; the
  band is 0.40–0.82 across 0–50%. Cloud fraction moves the result more than every other constant
  combined, and the Airbus set cannot settle it (it is 0.4% cloud).
* **SAHI vs our own evidence.** The paper is built on SAHI, but report 13 measured that although a
  380 px ship straddles a seam 74.5% of the time on a real 10k swath, **it is usually still
  detected** — so blanket SAHI buys little for 1.5× the compute. Phase 3 must face this honestly:
  argue selective/border-triggered slicing, or revise the claim. Do not quietly drop it.
* **SAHI needs large scenes; Airbus tiles are already 768×768 tiles.** You cannot slice a tile.
  Either stitch tiles into synthetic swaths (keeps the trained detector and all measurements) or
  bring in a large-scene dataset (xView / DOTA / VisDrone / HRSID) and retrain. **Decide this first
  — it gates most of Phase 3.**
* **INT8 is rejected, not pending.** 8.5× slower on CPU and −2.21 pts on small ships. The
  "1.5–3.3× faster" figure in the literature is TensorRT / static QDQ, not ONNX dynamic
  quantisation. **ONNX FP32 is the onboard build.**
* **Superseded numbers you may still find in old text:** "442×" and "511×" → **557×**;
  "97.1% ships" → **0.685**; "+16.7 points vs Phi-sat-2" → **+6.3**; "2.7× FIFO" → **2.26×**;
  "INT8 is lossless" → **false**; "coastal q40 costs no recall" → **costs 4.0 points**.
* Where a report has a **REFRESHED** banner, the banner is current and the table under it is not.
  **The `.csv` files always win.**

## 8. The rules this project works by

Keep these — they are why the paper was accepted with no numbers in it.

* Every figure is labelled **REAL** (measured on Airbus with our models) · **SIM** (orbit/link
  simulator over real detections) · **LIT** (published, cited) · **TARGET** (goal, not result) ·
  **ASSUMPTION** (invented — and all of these are swept in report 11).
* **Report negative results.** The classic pre-filter failing, INT8 being rejected, blanket SAHI
  being a poor trade, and our own unfair baseline are all in the write-ups.
* **Measure before adopting.** Several "obvious" improvements were killed by measurement, and one
  adopted setting (coastal q40) turned out to cost 4 points when finally tested properly.
* When a number looks too good, check the baseline first.

## 9. A sensible split for four people

* **A — Perception:** the SAHI/large-scene decision (§7), sliced inference, global-coordinate
  fusion, and the overlap/fusion ablations.
* **B — Energy:** the per-stage model, power measurements, `ES` ratio, and the `E` vs `T`
  trade-off curves.
* **C — Communication:** the inter-satellite relay, path choice, link windows, and the B4 runs.
* **D — Integration & paper:** the B0–B4 campaign, folding the unreported scheduler work into
  §IV-B, and keeping the figure labels honest.

Shared discipline: whoever changes a default **re-runs everything downstream** — that has caught
us out before.
