# 22 — Onboard feasibility: why "runs onboard" is a defensible TARGET, not a prototype

> **This is a demo, not a prototype.** Every timing in this project was measured on a **laptop
> RTX 5060 + its CPU** ([report 17](17-energy-model.md) §6, `docs/START_HERE.md` §5/§7). We have
> **not** run the pipeline on flight hardware. So the claim "this runs onboard a satellite" is a
> **TARGET**. This report says exactly how credible that target is, what makes it credible, and —
> honestly — what a real prototype would still have to measure. It introduces **no new result**;
> it reframes existing REAL/SIM numbers against published flight-hardware specs (LIT).

## 1. The claim and its label

| claim | label | why |
|---|---|---|
| The pipeline *executes* end to end on real data | **REAL** | [report 12](12-end-to-end-integration.md), `demo/` — measured on a laptop |
| The per-stage **energy ratios** carry to flight hardware | **TARGET**, but power-free (§4) | ES_proc is a time ratio, hardware-independent ([report 17](17-energy-model.md)) |
| The per-tile **wall-clock latency** on a flight VPU | **TARGET** | estimated in §3, never measured — a prototype must settle it |
| "It runs onboard" as a *flown* capability | **not claimed** | needs hardware-in-the-loop; §5 lists what is missing |

## 2. What we are actually asking the hardware to do

The onboard work per tile, with the compute volume and the laptop time we measured:

| stage | model size | compute / tile | laptop time (REAL) | source |
|---|---|---|---|---|
| classic pre-filter (replaced) | — (CV) | — | 56.3 ms CPU | [report 12](12-end-to-end-integration.md) |
| **learned gate** | **47k params** | negligible vs detector | 0.94 ms CPU / 0.05 ms GPU | [report 04](04-onboard-gate.md), [17](17-energy-model.md) |
| **detector (YOLOv8n @768)** | **3.15M params, ~12.5 GFLOPs/tile** [a] | the dominant cost | **38.6 ms CPU-ONNX** / 10.7 ms GPU | [report 02](02-detector-and-quantisation.md) |
| LoD encode + schedule | — | sub-ms | 0.5 ms (ASSUMPTION) | [report 17](17-energy-model.md) |

[a] YOLOv8n is **3.15M params / 8.7 GFLOPs at 640 px** (Ultralytics, official); scaling by
`(768/640)² ≈ 1.44` gives **~12.5 GFLOPs at our 768 px** (DERIVED). This is a *small* CNN by any
measure — it is the class of model built to run on edge accelerators.

## 3. The hardware this class of work already flies on

The directly relevant precedent is **our own baseline**: Φ-sat-2 already performs **onboard CNN
inference** on an Intel Movidius **Myriad 2** VPU (Ubotica CogniSAT-XE1 board, 12 vector cores);
Φ-sat-1 flew the same Myriad 2 for onboard cloud detection — the first AI inference on an EO
satellite (LIT, eoPortal). So "a small CNN runs onboard a smallsat" is not a hope; it is the
state of practice we are measured against.

| processor (flown / COTS for flight) | compute | power | label |
|---|---|---|---|
| Intel Movidius **Myriad 2** — Φ-sat-1/2 onboard AI | ~hundreds of GFLOPs/s DNN | ~1–2 W | LIT |
| Intel Movidius **Myriad X** | up to **4 TOPS**, **>1 TOPS** on DNN | **~1.5 W** | LIT |
| Unibap **SpaceCloud iX5-100** (CPU+GPU+FPGA+Myriad X) | heterogeneous | **10–30 W** | LIT |

**Order-of-magnitude latency on a flight VPU (TARGET).** ~12.5 GFLOPs/tile on a >1-TOPS DNN engine
is ~12 ms ideal; at a realistic 20–40% utilisation, **~30–150 ms/tile**. That is the *same order*
as our measured 38.6 ms laptop CPU-ONNX figure — not a measurement, an envelope. Only a hardware
run settles it (§5), and it depends on operator coverage and quantisation (see the ⚠ in §4).

**The throughput argument is far more robust than any single latency guess**, because it barely
depends on the VPU's exact speed. At 40,000 tiles/day (the operational load, ASSUMPTION per
[report 17](17-energy-model.md)):

| assumed detector latency/tile | relative to our laptop | compute time for 40k tiles | duty cycle of 24 h |
|---|---|---|---|
| 38.6 ms | 1× (our REAL ONNX CPU) | 0.43 h | **1.8%** |
| 100 ms | ~2.6× slower (TARGET) | 1.11 h | **4.6%** |
| 250 ms | ~6.5× slower (pessimistic TARGET) | 2.78 h | **11.6%** |

Even a flight VPU **6× slower than our laptop** needs only ~12% of the day to process the whole
imaging load — and that is a **ceiling**, because the learned gate ([report 04](04-onboard-gate.md))
sends only survivors to the detector; on a real swath of mostly-empty ocean it would drop most
tiles before they ever reach it. The compute fits the duty budget at *any* plausible VPU speed.

## 4. The one claim that is already hardware-independent

[Report 17](17-energy-model.md)'s headline **ES_proc = 57.6%** (the gating redesign saves more than
half the onboard processing energy) is, for the default all-CPU build, a **pure time ratio** —
`P_cpu` cancels, so it holds *whatever the processor draws*. That is the strongest portable result
in the project: it does not wait on a flight board. The absolute joules do; the *ratio* does not.

> ⚠ **The honest risk, carried from [report 02](02-detector-and-quantisation.md).** Flight VPUs
> typically run **INT8 / FP16**, and we *rejected* dynamic INT8 because it cost **−2.2 points on
> small ships**. So the quantisation a Myriad-class part needs is exactly the lever we found hurts
> our hardest class. This is not a blocker — static/QAT quantisation is a different regime than the
> dynamic ONNX path we tested — but it means **accuracy must be re-validated on the target**, and a
> demo must say so rather than assume FP32 numbers transfer.

## 5. What a real prototype would still have to do (the boundary we are not crossing)

Stated plainly so a judge sees we know where the demo stops:

1. **Hardware-in-the-loop** on a Myriad-class dev board / Ubotica or Unibap kit — real wall-clock
   latency and energy in **watt-hours**, replacing every laptop figure.
2. **Operator coverage & model port** — confirm every YOLOv8n layer maps to the VPU toolchain
   (OpenVINO / Ubotica SDK); fall back or re-architect any unsupported op.
3. **Quantisation accuracy re-validation** (§4) — measure small-ship recall under the target's
   INT8/FP16, not our rejected dynamic path.
4. **Thermal & duty-cycle** under the §3 load in a vacuum/thermal profile.
5. **Radiation tolerance / SEU handling** for the compute and memory — a flight-qualification item,
   not a laptop one.

None of these is started. All are **post-demo**. The demo's job is to prove the *idea* measures up;
the prototype's job is to prove the *board* does.

## 6. How to say it in the demo (one breath)

> "Every number you saw ran on a laptop — this is a demo, not flight hardware. But the work is a
> 3-million-parameter detector and a 47-thousand-parameter gate, and that is exactly the class of
> model our own baseline, Φ-sat-2, *already runs onboard* on a one-watt vision chip. At forty
> thousand tiles a day it is a few per cent of the duty cycle even on a VPU several times slower
> than our laptop, and the energy saving from our gating redesign is a time ratio that doesn't
> depend on the chip at all. What a flight prototype still owes is the wall-clock and the
> quantised accuracy on the board — and we're clear that we haven't done that yet."

---

### Sources
- [eoPortal — PhiSat-1 & -2 mission (Myriad 2, Ubotica CogniSAT, Eyes of Things)](https://www.eoportal.org/satellite-missions/phisat-1)
- [Intel Movidius Myriad X VPU product brief (up to 4 TOPS, >1 TOPS DNN, ~1.5 W)](https://www.intel.com/content/dam/www/public/us/en/documents/product-briefs/myriad-x-product-brief.pdf)
- [Unibap SpaceCloud iX5-100 product overview (CPU+GPU+FPGA+Myriad X, 10–30 W)](https://unibap.com/wp-content/uploads/2021/06/spacecloud-ix5-100-product-overview_v23.pdf)
- [Ultralytics YOLOv8 models — YOLOv8n 3.15M params / 8.7 GFLOPs @ 640](https://docs.ultralytics.com/models/yolov8)
- Internal: [report 02](02-detector-and-quantisation.md) (INT8), [04](04-onboard-gate.md) (gate), [12](12-end-to-end-integration.md) (end-to-end times), [17](17-energy-model.md) (energy model, ES_proc).
