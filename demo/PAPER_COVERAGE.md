# Accepted paper → Phase-3 evidence (coverage map)

**Paper:** *Energy-Aware Semantic Downlink for Maritime Object Detection with Optional
Inter-Satellite Relay* (Makni, Ben Saad, Ketata, Megdiche — ENIS; Track 4, Problem 7).
**Accepted in Phase 2.** It reports **no numbers by design** ("No numerical performance value is
reported without a corresponding experimental measurement") — it is a methodology + roadmap.
**Phase 3 = produce the measured values.** This map answers the judging question *"did you deliver
what the paper promised?"* line by line. Labels: **REAL** / **SIM** / **LIT** / **TARGET** / **ASSUMPTION**.

The demo is built to show this map in action: `python demo/summary.py` for the headline, then
`bash demo/run_demo.sh` to execute the B0→B4 chain live. Full narrative per item is in `reports/`.

---

## 1. The B0→B4 campaign (paper Table II) — **all five ran**
| cfg | paper definition | delivered | recall | label | evidence |
|---|---|---|---|---|---|
| **B0** | no onboard detection, full image (raw reference) | 60,124 MB/day raw baseline | — | SIM | `wp18_campaign.json`, [report 05](../reports/05-scheduler-on-real-detections.md) |
| **B1** | YOLO on original image, send detections | YOLOv8n, **one call on the whole 3072 px scene** (no slicing) | 0.339, precision 0.681 | REAL | `wp26_b0_b4.json` |
| **B2** | SAHI + YOLO, send detections | SAHI + global-coord fusion **on the same scenes** | 0.717, precision 0.696 @ 1.56× compute | REAL | `wp26_b0_b4.json`, [reports 15–16](../reports/16-swath-policies.md) |
| **B3** | SAHI + YOLO + semantic policy, send metadata/ROI | full LoD + value-greedy scheduler | 0.680 (@15% cloud) on its day; 0.685 three-day mean | SIM-over-REAL | `wp6_real_table.csv`, [reports 05–08](../reports/05-scheduler-on-real-detections.md) |
| **B4** | + direct/optional relay | relay reroute, `min(J_direct, J_relay)` | = B3 (reroute); worst latency 35.2→10.9 h, per-item median 7.9→3.4 h | latency SIM, energy TARGET | `wp18_campaign.json`, [reports 19–21](../reports/21-b0-b4-campaign.md) |

B1 and B2 share one input — 150 scenes of 4×4 real test tiles, 1,164 ships (`wp26_b0_b4.py`); B3
and B4 are a simulated 40,000-tile day. The two pairs are different inputs, so B2's detection
recall and B3's delivered recall are not steps of one curve. *Reference, not the paper's B1:* the
detector on native 768 px tiles (one call per tile) scores recall 0.766, precision 0.754 on all
8,173 test ships (`wp24_detection_eval.json`); until 10 Oct 2026 a 262-ship sample of that (0.782)
stood in this table as B1, and a 47-ship swath sample (0.745) as B2.

The B0→B4 campaign runs end to end (`wp18_campaign_runner.py`) with sanity gates that assert B3
reproduces the authoritative headline and B4 reproduces B3's recall/MB (relay changes only latency
+ energy). **Paper §VIII-F progression `B0→B1→B2→B3→B4` — closed.**

## 2. The 5 evaluation metrics (paper §VIII-E, eqs 27–30)
| paper metric | delivered as | value | evidence |
|---|---|---|---|
| **Downlink Reduction** `DR = 1 − D_tx/D_raw` | data-reduction factor | **341×** (DR = 99.71%) — three-day mean; coastal tiles at their measured JPEG size, thumbnails and crops still modeled. Earlier all-modeled estimate: 557× | `wp6_real_table.csv`, `wp28_coast_tile_model.json` (557×: `wp6_real_table_modeledcoast.csv`) |
| **Detection Quality** (P, R, F1, mAP, small-object recall) | detector eval, by size | mAP50 0.804; small-ship recall the tracked weak point | [report 02](../reports/02-detector-and-quantisation.md) |
| **Information Preservation** | `downlink_eff = recall / ceiling` + decision-flip integrity | **1.0** at nominal load; **0 decisions changed** on the real chain | [reports 05](../reports/05-scheduler-on-real-detections.md), [12](../reports/12-end-to-end-integration.md) |
| **Latency** `T_total = T_inf+T_enc+T_comm+T_dec` (relay: `T_ISL+T_GS`) | delivery latency, direct vs relay | per ship: median 4.5 h; per item: median 7.9→3.4 h and worst case 35.2→10.9 h with the relay. All of these are `T_comm` (wait + transmission, both relay legs); the other three stages are measured and add ≤ 71 ms | [reports 20–21](../reports/20-relay-path-choice.md), `wp29_validation.json` |
| **Energy** `E_total = E_proc + E_comm` | per-stage model + ES ratio | ES_proc 57.6%, ES_total ≈ 98%, proc:comm 5.5:1 — **estimates**: no power was measured, every P_k is an ASSUMPTION | [report 17](../reports/17-energy-model.md), `wp29_validation.json` |

## 3. The energy model (paper §V, eqs 8–19) — **built**
`E_proc = Σ P_k·T_k` (pre/tile/det/fusion/semantic), `E_comm,direct = P_tx·D_tx/R_tx`,
`E_comm,relay = E_ISL + E_GS`, `E_comm = min(direct, relay)`, `ES = 1 − E_proposed/E_baseline` —
all implemented in `wp17_energy_model.py` / [report 17](../reports/17-energy-model.md), reading every
stage time live from its source file with a sanity gate. Powers `P_k` are ASSUMPTION and swept.

## 4. The joint objective (paper §VI, eq 20) — **now solved as one program** (+ the proxy kept)
Paper: `min αE_total + βD_tx + γT_total` s.t. accuracy ≥ A_min, preservation ≥ I_min.
Delivered two ways, both honest: (a) `sat7.joint.solve` **solves the program directly** over each
object's P0–P3 level — unconstrained optimum (a true lower bound) + greedy constraint repair,
bracketed by a validity check like the scheduler's optimality gap, with `pareto()` sweeping (α,β,γ)
([report 23](../reports/23-paper-faithful-modules.md)); and (b) the **online** scheduler still
optimises value-per-byte (provably near-optimal, [report 08](../reports/08-optimality-gap.md)) and
the relay uses the exact `J = λ_E·E + λ_T·T` form ([report 20](../reports/20-relay-path-choice.md)).
Use (a) for the encoding-policy trade, (b) for delivered latency/throughput online. The joint `T`
term is per-item transmission time (a stated lower bound); end-to-end latency stays with the
scheduler/relay.

## 5. Phase-3 validation roadmap (paper §VIII-H, 10 steps)
| # | step | status | evidence |
|---|---|---|---|
| 1 | raw-image baseline | ✅ | B0, [report 05](../reports/05-scheduler-on-real-detections.md) |
| 2 | train/fine-tune YOLO | ✅ | [report 02](../reports/02-detector-and-quantisation.md) (mAP50 0.804) |
| 3 | integrate SAHI, optimise tile/overlap | ✅ | [reports 13, 16](../reports/16-swath-policies.md) |
| 4 | global-coord transform + fusion | ✅ | [report 16](../reports/16-swath-policies.md) |
| 5 | semantic records + ROI | ✅ | LoD encoder, [reports 05–06](../reports/06-byte-budget.md) |
| 6 | adaptive semantic-priority thresholds | ✅ | [reports 03, 07](../reports/07-queue-aware-encoding.md) (calibration + queue-aware) |
| 7 | processing + comm energy model | ✅ | [report 17](../reports/17-energy-model.md) |
| 8 | direct + optional relay paths | ✅ | [reports 19–20](../reports/20-relay-path-choice.md) |
| 9 | execute B0–B4 campaign | ✅ | [report 21](../reports/21-b0-b4-campaign.md), `wp18` |
| 10 | ablations + trade-off analysis | 🟡 mostly | §6 below; trade-off curves in reports 06/07/20 |

## 6. Planned ablations (paper §VIII-G)
| ablation | status | evidence |
|---|---|---|
| without SAHI | ✅ | B1 0.339 vs B2 0.717 on the same scenes, `wp26_b0_b4.json`; [report 16](../reports/16-swath-policies.md) |
| without tile overlap | ✅ | measured: recall 0.772 → 0.726 on 24 swaths (`wp24_detection_eval.json`); `PerceptionConfig(overlap=0.0)`, [report 23](../reports/23-paper-faithful-modules.md) |
| without detection fusion | ✅ | measured: precision 0.765 → 0.495, the double-count (`wp24_detection_eval.json`); `PerceptionConfig(fuse=False)`, [report 23](../reports/23-paper-faithful-modules.md) |
| without adaptive semantic downlink | ✅ | fixed vs queue-aware, [report 07](../reports/07-queue-aware-encoding.md) |
| without ROI transmission | ✅ | LoD rung ablations, [report 06](../reports/06-byte-budget.md) |
| without optional relay | ✅ | direct-only = B3 (sanity gate a), [report 21](../reports/21-b0-b4-campaign.md) |

## 7. Trade-off curves the paper promised (eqs 32–35)
`D_tx` vs recall → [report 06](../reports/06-byte-budget.md) sweeps; `E_total` vs `T_total` and
`(E,T)_direct` vs `(E,T)_relay` → [reports 19–20](../reports/20-relay-path-choice.md); `D_tx` vs
information preservation → downlink-efficiency curves, [report 05](../reports/05-scheduler-on-real-detections.md).

## 8. Where the implementation DIVERGES from the paper — say these before a judge does
1. **Dataset.** Paper suggests xView / VisDrone / DOTA / HRSID (SAR). We used **Airbus Ship
   Detection (optical)** — defensible (the paper allows "optical maritime datasets"), and SAHI is
   evaluated on **synthetic swaths** stitched from tiles, because Airbus tiles are already 768². A
   large-scene optical set would strengthen B2; flagged in `docs/START_HERE.md §7`. **Loaders for
   DOTA / YOLO / COCO-HRSID now exist** (`sat7.datasets`) with a SAHI eval path, so running B2 on a
   large-scene set is a download away — [report 23](../reports/23-paper-faithful-modules.md) §eval.4
   (detector needs fine-tuning on the new domain first; report recall honestly until then).
2. **SAHI window.** Paper says 512×512 / 20% overlap; we use **768** (window-size dependent —
   [report 16](../reports/16-swath-policies.md) shows the result reverses at 512). State the window.
3. **SAHI is a measured trade, not a free win.** [Report 13](../reports/13-tile-seams-and-sahi.md):
   a cut ship is usually still detected, so we argue **selective** slicing, not blanket SAHI.
   **Negative result:** on the stitched scenes SAHI (0.717) does not beat a plain tile-by-tile pass
   without overlap (0.730) and costs 1.56× the calls (`wp26_b0_b4.json`). The scenes are stitched
   from independent tiles, so no ship crosses a seam and the overlap has nothing to recover; B2's
   gain is over the whole-scene call (B1, 0.339), not over plain tiling.
4. **Onboard = TARGET.** All timings are a laptop; see [report 22](../reports/22-onboard-feasibility.md).
5. **Priority levels.** The paper's Table I names **P0–P3**; the measured pipeline used the richer
   **LoD ladder** (L0/L1/L2/tile/thumbnail). The P0–P3 scheme is now implemented verbatim
   (`sat7.priority`, [report 23](../reports/23-paper-faithful-modules.md)) and selectable alongside
   LoD; head-to-head it is leaner (90.5 vs 178.5 MB, same day, coastal tiles measured) at −6.3 pts
   recall — a trade, stated as one. (On the earlier flat coastal size the pair read 49.1 vs 108.5 MB.)
6. **Built, but NOT in the paper** (fold into the Phase-3 write-up under §IV-B *adaptive downlink
   policy*): the value-aware **multi-pass scheduler**, **dark-vessel/AIS** prioritisation, the
   **optimality bounds** ([reports 08–09](../reports/08-optimality-gap.md)), and the **leakage-free
   dataset split** ([report 01](../reports/01-data-integrity.md)). These are among the strongest
   results and currently have no home in the accepted narrative.
7. **The lead over the fair baseline is not robust under congestion (negative result).** In the
   80-setting sensitivity sweep (`wp10_sensitivity.csv`) "value-greedy ≥ FIFO" holds 80 of 80, but
   "ours ≥ the fair baseline" holds **74 of 80**. All six failures are at 160k tiles/day, where the
   lead is **1.3 points** (6.3 at 40k, where nothing fails): `thumb_value` 0.05 (−0.5 points) and 0.10 (−1.9), `conf_high` 0.90 (−0.02) and 0.99 (−0.4), `conf_low` 0.05 (−1.6) and 0.10 (−0.8).
