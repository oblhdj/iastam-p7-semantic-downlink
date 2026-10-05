# 21 — The B0–B4 campaign, end to end: what relay actually buys

Closes `../docs/START_HERE.md` §5 item 4 ("run the B0–B4 campaign end to end") and [report
18](18-campaign-runner-spec.md)'s plan. B0–B3 were already wired; this report wires **B4** into the
runner via `sat7.relay.route` ([report 20](20-relay-path-choice.md), windows + path choice) with the
energy primitives from [report 19](19-relay-energy.md), and runs all five configs in one pass.
Code: `../code/scripts/wp18_campaign_runner.py`. Data: `../code/results/wp18_campaign.json`.

**Run parameters (fixed for every B3/B4 number below).** B3/B4 are the canonical operational load:
**40,000 tiles/day, link share 0.25, seed 0, orbit mix, `det_thr` 0.25, `conf_high` 0.670**,
`cubesat_sband`, 36 h over Sfax, 8 GB storage — the same load as the headline
`wp6_real_table.csv` "Ours: LoD + value-greedy" row (B3 reproduces it; §2). B1/B2 are a small-scale
detector sample (200 tiles / 2 swaths), REAL recall, independent of the daily load.

**Every recall figure in this report is at an assumed 15 % cloud fraction; band 0.40–0.82 across
0–50 % cloud** (START_HERE §7) — stated once here and repeated on each number, never quoted bare.

**Labels (a campaign mixes them by row):** B0 **SIM** (raw volume); B1/B2 recall **REAL**; B3 recall
**SIM-over-REAL**; **B4 latency SIM — under an assumed relay orbit (RAAN +90°, ASSUMPTION) and
`R_isl` = ground rate (ASSUMPTION)**; **B4 energy TARGET** (report 19, pending real `P_isl`/`R_isl`),
**λ_E/λ_T ASSUMPTION**. B4's recall and MB-sent are **inherited from B3 verbatim** (§1) and keep
B3's label — wiring into the campaign upgrades nothing.

## 1. The design question, resolved by checking: reroute, not extra capacity

Does relay **reroute** items B3 already scheduled (same items/bytes/recall, only latency+energy), or
add **capacity** that lets dropped items through (different recall)? Checked, not assumed:
* `sat7/scheduler.py`: all **eviction**, **per-pass capacity**, and **aging** are inside `simulate()`,
  which finishes **before** any routing.
* `sat7/relay.py`: `route(plan, links)` is a **post-B3 per-item path chooser** that injects **no
  passes** into `simulate()`.

So relay **cannot change which items are sent → recall and MB-sent are fixed by B3**. B4 is a
**reroute** — only latency and energy move. This matches reports 19–20 and is **asserted** by gate (b).

## 2. The campaign, one labeled row per config

**Recall is not one quantity across these rows — read the sample/cloud column before comparing any
two.** B1/B2 are REAL detector recall on small fixed samples with **no cloud model** (every ship is
visible); B3/B4 recall is a **simulated operational day at an assumed 15 % cloud fraction**. A B1/B2
number and a B3 number are therefore **not comparable down a single column**, so each recall is
reported with the ship population and cloud basis it rests on.

| config | onboard | ship recall (REAL/SIM) | sample · cloud basis | reduction vs B0 | label |
|---|---|---|---|---|---|
| **B0** | none | — | — | 60,124 MB/day raw ref | SIM |
| **B1** | YOLO per tile | **0.782** (95 % CI 0.73–0.83) | 262 ships / 200 tiles · **no cloud** | — | REAL |
| **B2** | SAHI + YOLO + fusion | **0.745** (95 % CI 0.62–0.87) @ 1.56× compute | 47 ships / 2 swaths · **no cloud** | — | REAL |
| **B3** | + gate + LoD + scheduler | **0.680** *(band 0.40–0.82)* | simulated day · **15 % cloud** | **554×** (108.5 MB/day) | SIM-over-REAL |
| **B4** | + relay / adaptive comms | **= B3** (0.680; reroute) | simulated day · **15 % cloud** | **554×** (108.5 MB) | latency SIM · energy TARGET |

(95 % CIs are Wald binomial on the ship count: B1 205/262, B2 35/47. B4 changes **latency, not
recall or bytes** — §3; its worst-case latency falls 11.58 h → 6.16 h at the same load.)

**B2 (0.745) comes in below B1 (0.782) — stated plainly, not hidden.** On these samples SAHI+fusion
scores 3.7 points under per-tile YOLO. But this is **not** a controlled SAHI-vs-YOLO result: the two
rows are different, tiny ship populations (47 vs 262 ships) and their 95 % CIs overlap heavily
(0.62–0.87 vs 0.73–0.83), so the ordering is **within sampling noise**. The controlled comparison is
[report 16](16-swath-policies.md), which runs every tiling policy on the *same* 24 swaths / 716 ships
at the **native 768 px window** — B2's window (`wp18_campaign.json` `B2.window_px = 768`): there SAHI
reaches **0.772 overall and 0.922 on seam ships, matching the never-cut oracle (0.774 / 0.909) at
1.56× compute**. So at scale SAHI does **not** lose to plain tiling, and the window that applies is
**768 px, not the paper's 512** (report 16 §"Window size": at 512 the policy ordering reverts to
report 13's). Note **report 16 does not itself put B1 against B2** (it compares tiling policies on one
fixed ship set), so it does not *literally* reconcile this B2<B1 number — what it reconciles is the
only form of the question that is controlled for sample and window.

**B3 reproduces the headline, and 557× stays authoritative.** The **557× / 108.0 MB headline
(`wp6_real_table.csv`, "Ours: LoD + value-greedy") remains the cited figure.** wp18's **554× /
108.5 MB** is the **same configuration reproduced within the 2 % traceability gate — not a
replacement** for 557×; the 0.5 % gap is the gate tolerance. Its recall **0.680** (15 % cloud; band
0.40–0.82) sits within the CSV row's documented **±0.005 day-resampling spread around its 0.6851**
(report 05 §"3 resampled days"; report 11 §sensitivity) — the gate tolerance is justified against
that spread in §4.

## 3. What B4 changes from B3, at the chosen λ

**Headline at λ_E = 0 (pure latency-min)**, deliberately: it is the case [report 20](20-relay-path-choice.md)
validated (f = 0.440), and it isolates the latency buy from the energy term, which is still TARGET.
A middle λ would blend a SIM result with a placeholder energy number.

All numbers below at **40k tiles/day, share 0.25, seed 0; recall at 15 % cloud (band 0.40–0.82)**.
**The latency figures are SIM under an assumed relay orbit** (`wp18_campaign.json`
`relay_raan_offset_deg = 90` — a RAAN +90° companion, **ASSUMPTION**) **and `R_isl` = ground rate
(ASSUMPTION)**; they move with that orbit and that rate, which are not yet sourced:

| quantity | B3 (direct) | B4 (relay reroute) | change |
|---|---|---|---|
| ship recall *(15% cloud)* | 0.680 | **0.680** | **0 (reroute)** |
| MB sent | 108.5 | **108.5** | **0 (reroute)** |
| latency median | 4.66 h | **1.99 h** | −2.67 h |
| latency p90 | 9.54 h | **4.51 h** | −5.03 h |
| **latency max (worst case)** | 11.58 h | **6.16 h** | **−5.42 h** |
| items via relay | — | **43.6 %** | — |
| total comm energy (TARGET) | 5.15 kJ | **6.92 kJ** | **+1.77 kJ (+34 %)** |

**Relay roughly halves worst-case latency (11.58 → 6.16 h) for +34 % comm energy** — the whole story
of B4: it spends energy to buy time, exactly as reports 19–20 framed it. The 11.58 h worst case is
Sfax's single-station gap ([report 09](09-whole-day-bound.md)); a complementary relay closes it. The
43.6 % relay fraction is **recomputed from the 40k run, not carried over** from any smaller-scale
run: `wp18_campaign.json` records **n_relayed / n_items_routed = 19,731 / 45,264 = 0.436** at
`B3.day_tiles = 40,000` (day_ships 21,137) — the routed-item count scales with the 40k plan,
confirming it is that plan being routed. It tracks report 20's 44.0 % over a uniform grid. (B4's direct-only energy
5.15 kJ also reconciles with [report 19](19-relay-energy.md)'s `E_comm` 5.12 kJ/day at 108 MB — the
two energy models agree on the same downlink.)

**Secondary — the energy trade (TARGET, λ_E ASSUMPTION).** With report 19's real joules, the relay
fraction falls as energy is weighted: 0.436 (λ_E 0) → 0.41 (50) → 0.315 (150) → 0.064 (500+).
Crossover ≈ λ_E 150–500 **at report 19's placeholder `P_isl`/`R_isl`**; where it truly lands is TARGET
until those are sourced.

## 4. Sanity gates — all pass before any number is trusted

* **B3 traceability gate (pins to a named CSV row).** B3 must reproduce
  `results/wp6_real_table.csv` row **"Ours: LoD + value-greedy"** (`ship_recall` 0.6851,
  `MB_sent` 108.0, `data_reduction_x` 557): **MB_sent within 2 %** (108.5 vs 108.0 ✓) and
  **ship_recall within 0.01** (0.680 vs 0.6851 ✓). The runner aborts if either fails.
  **Why 0.01 for recall:** WP6 measured the rep-to-rep (day-resampling) spread at **±0.005** (report
  05 §"3 resampled days"; report 11 §sensitivity, "±0.005 seed noise"), so one seed
  can differ from another realization by up to ~0.005; **0.01 ≈ 2× that spread**, the smallest
  tolerance that lets any single-seed run reproduce any documented realization of this row. It is not
  slack hiding an error: at this exact 40k / 0.25 / 15 %-cloud load, **reports 05 and 11 both report
  0.680 and report 12 reports 0.6798/0.6799** — so wp18's 0.680 *agrees* with three other reports;
  it is `wp6_real_table.csv`'s **0.6851** realization that sits 0.005 above them, and the gate's 0.01
  spans exactly that documented spread. MB, which has no resampling spread, is held to a tight 2 %.
* **(a) direct-only reproduces B3 exactly** — relay disabled, every item keeps B3's actual delivered
  latency and bytes: latency identical per item, MB equal to B3's to < 1e-3. **Pass.** Because B3 is
  pinned to the CSV row above, gate (a) ties B4's direct baseline to that row within its tolerance.
* **(b) reroute keeps recall & MB-sent** — B4's delivered-byte set equals B3's (fraction-independent).
  **Pass** — the executable form of §1's answer.
* **(c) report 20's f = 0.440 reproduced in-context** — `route` over report 20's uniform grid at
  λ_E = 0 with the wired-in energy primitives gives f = **0.440**. **Pass.**

## 5. Interface notes (surfaced and fed back, not patched around)

* **Catalogue-build adapter (B1/B2 → B3).** B1/B2 emit raw detections; B3 consumes per-tile catalogue
  frames (B2's unit is a swath). The adapter (`run_prefilter` + `flag_predictions` + GT match = wp11
  stage 4) is a required step; [report 12](12-end-to-end-integration.md) proved it reproduces the
  stored catalogue to 1e-4, so B3 runs from that catalogue. Named in report 18.
* **`direct_latency_s` seam (B3 → B4).** `route()` models direct latency as next-pass wait, but B3
  delivers mid-pass (median 4.66 h vs ~4.5 h next-pass). Rather than patch a peer-owned module, the
  `sat7.relay` owner added a backward-compatible `direct_latency_s` (default None = unchanged; wp20
  numbers verified unmoved); B4 passes B3's actual delivered latency through it, so gate (a) is exact.
* **One modeled asymmetry.** `T_direct` is B3's actual mid-pass latency; `T_relay` is still the window
  estimate (relay's next pass *rise*), since B3 never relayed. So `min(J)` is **slightly conservative
  against relay** — the −5.42 h cut is a mild under-estimate. A relay-side delivered-time model is
  future work.

## 6. Limits

* **B1/B2 are a small-scale detector sample** (200 tiles / 2 swaths), for wiring correctness; their
  REAL recalls are samples, not the full-split figures (report 02's 0.804 mAP50 is the detector).
* **B3/B4 are the canonical 40k-tile load** and reproduce the CSV headline, but one seed / one Sfax
  day; recall is a single resampling realization, within the CSV row's ±0.005 day-resampling spread.
* **B4 energy is TARGET** (report 19 placeholders; `P_isl`/`R_isl` ASSUMPTION-and-swept), so +34 %
  energy and the λ_E crossover are provisional. The **latency** cut is **SIM under an assumed relay
  orbit (RAAN +90°, ASSUMPTION) and `R_isl` = ground rate (ASSUMPTION)** — firm *given those
  assumptions*, not absolute; recall/MB are B3's and exact.
* **The ISL geometry is a SIM output, not literature.** The min ISL range 1252 km (report 19, from
  `wp20_isl_windows.csv`) is **SIM** — computed from the propagated RAAN +90° orbit — **not LIT**; it
  would become LIT only if matched to a published crosslink range, and it does not set `R_isl` until a
  link budget is applied.
* **Single relay, single ground station** — the benefit is complementary Sfax coverage; more stations
  shrink the direct gaps and the relay's win (report 20 §6).
* **B3 recall carries the 15 % cloud assumption** (band 0.40–0.82) and must never be quoted bare
  (START_HERE §7); all processing energy is laptop-derived (report 17 §6).

**START_HERE §5 item 4 is closed:** B0–B4 runs end to end, B3 is pinned to a named CSV row within an
explicit tolerance, every figure carries its label and its run parameters, the reroute-vs-capacity
question is answered and asserted, and the one piece that is not yet a measured result — relay energy
— stays labeled TARGET.
