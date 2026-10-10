# Audit update — what changed since `REPO_AUDIT.md`, and what did not

**Date:** 10 Oct 2026. **Against:** [`REPO_AUDIT.md`](REPO_AUDIT.md) (9 Oct 2026, baseline `main` @
`d69f85d`). **State described:** branch `checkpoint/canonical-2026-10-10-v5`.

The original audit is left as written, with its own errata; this file is the difference. Each row
names an original finding by its identifier, says where it stands, and points at the file that
shows it. Statuses were checked against the repository on 10 Oct, not taken from commit messages:
a finding is **resolved** only where the code, the result file and the judge-facing documents now
agree. Nothing in this file changes a number; where a judge-facing document still disagrees with a
result file, that is listed in §3 and left for a decision.

Vocabulary: **resolved** (done and visible in the documents), **partly** (improved or disclosed,
with a stated remainder), **open** (as the audit found it), **unchanged** (was never a defect).

---

## 1. The five issues the audit ranked highest

| # | Original finding | Now | Evidence |
|---|---|---|---|
| 1 | **G1, B8, H3 — B1 / B2 came from a 262-ship and a 47-ship sample**, and "B2 < B1" was concluded from them | **resolved** | B1 and B2 are measured on the same 150 scenes of 4×4 tiles, 1,164 ships: 0.339 and 0.717 (`wp26_b0_b4.json`). The detector on native tiles (0.766, `wp24_detection_eval.json`) is labelled a reference, not B1. README, PHASE3 §2, PAPER_COVERAGE §1, slides, QA card and `summary.py` carry these; the old samples sit in `wp18_campaign.json` under `legacy_small_samples`. The negative result is stated: against plain tiling SAHI does not win on these scenes (0.717 vs 0.730) |
| 2 | **D3 — the 0.685 headline assumes the ground finds every coastal ship the onboard detector missed** (6.3 % of all ships in the day), credited whenever the coastal tile is delivered | **open** | The coastal tile item still carries every ground-truth ship on the tile (`LoDConfig.coast_mode = "tile"`, `scheduler.encode_tile`). The assumption is not in the stated assumptions of README, PHASE3 or the QA card, and it is not among the gaps `wp29_validation.json` lists. No lower bound without ground recovery (`coast_mode="mosaic"`) is published beside 0.685. The nearest evidence is `wp26`'s cross-check, where only transmitted detections count: 0.624 delivered on its own scenes (PHASE3 §2), a different input |
| 3 | **F1 — energy savings against configurations that do not run** | **partly** | *(b) resolved by disclosure:* ES_total is now explained as against a bent pipe that would need 92.6 days of contact per day; the link-limited figure (30.8 kJ/day, less than ours) is stated in README, PHASE3 §6 and the QA card. *(c)* every power is labelled an assumption, "no power was measured" is stated. *(a) open:* ES_proc 57.6 % still compares a baseline that runs the classic pre-filter (56.4 ms) with a proposed pipeline that does not, while the B3 encoder still takes each tile's context from that pre-filter. Neither PHASE3 §6 nor `wp29` mentions it. *New since:* a labelled SAHI variant in `wp17_energy_model.json` (70.0 kJ/day of processing, ES_proc 47.0 %) |
| 4 | **A5 — Airbus pixels redistributed** in three tracked galleries; weight licence unstated | **partly** | The three files are untracked and git-ignored on the checkpoint branches. They remain in earlier commits and on GitHub `main`; rewriting history is a team decision that has not been taken. The weights' AGPL-3.0 origin is stated in `demo/quickstart/README.md`, not in the top-level README |
| 5 | **B1, B2, A1 — a clean clone cannot run any detection** | **resolved for CPU inference; partly for the rest** | The ONNX detector and nine synthetic tiles are tracked. From a fresh clone, in new environments on Python 3.14 and 3.12, the quickstart and the dashboard run with no manual step (10 Oct). `best.pt` and `gate.pt` are still not published, so training-side and gate-in-the-loop scripts need a local build. `code/requirements-gpu.txt` now records the torch environment (recorded, not clean-installed) |

## 2. Every other finding

| ID | Finding | Now | Evidence / remainder |
|---|---|---|---|
| A1 | no requirements file for the torch env; `rerun_all.sh` covers WP5–11 only; `code/README.md` stale | **partly** | `code/requirements-gpu.txt` added; `rerun_all.sh` now runs every dependent script through wp29; `docs/REPRODUCE.md` documents the three environments. `code/README.md` is still the 18 Sept text under a "stale" banner |
| A2–A4 | dataset build, legacy converter, RLE | unchanged | — |
| B2 | detector adapter: hard-coded weights path, second NMS in whole mode, torch imported at module top | **partly** | `sat7.perception.load_detector` loads ONNX or `.pt`; whole mode no longer applies a second NMS. `wp11` and `wp18` still `import torch` at module top, so their scheduling halves need torch |
| B3 | `best.onnx` not on `main` | **partly** | tracked on the checkpoint branches (`demo/quickstart/model/`); `main` unchanged |
| B6 | learned gate not in the B3 headline | **open** | `wp6_tiles.csv` has no `gate_score`; `real_workload._gate_says_empty` falls back to the classic rule. README's pipeline diagram still places the 47k CNN in the chain |
| B9 | "without overlap" / "without fusion" marked done but not run | **resolved** | measured on 24 swaths, 716 ships (`wp24_detection_eval.json`): no overlap 0.726 recall, no fusion precision 0.495; PHASE3 §9 and PAPER_COVERAGE §6 quote them |
| B10 | DOTA / HRSID loaders never run on real data | open | as found; kept as infrastructure |
| C1 | L0 = 40 B is an assumption; stale "no recall cost" comment | **partly** | the serialized record is measured (34 B with packet header and CRC) and the q40 comment now states the 4.0-point cost. One source comment still gives the coastal tile as "62 % of the budget" (76 % since the tiles were measured) |
| C2 | no semantic packet is ever built | **resolved** | `sat7/semantic.py`: binary records, JPEG ROI and context crops, CCSDS packets, ground decoder; measured in `wp25_semantic_packets.json`; the demos transmit and decode the real stream |
| C3 | P3 rule and its docstring disagree | **resolved** | `PriorityConfig.coast_escalation` ("all", the measured default, or "uncertain"), documented in `demo/quickstart/README.md` |
| D2 | a tile truncated to 10 % credits all its ships | **partly** | `Item.min_fraction` exists and real items carry the measured end of their first JPEG scan. That a truncated product still delivers its ships remains an assumption; it is now stated wherever the flat recall is quoted (README, PHASE3 §5, QA card, `wp29` gap) |
| D4 | optimality gap quoted from unsaturated windows; stale "4.62 %" | **resolved** | PHASE3 §8 quotes 0.88 % at operational scale (checked by the `wp29` ledger against `wp5_optimality_gap.csv`); the stale figure is gone |
| D5 | cloud share rests on 21 tiles | unchanged | disclosed; recall is quoted as a band |
| D6 | joint program returns one assignment for every weighting | **open** | `wp23_semantic_compare.json` → `weighting_sweep` still shows information preservation at its 0.5 floor and 0.52 MB in each row |
| E3 | relay has no capacity model | **partly** | `sat7/comms.py` + `wp27_comms.json`: both legs, volume, rate and capacity, direct-only reproduces the scheduler. The canonical B4 row (`wp18`) still uses the window estimate; stated as a gap in PHASE3 §12 and the QA card |
| F2, F4 | relay energy and onboard feasibility are TARGET | unchanged | — |
| F3 | T_inf, T_enc, T_dec not modelled | **resolved** | `sat7/accounting.py` states the convention; the three stages are measured and add at most 71 ms (PHASE3 §7, `wp29` item 9) |
| G2 | "0 decisions changed at every threshold" overstates `wp11` | **partly** | `wp11_integration.json` → `decision_flips`: `det_thr` 1, `conf_low` 1, `conf_high` 0. Both names are the same 0.25 threshold in that run, so this is one ship in 8,173. PHASE3 §5 was corrected to say so on 10 Oct (`df02727`). Remaining: `demo/PAPER_COVERAGE.md` §2 still says "0 decisions changed on the real chain", and the script still checks `conf_high` at 0.9, not the 0.670 in use. `summary.py` and DEMO.md correctly say "@conf_high = 0" |
| G3 | sensitivity count did not map onto the CSV | **resolved** | 80 settings; "ours ≥ fair baseline" in 74, the six failures named, all at 160,000 tiles/day (README, PHASE3 §8 and §10, QA card; `wp10_sensitivity.csv`) |
| G4 | reproducibility; stale `sim_*` files | **partly** | data-free scripts rerun in a fresh clone on 10 Oct (`docs/REPRODUCE.md` §3 gives the result). The legacy `sim_table.csv`, `sim_load_sweep.*`, `sim_latency.png`, `sim_recall.png` are still tracked |
| G5 | tests; no CI | **partly** | 139 tests on `main` at the audit → 294 on this branch, in three tiers (`code/pytest.ini`); still no CI |
| H1 | `summary.py` shows two B3 figures without saying why | **partly** | it now prints the 341× three-day headline and, in the B0→B4 table, that day's own 337× / 0.680; the reason is in PHASE3 §2, not on the screen |
| H2 | WSL path fix not on `main` | **partly** | on every checkpoint branch; `main` unchanged |
| H3 | slides quote the old B1 / B2 | **resolved** | `paper/slides/index.html` carries 0.339 / 0.717 and 341× |
| H4 | status documents overclaim | **partly** | README, PHASE3, PAPER_COVERAGE corrected. `docs/START_HERE.md`, `docs/STATUS.md`, `docs/ARCHITECTURE.md` and the bodies of reports 05–22 still narrate the earlier estimate, under banners |
| H5 | CPU quickstart uncommitted | **resolved** (on the checkpoint branches) | tracked, tested, run from a fresh clone |
| H6 | Streamlit dashboard | **resolved** | see the rewritten row in `REPO_AUDIT.md` |

## 3. Where a judge-facing document and a result file still disagree

Reported, not changed.

1. **`demo/PAPER_COVERAGE.md` §2, "0 decisions changed on the real chain"** — `wp11_integration.json`
   records one ship changing side at the 0.25 cut (G2). The same claim in PHASE3 §5 was corrected on
   10 Oct. Both documents, and the README, say the chain reproduces the catalogue "to 4 decimals";
   the recalls are 0.6798 and 0.6799.
2. **README, the test counts** — "254 tests (5 skip in `.venv`…)" and "249 pass, 5 skip". The suite
   is now 294 tests: 278 pass and 16 skip in `.venv`, 291 pass and 3 skip in `.venv312`.
3. **README "Running it"** — it does not mention the dashboard or `demo/launch.py`.
4. **The headline's coastal assumption (D3)** and **ES_proc's pre-filter (F1a)** are not stated in
   any judge-facing document. These are omissions, not wrong numbers.
5. **Slide 9 of `paper/slides/index.html`** — "137 tests pass", and the commands shown are
   `demo/summary.py` and `bash demo/run_demo.sh`; neither the quickstart nor the dashboard appears.
6. **A source comment** (`LoDConfig.coast_mode` in `code/sat7/scheduler.py`) still gives the coastal
   tile as "62% of the budget"; it is 76 % since the tiles were measured. `docs/CONFIGURATION.md`
   quotes source comments and says so.

The four documents checked against this step's work are in §6.

## 4. Found since the audit

| Finding | Where it is recorded |
|---|---|
| **A nominal day does not fit a day.** It offers 130 % of what the 25 % link share sustains per 24 h (134.5 MB). Over six consecutive days value-greedy holds recall at 0.677–0.688; FIFO's median latency goes 9.4 h → 42.9 h | `wp29_validation.json` → `steady_state`; README, PHASE3 §5 and §12, QA card, DEMO.md |
| **Headline 557× → 341×** once each coastal tile is charged its own measured size; 24 % of the bytes are still model sizes | `wp28_coast_tile_model.json`, `wp6_real_table.csv` |
| **Units, energy, latency and fairness checked**: 38 checks, none failing, five stated gaps | `wp29_validation.json`, PHASE3 §12 |
| **Windows power throttling** slows a background demo process about 25-fold; the demo processes opt out | `demo/host_power.py`, `demo/DASHBOARD.md` |
| **Upload path**: an empty label file, a malformed one and a leftover one each misbehaved; fixed | `REPO_AUDIT.md` row H6 |
| **First run** of the dashboard stopped at Streamlit's email prompt; fixed | `demo/.streamlit/config.toml` |

---

## 5. Feature status

**Implemented** means the code exists and was run (by its tests, by a committed result, or in the
demo). **Partly** means it runs but rests on a stated assumption or covers part of what is claimed.
**Not implemented** means no code does it.

### Implemented

| Feature | Where | Shown by |
|---|---|---|
| Leakage-free dataset build with near-duplicate grouping | `scripts/wp0_build_dataset.py`, `sat7/dedup.py` | `wp0_stats.json`: 0 groups span splits |
| Detector training and evaluation (YOLOv8n) | `scripts/wp1_train_detector.py` | `wp1_metrics.json`: mAP50 0.804 |
| ONNX FP32 export; INT8 evaluated and rejected | `scripts/wp1_export_int8.py` | `wp1_export.json` |
| Confidence calibration | `scripts/wp2_calibrate.py` | `wp2_calibration.json` |
| Whole-image detection (B1) and SAHI with global-coordinate fusion (B2), overlap and fusion as switches | `sat7/perception.py`, `sat7/b2_sahi_fusion.py` | `wp26_b0_b4.json`, `wp24_detection_eval.json` |
| Semantic records, ROI and context crops, CCSDS packets, ground decoder | `sat7/semantic.py` | `wp25_semantic_packets.json`; the demos |
| P0–P3 priority policy (paper Table I) | `sat7/priority.py` | `wp23_semantic_compare.json` |
| Level-of-detail encoder with measured sizes | `sat7/scheduler.py` | `wp6_real_table.csv` |
| Value-greedy multi-pass scheduler, FIFO and no-buffer baselines | `sat7/scheduler.py` | `wp6_real_table.csv` |
| Exact optimum and optimality bounds | `sat7/optimum.py` | `wp5_optimality_gap.csv` |
| Queue-aware coastal level of detail | `scheduler.simulate_online` | `wp8_queue_aware.csv` |
| Orbit, passes and per-pass capacity (SGP4) | `sat7/orbit.py` | `passes.csv` |
| B0–B4 as modes of one pipeline on shared inputs | `sat7/campaign.py`, `scripts/wp26_b0_b4.py` | `wp26_b0_b4.json`, five checks |
| Two-path link simulator with capacity (direct / relay) | `sat7/comms.py` | `wp27_comms.json` |
| Units, data-reduction, energy and latency accounting | `sat7/accounting.py` | `wp29_validation.json` |
| Sensitivity sweeps | `scripts/wp10_sensitivity.py`, `wp10b_interaction.py` | `wp10_sensitivity.csv` |
| Terminal quickstart, dashboard, static fallback, launcher | `demo/` | 294 tests; fresh-clone runs |

### Partly implemented

| Feature | What is missing or assumed |
|---|---|
| Learned gate in the pipeline | trained and measured (recall 0.988), but the B3 headline uses the classic pre-filter's "empty sea" rule; not in the CPU demos (torch only) |
| Energy model | equations and stage times are real; every power is an assumption; ES_proc omits the pre-filter the encoder still needs; the SAHI variant assumes fusion costs 0 ms |
| Delivered recall under congestion | a truncated progressive product counts as delivering its ships |
| Coastal ships the detector missed | counted as recovered on the ground whenever the tile arrives |
| Byte sizes | coastal tiles measured; thumbnails and crops modeled (24 % of the bytes) |
| Relay in the canonical B4 row | window estimate; the capacity-aware simulation exists beside it |
| Joint program (paper eq 20) | solves, but returns the cheapest feasible assignment for every weighting |
| Dark-vessel prioritisation | implemented and fault-tested; no AIS data, the flag is random |
| Multi-ground-station passes | implemented; not in any headline; overlapping passes double-count |
| Large-scene dataset loaders | tested on synthetic annotation lines only |
| Cloud handling | the pre-filter drops cloud tiles; the assumed 15 % share rests on 21 real cloud tiles |

### Not implemented

| Feature | Note |
|---|---|
| Execution on flight hardware | every timing is a laptop; onboard is a stated target |
| Measured power | no instrument was used; every joule is an estimate |
| AIS matching | no AIS feed exists in the data |
| A real large-scene evaluation of SAHI | scenes are stitched from independent tiles |
| Continuous integration | tests are run by hand |
| Ground-side re-detection on delivered coastal tiles | assumed perfect, never run as a stage |

---

## 6. Cross-check of four documents against the demo and reproduction work

`demo/DEMO.md`, `demo/QA_CARD.md`, `paper/video_script.md` and `demo/quickstart/README.md` were read
in full on 10 Oct against what now exists (the dashboard, `demo/launch.py`, the static fallback,
the power opt-out, the test tiers, `docs/REPRODUCE.md`). **Every number in the four still matches
its result file** — the ledger in `wp29_validation.json` covers them and its tests pass, and the
quickstart figures were re-run (45,681 B on the synthetic tiles; 158,034 B measured and 153,948 B
modeled on the real ones; 40 of 40 boxes paired, no decision flip). What no longer matches is
about commands and coverage, not results. None of it was edited.

| Document | Still accurate | No longer matches what exists |
|---|---|---|
| `demo/DEMO.md` | every figure; the three commands it gives still work; "~2 seconds" for the quickstart (1.6–2.0 s measured) | The heading says "Three ways to run it" and a fourth, the dashboard, is now pointed at beneath it. `demo/launch.py` is not mentioned. "The recommended live demo" is the terminal quickstart, and "Click-by-click (live demo)" and the 90-second talk track describe the terminal path, while `demo/LIVE_DEMO_SCRIPT.md` scripts five minutes on the dashboard: two live-demo scripts exist and neither names the other. The pre-flight checklist has no dashboard or power-throttling item. "If something breaks" does not mention `python demo/launch.py --fallback` |
| `demo/QA_CARD.md` | every figure and every answer | "Did you measure power?" quotes ES_proc 57.6 % alone; the labelled SAHI variant (70.0 kJ/day of processing, 47.0 %) is in PHASE3 §6 only. There is no answer for the question the dashboard invites: "is what I am watching real data?" (the bundled tiles are synthetic, labelled SYNTH). The one-line pitch says "100 % of what the satellite still knew" without the single-day qualifier the README attaches to it |
| `paper/video_script.md` | every figure; every source file it lists exists | Shots 1 and 4 put a real Airbus tile in a published video, while `demo/quickstart/README.md` reads the competition rules (§7B) as forbidding even a handful of demo tiles. A decision for the team, not an edit. "The gate decides there is something worth a closer look" describes the `wp11` chain; the CPU demos run no learned gate, so dashboard footage would not show that step. The six-day and 74-of-80 findings are absent, by the two-minute design |
| `demo/quickstart/README.md` | every figure, the model checksum, the 9 Oct reproducibility check (re-run: same result), the runtime (27–36 ms per call measured today) | `run_demo.py` now imports `demo/host_power.py`, one folder up, so "one script" and the Files table are incomplete. The Windows power-throttling opt-out is not mentioned (throttled, the run took 27 s; see `demo/DASHBOARD.md`). Neither the dashboard nor `python demo/launch.py --quickstart` is pointed at. The end-to-end test it lists is now in the `inference` tier |
