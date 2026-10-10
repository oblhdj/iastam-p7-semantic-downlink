# Repository audit — `iastam-p7-semantic-downlink`

**Date:** 9 Oct 2026 (Phase 3 closes 14 Oct). **Scope:** read-only audit. No code was changed;
this file is the only addition.

**Baseline:** GitHub `main` @ `d69f85d`. The local checkout is branch `cleanup/demo-focus` @
`9200e55`, which is `main` plus one pushed-but-unmerged fix (`demo/run_demo.sh` WSL paths). The
working tree also holds **uncommitted** work from 9 Oct: `demo/quickstart/`,
`code/tests/test_quickstart.py`, and edits to `README.md`, `demo/DEMO.md` and `.gitignore`. It also
holds an **untracked** `demo/dashboard.py`. Rows about those carry the tag **LOCAL**.

**Status vocabulary.**
* **implemented**: the code exists and I ran it or its tests.
* **partially implemented**: the code exists, but part of what is claimed is missing or rests on
  an unstated assumption.
* **missing**: no code does it. I checked the code, not just the README.
* **unverified**: the code exists, but nothing in the repo exercises it on real inputs.

**Priority vocabulary.**
* **critical**: a judge-facing number or conclusion is wrong or unsupported by the repo's own data.
* **high**: a headline claim rests on an unmeasured assumption, a reviewer cannot run a core
  path, or a compliance problem.
* **medium**: a correctness or fidelity gap that does not move a headline number much.
* **low**: hygiene.


> **Erratum (9 Oct 2026, found while completing the CV pipeline).** R8 below gave full-split B1
> recall as **0.725**. That figure is an artefact of the committed catalogue: `wp1_gt_matched.csv`
> → `wp6_ships.csv` match each ship to its best-overlapping box among **all** boxes ≥ 0.05, and
> only then apply the 0.25 cut to that box's confidence. A low-confidence fragment that overlaps
> a ship better than the confident box makes the ship count as missed at 0.25, even though the
> detector at 0.25 does find it. Thresholding first and then matching (what wp18's B1 and every
> detector run do) gives **0.766** on the same 8,173 ships (`results/wp24_detection_eval.json`,
> committed and live). The correction is applied to R8, G1, §4 and §5 below. The catalogue rule
> also feeds B3, whose recall therefore leans *conservative*; that is a new finding, not yet
> fixed (rebuilding the catalogue would move canonical results).

---

## 1. What was run (evidence log)

Nothing was downloaded except a `git clone` of the public repo (15 MB). The Airbus split, the
weights and the torch env were already on this laptop.

| # | What | Environment | Result |
|---|---|---|---|
| R1 | Fresh clone of GitHub `main` → `pytest code/tests` | `.venv` (Py 3.14, no torch) | **137 passed, 2 skipped** (both need torch), 11 s |
| R2 | `pytest code/tests/test_gate.py` | `.venv312` (torch 2.11 + CUDA) | **10 passed**: the 2 torch-gated tests pass |
| R3 | Full suite on the working tree (incl. LOCAL quickstart tests) | `.venv` | 146 passed, 3 skipped |
| R4 | `python -m sat7.<module>` self-checks (15 modules) | `.venv` | all exit 0. Only 6 have a self-check (b2_sahi_fusion, perception, priority, joint, energy, datasets, all printing OK); the other 9 have no `__main__` block, so their exit 0 proves nothing |
| R5 | `demo/summary.py` in the fresh clone | `.venv` | runs with no data and no GPU |
| R6 | `bash demo/run_demo.sh` (live B0→B4 + wp11 on 400 real tiles) | `.venv312`, RTX 5060, local dataset | **0 failures, 44.8 s**. B1 0.7824 (262 ships), B2 0.7447 (47 ships), B3 0.6799 / 108.5 MB / 554.1×. B4 sanity gates pass, latency 11.58→6.16 h. wp11 context agreement 1.0, 0 flips |
| R7 | 15 data-free scripts rerun in the fresh clone: wp5 ×2, wp6_fit, wp6_simulate_real, wp7_budget_sweep, wp8, wp10, wp10b, wp14, wp17, wp19, wp20, wp23, contact_windows, simulate_day | `.venv`, no torch, no data | **all exit 0**. Every current artefact regenerates **byte-identically**, including `wp6_real_table.csv` (the 557× row), wp5 (with `--frac-steps 16`, as `rerun_all.sh` uses), wp7, wp8, wp10, wp14, wp17, wp19, wp20, wp23. **Exception:** the legacy synthetic `sim_table.csv` / `sim_load_sweep.csv` (from `simulate_day.py`) no longer match the code |
| R8 | B1 recall on the **full** test split, from the committed `wp6_ships.csv` | `.venv` | ~~0.7249~~ (catalogue match-then-threshold, see erratum). **0.7659 @ conf 0.25 on 8,173 ships** thresholding first (wp24; small 0.569 / medium 0.915 / large 0.992) |
| R9 | Split integrity: file sets of `train`/`val`/`test` in `code/data/yolo_ships` | local data | pairwise **disjoint**; 0 dedup groups span splits. val and test both total 5,320 tiles / 8,173 ships but differ in composition (4,211 vs 4,234 ship tiles) |
| R10 | LOCAL quickstart: clean venv with no torch, then `--data` on the 9 real tiles | onnxruntime CPU | 2.1 s wall. ONNX boxes vs committed `wp1_predictions.csv`: 40/40 paired, max \|Δconf\| 3.4e-4, 0 flips at 0.25 and 0.670 |
| R11 | Ship-share arithmetic on the canonical simulated day (`workload_from_catalogue`, defaults) | `.venv` | 21,137 ships. 3,554 (16.8 %) on coastal tiles, of which 1,338 (**6.3 % of all ships**) have onboard conf < 0.25. 3,484 (16.5 %) on cloud tiles |

---

## 2. Feature audit

### A. Setup, data, licensing

| Feature | Files | Status | Evidence | Bugs / limitations | Recommended action | Priority |
|---|---|---|---|---|---|---|
| A1 Environment & setup | `README.md`, `code/requirements.txt`, `code/README.md`, `code/rerun_all.sh` | partially implemented | R1: the analysis env installs and tests pass from a clean clone | The torch env (`.venv312`: torch 2.11+cu128, ultralytics 8.4.160) has **no requirements file**; its versions exist only in prose. Analysis pins (numpy 2.5.3, pandas 3.0.6, opencv 5.0) are bleeding-edge and verified only on Python 3.14. `code/README.md` is stale ("17 tests", banner says STALE). `rerun_all.sh` covers WP5–WP11 only, none of wp12–wp23. skyfield raises a NumPy-2.5 `DeprecationWarning` (future break). | Add `requirements-gpu.txt`. Extend `rerun_all.sh` to wp12–wp23 with the flags each committed artefact used. Rewrite or delete `code/README.md`. | medium |
| A2 Leakage-free dataset build + near-duplicate dedup | `scripts/wp0_build_dataset.py`, `sat7/dedup.py`, `tests/test_dedup.py`, `results/wp0_stats.json` | implemented | 11 dedup tests pass. `wp0_stats`: 6,282 duplicate pairs, 1,617 groups a naive split would leak, 0 spanning splits. R9 confirms disjoint splits | Needs the ~30 GB Kaggle zip and ~35 min (`runtime_s` 2,110). Dedup recall (missed pairs) is unmeasurable without labels. Equal val/test totals look like a bug on first read (they aren't, R9). | Note in report 01 that val/test totals are balanced by construction. | low |
| A3 Legacy random-split converter | `scripts/airbus_to_yolo.py`, `scripts/eval_prefilter.py` | implemented (superseded) | Its own docstring warns that a random split leaks | `code/README.md` still points readers to it. | Mark deprecated and point to wp0. | low |
| A4 RLE ↔ masks ↔ boxes | `sat7/rle.py`, `tests/test_rle.py` | implemented | 5 tests pass | — | none | low |
| A5 Data & model licence compliance | `README.md` §Licence, `code/results/*.jpg`, `LICENSE` | partially implemented | README: "None of it is redistributed here." Kaggle rules §7A (non-commercial only) and §7B (no redistribution) were read 9 Oct | **4 tracked JPGs contain real Airbus pixels:** `prefilter_demo_gallery.jpg`, `prefilter_real_gallery.jpg`, `wp0_duplicates.jpg`, `wp0_duplicates_grid.jpg` (verified visually on `prefilter_real_gallery.jpg`). The weights' ONNX metadata says Ultralytics AGPL-3.0, while the repo is MIT. | Untrack the 4 images (scripts regenerate them) and fix the README sentence. Decide as a team whether to rewrite history. State the weight licence wherever weights ship. | **high** |

### B. Perception

| Feature | Files | Status | Evidence | Bugs / limitations | Recommended action | Priority |
|---|---|---|---|---|---|---|
| B1 Detector training | `scripts/wp1_train_detector.py`, `kaggle/wp1_train_kaggle.ipynb` | implemented | `wp1_metrics.json`: mAP50 0.804, P 0.817, R 0.729 on 5,320 test tiles [REAL] | Weights (`code/runs/`) are gitignored, so **nothing detector-dependent runs from a clone**. Training needs a GPU (Kaggle T4). Small-ship recall at the operating 0.25 cut is 0.496 (R8). | Publish weights with sha256 and licence (release asset, or the LOCAL quickstart's ONNX). | high |
| B2 Model loading / inference adapter | `sat7/perception.py` (`make_yolo_detect_fn`, `detect_image`), wp11/12/13/16/18 | implemented | R6: B1, B2 and wp11 executed live | Every caller hard-codes `code/runs/ships/weights/best.pt`. `wp18` and `wp11` `import torch` at module top, so even their B3/B4 halves need torch. `detect_image(mode="whole")` with the default `fuse=True` runs an extra greedy NMS @0.5 after YOLO's NMS @0.7, so "whole" ≠ raw detector output. | Lazy-import torch. Let wp18's B3/B4 run without it. Default `fuse=False` for whole mode, or document it. | medium |
| B3 ONNX FP32 export, INT8 evaluation | `scripts/wp1_export_int8.py`, `results/wp1_export.json` | implemented | ONNX FP32 recall == PyTorch in every size bucket. INT8 −2.2 pts on small ships and 8.5× slower → rejected [REAL]. R10 re-verified ONNX on CPU | `best.onnx` is not on `main` (the LOCAL quickstart copies it). | Commit with the quickstart. | low |
| B4 Confidence calibration | `scripts/wp2_calibrate.py`, `wp2_predict_split.py`, `tests/test_calibration.py` | implemented | ECE 0.0878 → 0.0099; `conf_high` 0.9 → 0.670 adopted; 7 tests | The stale 0.9 survives in wp11's integrity check (G2). | — | low |
| B5 Classic pre-filter | `sat7/prefilter.py`, `tests/test_prefilter.py` | implemented | 5 tests. wp11 context agreement 1.0 on 5,320 tiles | Failed as a gate (0.645 recall, report 04) but **still required**: it supplies the cloud/coast context and the "unconfirmed candidates" the encoder branches on. At **56.3 ms/tile on CPU** it is the costliest CPU stage (ONNX detector: 38.6 ms). See F1. | Count it in the energy/latency budgets, or derive context from the gate/detector. | medium |
| B6 Learned gate | `scripts/wp3_train_gate.py`, `wp3_score_tiles.py`, `models/gate.pt` (local), `tests/test_gate.py` | partially implemented | R2: 10/10 tests. Ship recall 0.988 vs the classic filter's 0.645 [REAL] | **Not in the B3 headline.** `wp6_tiles.csv` has no `gate_score`, so `real_workload._gate_says_empty` falls back to the classic `"empty_sea"` rule. The learned-gate re-key exists only in wp14 / report 15. `gate.pt` is untracked, yet README's pipeline diagram puts the 47k CNN in the chain. | Fold wp14's gate into the canonical B3 run, or relabel B3 "classic gate". Export the gate to ONNX. | medium |
| B7 SAHI slicing + global-coordinate fusion | `sat7/b2_sahi_fusion.py`, `sat7/perception.py`, `tests/test_perception.py` | implemented | Self-check and 7 tests. wp16 sanity gate: fused whole-swath recall == per-tile, Δ 0.0000 | Class-agnostic O(n²) greedy NMS (fine at these counts). The clamped last window overlaps unevenly. A fixed-shape ONNX needs window = 768. | none | low |
| B8 B2 measurement: swaths, seams, tiling policies | `scripts/wp15_build_swaths.py`, `wp16_swath_policies.py`, `wp12_seams.py`, `wp12b_cut_recall.py` | implemented | `wp16_swath_policies.json` [REAL], **24 swaths / 716 ships**: whole 0.774, **SAHI 0.772**, edge 0.772, naive 0.728; SAHI = 1.5625× compute | The swaths are stitched Airbus tiles, not real large scenes. **The headline B2 (0.745) is not this measurement** but a 47-ship sample (G1). | Quote wp16 for B2. | critical (via G1) |
| B9 SAHI ablations: no overlap / no fusion | `PerceptionConfig(overlap=0.0)`, `PerceptionConfig(fuse=False)` | partially implemented | Flags exist and are tested. PHASE3_RESULTS §9: "measured rows not yet run" | `demo/PAPER_COVERAGE.md` §6 marks both **✅**. | Run both rows on the wp16 swaths (minutes with GPU), or mark them 🟡. | medium |
| B10 Large-scene loaders (DOTA / HRSID) | `sat7/datasets.py`, `tests/test_datasets.py` | unverified | 7 tests on synthetic annotation lines | Never run on a downloaded split. The detector is not trained for those domains. | Keep as infrastructure; don't cite it as a result. | low |

### C. Semantic encoding, ROI, priority

| Feature | Files | Status | Evidence | Bugs / limitations | Recommended action | Priority |
|---|---|---|---|---|---|---|
| C1 Level-of-detail (LoD) encoder + measured size model | `sat7/scheduler.py` (`encode_lod`, `encode_tile`, `LoDConfig`, `SizeModel`), `scripts/wp4_measure_lod.py`, `wp6_fit_size_model.py` | implemented (as byte accounting) | Sizes measured on 551 crops / 300 tiles. Power-law R² 0.76 / 0.89. The fit regenerates byte-identically (R7) | L0 = 40 B is an assumption. The `LoDConfig.coast_tile_bytes` comment still says q40 had "no recall cost", contradicted by report 14 (−4.0 pts). | Fix the comment. Label 40 B as ASSUMPTION in tables. | low |
| C2 ROI extraction & semantic packet serialization | none at runtime; crops exist only in `wp4`, `wp7`, `wp13` | **missing** (sized, never built) | `grep` finds `imencode` only in those three measurement scripts. No code packs a 40 B record, crops or JPEG-encodes a ROI at runtime, or decodes on the ground. `Item`s carry sizes, not payloads | The "semantic packet" is an accounting abstraction, so T_enc and T_dec cannot be measured. PAPER_COVERAGE step 5 "semantic records + ROI ✅" overstates this. The LOCAL quickstart also only lists `Item`s. | Implement `encode_packet(frame, dets, level) → bytes` (struct-packed P1 record + JPEG ROI/context) and `decode_packet`. Measure real sizes against the size model. | **high** |
| C3 Paper Table I P0–P3 scheme | `sat7/priority.py`, `scripts/wp23_semantic_compare.py`, `tests/test_priority.py` | implemented | 7 tests. wp23 regenerates byte-identically: P0–P3 49.9 MB / recall 0.617 vs LoD 108.5 MB / 0.680 [SIM-over-REAL] | `classify()` escalates **every** ≥ 0.25 coastal detection to P3, confident ones included. The module docstring says only *uncertain* coastal ships. Thresholds are ASSUMPTION. | Decide the intended rule, align code and docstring, rerun wp23 if the code changes. | medium |
| C4 Queue-aware coastal LoD | `scheduler.simulate_online`, `escalation_for_pressure`, `scripts/wp8_queue_aware.py`, `tests/test_queue_aware.py` | implemented | 9 tests. wp8 regenerates byte-identically | The 4-parameter law is fitted to 5 measured points (held out on 2 link budgets, report 07). | — | low |
| C5 Coastal recompression, cheap products | `scripts/wp7_measure_cheap_products.py`, `wp7_budget_sweep.py`, `wp13_recompress_check.py` | implemented | wp7 regenerates byte-identically. wp13: q40 costs **−4.01 pts** of coastal recall [REAL] | q40 was kept knowingly (report 14), but the simulator still credits delivered q40 tiles at full recall (D3). | — | low |
| C6 Dark-vessel / AIS prioritisation | `LoDConfig.dark_mode="decoupled"`, `priority.dark_bump`, `tests/test_failure_modes.py` | implemented (simulated) | 14 fault-injection tests. Report 10's fix is in place | No AIS data: the dark flag is random with p = 0.10 [ASSUMPTION]. No demo can exercise it. | Keep it labelled. An AIS-matching stub would make it demonstrable. | low |

### D. Scheduling, optimisation, recall accounting

| Feature | Files | Status | Evidence | Bugs / limitations | Recommended action | Priority |
|---|---|---|---|---|---|---|
| D1 Value-greedy multi-pass scheduler + baselines | `sat7/scheduler.py` (`ValueGreedy`, `FIFO`, `NoBuffer`, `simulate`, `metrics`), `tests/test_scheduler.py` | implemented | R7: `wp6_simulate_real.py` regenerates `wp6_real_table.csv` byte-identically (557× / 0.685 row) | `aging_per_hour` does nothing (the repo's own finding). | — | low |
| D2 Recall credit for truncated progressive items | `Policy.plan`, `scheduler.metrics` | partially implemented | `plan` may truncate a progressive item down to 10 % of its bytes. `metrics` credits **every ship the item covers** as delivered at any fraction; only *value* is scaled (√f) | A coastal tile delivered at 10 % counts all its ships as found. This only bites under saturation (160k+ tiles/day), where the 2.26× vs FIFO claim lives. | Require a minimum fraction, or a measured detectability-vs-fraction curve, for recall credit. Report the sensitivity. | medium |
| D3 Ground-side recovery of ships the detector missed on coasts | `scheduler.encode_tile` (the coastal `"tile"` item carries **all ground-truth ids**), `metrics` | partially implemented (unstated assumption) | R11: **1,338 of 21,137 day ships (6.3 %)** sit on coastal tiles with onboard conf < 0.25. They count as recovered whenever the tile is delivered, which at nominal load is always. Report 14 measured the same detector at 0.63–0.67 recall on coastal tiles | This implies **100 % ground re-detection** of exactly the ships the onboard model missed. It is absent from `real_workload`'s stated ASSUMPTION list. **Up to ~6 pts of the 0.68 headline depend on it.** | State the assumption. Publish the lower bound beside 0.685: `coast_mode="mosaic"` (no ground recovery) is already implemented and swept in wp7/wp8. Or apply a measured ground recall. | **critical** |
| D4 Exact optimum & optimality bounds | `sat7/optimum.py`, `scripts/wp5_optimality_gap.py`, `wp5_joint_bound.py`, `tests/test_optimum.py` | implemented | 15 tests; both CSVs regenerate byte-identically. Greedy gap: **0.251 % mean over all 48 operational windows**; over the 22 saturated ones, 0.55 % mean / 6.26 % max. Small saturated windows: 6.67 % / 54.8 % | 0.251 % averages in unsaturated windows, where the gap is trivially 0. `PHASE3_RESULTS.md` §8 still quotes the stale **"4.62 %"** (report 08 updated it to 6.67 %). | Quote saturated-window figures. Fix PHASE3. | low |
| D5 Real workload from the catalogue | `sat7/real_workload.py`, `scripts/wp6_build_catalogue.py`, `tests/test_real_workload.py` | implemented | 13 tests. wp11 cross-checks the catalogue live | The 15 % cloud share is resampled from **21** real cloud tiles (×287, documented). 16.5 % of day ships are on cloud tiles and are lost by design. Context mix, tiles/day and dark share are ASSUMPTION. | Keep quoting the band. A cloud climatology would settle it. | medium |
| D6 Joint program (paper eq. 20) | `sat7/joint.py`, `wp23`, `tests/test_joint.py` | partially implemented | 8 tests; `solve()` is valid. But wp23's weighting sweep returns the **same assignment for every (α, β, γ)**: 0.52 MB = all 13,033 objects at P1, information preservation exactly I_min = 0.5, accuracy 1.0 | E, D and T all rise with level and nothing rewards detail beyond the I_min floor, so the optimum is always "cheapest feasible". Accuracy is trivially 1.0 (only already-detected objects count). The T term is ~0.002 h. There is no real trade-off surface. | Add a value/recall term, or make accuracy depend on level (e.g. ground-confirmation probability). Until then, don't present a Pareto surface. | medium |

### E. Communications: direct and relay

| Feature | Files | Status | Evidence | Bugs / limitations | Recommended action | Priority |
|---|---|---|---|---|---|---|
| E1 Orbit, passes, per-pass capacity | `sat7/orbit.py`, `scripts/contact_windows.py`, `results/passes.csv`, `tests/test_orbit.py` | implemented | 5 passes/day, 34.2 min contact, 649.2 MB/day capacity, R_eff 2.53 Mbps (`wp17_energy_model.json`) [SIM]. 2 tests | Link model and `link_share` 0.25 are ASSUMPTION. A pass already in progress at the window start is dropped. skyfield deprecation warning. | — | low |
| E2 Multi-ground-station | `orbit.find_passes_multi`, `max_gap_h`, `tests/test_multistation.py` | implemented (not in any headline) | 3 tests | Overlapping passes at two stations double-count capacity (the docstring acknowledges it). | Merge overlapping windows before scheduling. | low |
| E3 Relay: ISL windows + direct-vs-relay choice (B4) | `sat7/relay.py`, `scripts/wp20_relay_path_choice.py`, wp18 B4 | partially implemented | wp20 regenerates byte-identically. R6: the direct-only == B3 gate passes; worst-case latency 11.58 → 6.16 h; 44 % of items rerouted [SIM] | **No capacity model on the relay path.** An item is assumed to cross the next ISL window whatever its size, rate or the window's length, and the relay's ground pass has unlimited room: no contention with the relay's own traffic or with the primary's simultaneous pass. Results use RAAN +90°; `RelayConfig` defaults to +30°. Report 20 does not mention capacity. | Budget ISL bytes = R_isl × duration and relay-pass capacity, then rerun B4. | **high** |
| E4 Deadline-aware routing | `route_item(deadline_s=…)`, `tests/test_relay_deadline.py` | implemented (not exercised) | 4 tests | No campaign run sets deadlines. | Optional. | low |

### F. Energy and latency models

| Feature | Files | Status | Evidence | Bugs / limitations | Recommended action | Priority |
|---|---|---|---|---|---|---|
| F1 Per-stage energy model & ES ratios | `sat7/energy.py`, `scripts/wp17_energy_model.py`, `tests/test_energy.py` | implemented (equations); claims partially supported | wp17 regenerates byte-identically: ES_proc 0.5759, ES_total 0.9822, proc:comm 8.92 [stage times REAL on the laptop, powers ASSUMPTION] | **(a)** The "proposed" arm drops the 56.3 ms classic pre-filter ("the stage it replaces", report 17), but B3's encoder still needs its context, and wp11's measured chain runs it. With the pre-filter kept in both arms, ES_proc ≈ −1 % (the gate *adds* 0.94 ms; auditor's arithmetic on wp17's per-stage joules). **(b)** ES_total is measured against a bent pipe that would need **92.6 days of contact per day** (`bent_pipe_days_of_contact_needed` in wp17's own JSON). Against the link-limited bent pipe (201.7 MB/day actually sent, `wp6_real_table.csv`), E_comm ≈ **9.6 kJ/day** (auditor's arithmetic: 15 W × 201.7 MB / 2.53 Mbps), vs **50.8 kJ/day** total for the proposed pipeline. That is more energy, spent to deliver 0.68 vs 0.0024 recall. **(c)** Powers are laptop-class (CPU 28 W). | Restate ES_proc with the pre-filter in both arms, or show the context coming from the gate. Report **energy per delivered ship** against a capacity-limited baseline. Keep "98 %" only with its caveat. | **high** |
| F2 Relay energy | `scripts/wp19_relay_energy.py` | implemented (TARGET) | Relay vs direct 1.8× per byte; ES_weighted 0.9816 [TARGET] | P_isl and R_isl are invented. `sat7.relay`'s defaults are unit placeholders. | Keep the TARGET label. | low |
| F3 Latency T_total = T_inf + T_enc + T_comm + T_dec | `scheduler.metrics` (delivery times), `relay` | partially implemented | Median 4.49 h [SIM] = queueing + pass timing only | T_inf, T_enc and T_dec are not modelled; T_dec has no code at all (C2). Capture times are uniform over the day (ASSUMPTION). | Add the stage terms from measured stage times: small, but the paper names them. | low |
| F4 Onboard feasibility | `reports/22-onboard-feasibility.md` | partially implemented (TARGET, argued) | LIT mapping onto Myriad-class hardware; every timing is from the laptop | No hardware run. | Keep the TARGET label. | low |

### G. Evidence pipeline: campaign, integration, sensitivity, artefacts, tests

| Feature | Files | Status | Evidence | Bugs / limitations | Recommended action | Priority |
|---|---|---|---|---|---|---|
| G1 B0 → B4 campaign runner | `scripts/wp18_campaign_runner.py`, `results/wp18_campaign.json` | partially implemented | R6: 0 failures. B3 reproduces the wp6 row (108.5 vs 108.0 MB; 0.6799 vs 0.6851). B4 gates pass | The committed JSON says `"scale": "small-scale validation run … not a full campaign"`: **B1 = 0.782 on 262 ships** (Wilson 95 % CI 0.729–0.828), **B2 = 0.745 on 47 ships** (0.605–0.847). Full-split B1 is **0.766** (R8 corrected, n = 8,173) and full B2 is 0.772 vs whole 0.774 (B8, n = 716). README, slides, PHASE3 §2/§9 and PAPER_COVERAGE quote 0.782 / 0.745 and conclude *"SAHI is a measured trade, B2 < B1"*. Report 21 itself says the two are not comparable. **B2 → B3 is not chained**: B3 consumes the per-tile *whole-image* catalogue, so SAHI output never feeds the semantic stage (stated only in wp18's docstring). | Rerun wp18 at full scale, or quote R8 + wp16. Drop the B2 < B1 conclusion. State that B3's input is B1-type detections. | **critical** |
| G2 End-to-end integration check (wp11) | `scripts/wp11_integration_demo.py`, `results/wp11_integration.json` | implemented | Full 5,320-tile run: context agreement 1.0; recall 0.6798 identical to the catalogue replay. R6 reran 400 tiles | The committed JSON shows **1 decision flip at det_thr 0.25**, max \|Δconf\| 0.25, and 1 tile's FP count differing; report 12 explains this honestly. But PHASE3 §5 and PAPER_COVERAGE say "**0 decisions changed at every threshold**". `summary.py` prints only the `conf_high` flip, which line 238 checks at the **stale 0.9**, not 0.670. Needs `gate.pt`, weights, dataset and torch. | Check at 0.670 and print every threshold. Fix the two docs. | medium |
| G3 Sensitivity sweeps | `scripts/wp10_sensitivity.py`, `wp10b_interaction.py` | implemented | Regenerates byte-identically. CSV: 82 rows, ours > baseline and greedy ≥ FIFO in all 82 | Report 11 and README say "70 settings, held in 69". That count and the failing setting do not map directly onto the CSV. | Reconcile the count; name the failing setting with its row. | low |
| G4 Results artefacts & reproducibility | `code/results/*`, `code/rerun_all.sh` | implemented | R7: every current data-free artefact regenerates byte-identically from a clean clone | Detector-dependent artefacts need weights + dataset + torch. The legacy `sim_table.csv` / `sim_load_sweep.csv` are **stale against the current code** (e.g. Phi-sat-2 0.666 committed vs 0.958 regenerated) but still committed. `rerun_all.sh` is out of date. | Delete or regenerate the legacy `sim_*` files. Record the flags behind each artefact. | medium |
| G5 Automated tests | `code/tests/` (18 files on `main`) | implemented | R1–R3 | No tests for wp18 / wp11 logic, the relay geometry (`isl_windows`, `_los_clear`; only `route_item` deadlines are tested), the energy claims against wp17 values, or a pinned reproduction of the headline row. **No CI.** | Add a GitHub Actions job for the `.venv` suite and a regression test pinning the wp6 headline row. | medium |

### H. Demos and documentation

| Feature | Files | Status | Evidence | Bugs / limitations | Recommended action | Priority |
|---|---|---|---|---|---|---|
| H1 Instant summary (no data) | `demo/summary.py` | implemented | R5: runs from a clean clone, stdlib only | One screen shows **557× / 0.685** (wp6 table) and **554× / 0.680** (wp18 B3) without saying why. It shows only the stale-threshold flip (G2). | Print one canonical B3 or explain the pair. Show the det_thr flips. | low |
| H2 Live demo script | `demo/run_demo.sh` | implemented | R6: 44.8 s, 0 failures, writes only to `demo/_live/` | Needs bash, `.venv312`, weights, `gate.pt` and the dataset. `main` lacks the WSL path fix (`9200e55` is on `cleanup/demo-focus` only). | Merge `9200e55` into `main`. | low |
| H3 Slides & video script | `paper/slides/index.html`, `paper/video_script.md` | implemented | — | The slides quote B1 0.782 / B2 0.745 (G1) and ES_proc 57.6 % (F1). | Update after the G1 and F1 fixes. | **high** (judge-facing) |
| H4 Status documentation | `README.md`, `paper/PHASE3_RESULTS.md`, `demo/PAPER_COVERAGE.md`, `docs/STATUS.md`, `docs/ARCHITECTURE.md`, `code/README.md` | partially implemented | The superseded docs carry banners | Overclaims are listed in G1, G2, B9, C2, D4. PAPER_COVERAGE is ✅ almost everywhere, including items rated partial above. `code/README.md` is stale. | One correction pass once the measurement fixes land. | **high** |
| H5 CPU quickstart (**LOCAL**, uncommitted) | `demo/quickstart/*`, `code/tests/test_quickstart.py` | implemented (local only) | R10: 2.1 s without torch. 10/10 tests. Reproduces the committed detections on real tiles | Not on GitHub. Adds a 12.3 MB ONNX. Synthetic tiles (because of A5). Lists `Item`s rather than a real packet (C2). | Commit on a branch and open a PR. | high |
| H6 Streamlit dashboard (**LOCAL**, untracked) | `demo/dashboard.py` | partially implemented | Read, not run | Passes an **RGB** array to ultralytics, which treats numpy input as **BGR**. Costs every box ≥ 0.05 at a flat 900 B, while the operating cut is 0.25 and sizes come from the size model. "Raw" is the uploaded JPEG's size, not B0's 1.77 MB tile. It announces a gate decision without running a gate. Hard-codes "557×". Needs streamlit + torch, which are in no requirements file. | Fix it or delete it before showing anyone. | low |

---

## 3. Architecture summary

```
 Kaggle zip (~30 GB) --wp0+dedup--> code/data/yolo_ships (53,195 tiles, ~7.7 GB, leakage-free split)
        |                                    |
   wp1 train (GPU) -> best.pt/onnx      wp3 gate -> gate.pt         <- torch + data + weights
        |                                    |                         (detector-in-the-loop half)
        +--> wp1/wp2/wp6: catalogue CSVs ----+--> wp11 / wp12-13 / wp16 / wp18(B1,B2)
             (wp1_predictions, wp6_tiles/ships, wp3 scores)      REAL measurements
                         |
             ============|=============  committed in git (code/results/*.csv|json)
                         v
  sat7.real_workload -> encode (LoD | P0-P3) -> scheduler.simulate over sat7.orbit passes
     -> metrics -> wp5/6/7/8/10/14/23 ; energy (wp17/19) ; relay (wp20 -> B4)   <- no torch, no data
                                                              (simulation half, byte-reproducible)
  demo: summary.py (reads results) | run_demo.sh (both halves) | quickstart (LOCAL, ONNX on CPU)
```

The codebase is a pure-Python library, `code/sat7/`: pre-filter, SAHI/fusion, perception,
scheduler, priority, optimum, real_workload, orbit, relay, energy, joint, datasets, dedup and rle.
Around it sit one script per work package. The design splits cleanly in two:

* **The detector-in-the-loop half.** It needs torch, the weights and the dataset. It turns real
  tiles into **catalogue CSVs**.
* **The simulation half.** It needs neither. It replays those catalogues through the encoder,
  the scheduler, the orbit, energy and relay models.

That boundary is the repo's biggest strength: the simulation half regenerates **byte-identically**
from a clean clone (R7). wp11 checks the boundary itself by reproducing the catalogue live.

The weak joints:
1. B2's SAHI output never feeds B3 (G1).
2. The "semantic encoder" is a byte-accounting model with no real packet (C2).
3. The energy model's proposed pipeline is not the pipeline wp11 measures (F1).
4. The relay is a latency re-timing without capacity (E3).

---

## 4. The five most important issues

1. **Judge-facing B1/B2 numbers come from a smoke-scale run, and the conclusion drawn from them is
   contradicted by the repo's own larger data (G1, B8, H3).** B1 0.782 rests on 262 ships and B2
   0.745 on 47. Full-split B1 is **0.766** (erratum) and full B2 is **0.772 ≈ whole 0.774** (716 ships). The
   README, slides, PHASE3 and PAPER_COVERAGE all repeat the small-sample figures and the
   "B2 < B1, SAHI is a trade" reading, which report 21 itself says is unsupported.
   **Priority: critical.**
2. **The 0.685 headline assumes perfect ground-side recovery of the coastal ships the onboard
   detector missed (D3).** That is 6.3 % of all ships in the simulated day, credited whenever the
   coastal tile is delivered. This assumption appears nowhere in the stated assumptions. The
   lower bound is one config flag away (`coast_mode="mosaic"`). **Priority: critical.**
3. **The energy savings are measured against configurations that don't run (F1).**
   * ES_proc 57.6 % removes a pre-filter the encoder still depends on.
   * ES_total 98 % is relative to a bent pipe needing 92.6 contact-days per day. Against the
     link-limited bent pipe, the proposed pipeline uses *more* energy (~51 vs ~10 kJ/day) to
     deliver ~285× more ships (0.685 vs 0.0024 recall). That is a strong result, but a different one.

   **Priority: high.**
4. **The public repo redistributes Airbus pixels against competition rule §7B (A5).** Four tracked
   galleries contradict the README's own licence statement; the weights' AGPL origin is not
   stated. **Priority: high (compliance).**
5. **A clean clone cannot run any detection (B1, B2, A1).** The weights, gate and torch env are
   absent or unpinned, so only the summary and the simulation half run. The LOCAL quickstart
   (H5) fixes most of this once committed; a `requirements-gpu.txt` fixes the rest.
   **Priority: high.**

Next in line: the relay has no capacity model (E3); no semantic packet is ever built (C2); the
wp11 integrity claim is overstated and checked at a stale threshold (G2).

---

## 5. Proposed implementation plan, by priority

**P0: before 14 Oct. Mostly re-quoting and existing flags; a few minutes of GPU.**
1. **G1:** quote full-split B1 (0.766 @ 0.25, `wp24_detection_eval.json`) and wp16's B2 (0.772 vs 0.774,
   n = 716), or run `wp18 --full`. Remove the B2 < B1 conclusion. Update README, PHASE3 §2/§9,
   PAPER_COVERAGE, slides and `summary.py`.
2. **D3:** run `wp6_simulate_real` / `wp7` with `coast_mode="mosaic"` to publish the
   no-ground-recovery lower bound beside 0.685. Add "ground re-detects coastal misses" to the
   stated assumptions.
3. **A5:** untrack the 4 galleries and fix the README licence sentence. The team decides on
   history rewriting (it means a force-push).
4. **H5, H2:** commit the quickstart on a branch, open a PR, merge `9200e55`.
5. **Docs:** correct G2 ("0 flips at every threshold"), B9 (ablation ✅), D4 ("4.62 %"), H1 (the
   two B3 rows).

**P1: energy and integrity, about a day.**
6. **F1:** rerun wp17 with the pre-filter in both arms (or context sourced from the gate). Add a
   capacity-limited bent-pipe baseline and energy per delivered ship. Update report 17 and the
   slides.
7. **G2:** wp11 flips at 0.670 for every threshold, printed; rerun on 400 tiles (GPU, ~1 min).
8. **B6:** feed wp14's learned gate into the canonical B3, or relabel B3 "classic gate".

**P2: model fidelity, days.**
9. **E3:** add relay capacity and contention (ISL bytes per window, relay-pass budget); rerun B4.
10. **D2:** set a minimum delivered fraction for recall credit; sensitivity sweep.
11. **C2:** build a real packet encoder/decoder (40 B struct record, runtime JPEG ROI/context
    crops, ground decoder). Measure sizes against the size model; add T_enc / T_dec.
12. **C3, D6, B9:** settle the P3 rule; give the joint program a value term; run the
    overlap/fusion ablation rows; chain B2 output into B3.

**P3: hygiene.**
13. `requirements-gpu.txt`; `rerun_all.sh` covering wp12–wp23 with flags; CI; rewrite
    `code/README.md`; delete the stale `sim_*` artefacts; fix or remove `dashboard.py`; tests for
    wp18, wp11 and `relay.route`.

---

## 6. Environment and dataset blockers

**Does the full pipeline require GPU/torch and the full Airbus dataset? Partly. Here is what each needs:**

* **Torch: yes, for anything that runs the detector or the gate.** That is wp1, wp2_predict, wp3,
  wp11, wp12, wp12b, wp13, wp16 and wp18. wp11 and wp18 import torch at module top, so even
  their scheduling parts need it.
* **GPU: not strictly.** The inference scripts select CUDA if available and otherwise fall back to
  CPU (wp11, wp18, `perception.make_yolo_detect_fn`), but full-split CPU runs were not timed here.
  Training is GPU-only in practice (the detector was trained on a Kaggle T4).
* **Dataset: yes, for all detector-in-the-loop work.** You need the built split
  (`code/data/yolo_ships`, ~7.7 GB). That comes from the ~30 GB Kaggle zip, which requires a
  Kaggle account, acceptance of the competition rules, and ~35 min of wp0. The data may not be
  redistributed (§7B).
* **Weights:** `best.pt`/`best.onnx` and `gate.pt` are **not in git**. Without them, even a team
  member holding the dataset cannot run B1, B2 or wp11 from a clone. The gate retrains in ~2 min
  from the dataset; the detector needs a GPU training run.

**Paths that work without both (verified):**

| Path | Needs | What it proves |
|---|---|---|
| `pytest code/tests` | `.venv` | 137 / 2 skip on `main` (R1) |
| `python demo/summary.py` | any Python | reads committed results; no computation |
| wp5, wp6_sim, wp7, wp8, wp10, wp10b, wp14, wp17, wp19, wp20, wp23 | `.venv` | regenerates every scheduler / optimum / energy / relay / P0–P3 result **byte-identically** from committed CSVs (R7) |
| `python demo/quickstart/run_demo.py` (**LOCAL**, not on `main` yet) | numpy, opencv, onnxruntime, sgp4, skyfield | live pre-filter → YOLO (ONNX, CPU) → SAHI + fusion → P0–P3 → scheduler in ~2 s on synthetic tiles; with `--data`, reproduces the committed detections (R10) |

**Local environment as found:**
* `.venv`: Python 3.14, pinned analysis stack, no torch.
* `.venv312`: Python 3.12, torch 2.11 + cu128, ultralytics 8.4.160, onnxruntime 1.30; unpinned.
* RTX 5060.
* Dataset at `code/data/yolo_ships`; weights at `code/runs/ships/weights/`; gate at `code/models/`.
* Git Bash available for `run_demo.sh`.

---

*Method note: statuses come from reading the code paths, running the tests, self-checks and demos
listed in §1, and recomputing the key figures from the committed CSV/JSON files. Where this file
gives a number not found in the repo (the 9.6 kJ/day bent-pipe energy, the Wilson intervals, the
6.3 % coastal share, the saturated-window gap means), it is the auditor's arithmetic on repo
values, shown so it can be checked.*
