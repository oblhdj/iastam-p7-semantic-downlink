# IASTAM P7 — HANDOFF (scan + organize)

_Written 2026-10-05. Phase 3 window is 1–14 Oct, so ~9 days left. Deep state & every
number live in [`docs/STATUS.md`](docs/STATUS.md); this file is the map + the decisions._

## 1. What this project is (one line)
Satellite that downlinks **information about ships, not pictures**: a confidence-aware
level-of-detail encoder + a value-aware multi-pass scheduler. Headline: **557× less data
than raw**, ~0.685 ship recall at 15% cloud, provably near-optimal scheduling.

## 2. Where we are  (Phase 3, 1–14 Oct)
- The **Phase-2 paper is accepted and FIXED** ("Energy-Aware Semantic Downlink…"). It reports **no
  numbers by design**, so there is **no "final paper" to rewrite** — Phase 3 produces the measured
  values *against its framework*. Full paper→evidence map: **`demo/PAPER_COVERAGE.md`**.
- **Engineering backlog is empty.** WP0–WP22 are done and measured (86 tests pass); the paper's
  whole B0→B4 campaign, energy model, relay, and 9/10 roadmap steps are closed.
- Repo is on git, clean, restructured for publication.
- **What's left is the Phase-3 write-up/presentation + the 2-min video + a demo** — all of which
  draw on `reports/` and `demo/`. (The old PDFs in `paper/` — the *internal* technical report, not
  the accepted paper — still carry superseded numbers; see §7.)

## 3. What was achieved (narrated in `reports/`, numbers in `code/results/`)
| WP | Report | Result in one line |
|----|--------|---------------------|
| 0  | `reports/01` | 53,195 tiles, leakage-free split (19.1% have a duplicate twin) |
| 1  | `reports/02` | detector mAP50 0.804; ONNX FP32 lossless; INT8 **rejected** (slower, −2.2pts small) |
| 2  | `reports/03` | detector underconfident; isotonic cuts ECE 8.9×; `conf_high` 0.9→0.670 |
| 3  | `reports/04` | 47k-param learned gate keeps 98.8% ships, skips 40% empty tiles |
| 5/6| `reports/05` | real detections drive scheduler: 557× reduction, 0.685 recall, 2.26× FIFO |
| 7  | `reports/06` | 86% of downlink wasn't semantic; −19% bytes at zero recall cost |
| 8  | `reports/07` | queue-aware encoder within 1.1 pts of best fixed policy at every load |
| 5  | `reports/08` | exact DP: greedy 0.251% from optimal at scale |
| 5b | `reports/09` | perfect foresight worth ≤0.2% → **don't build arrival prediction** |
| 9  | `reports/10` | FMEA → 14 fault-injection tests; found + fixed a dark-vessel ranking bug |
| 10 | `reports/11` | every constant swept; conclusions hold 69/70; cloud fraction dominates |
| 11 | `reports/12` | real JPEGs reproduce the sim to 4 decimals |
| 12–14 | `reports/13–14` | tile seams / SAHI / coastal recompression measured |
| 15 | `reports/15` | thumbnail gating on learned gate: +3.8% downlink back, free |
| 16–21 | `reports/16–21` | Phase 3: swath policies, energy model, B0–B4 campaign + relay |

Read-three shortcut: `reports/05`, `reports/12`, `reports/10`.

## 4. Project map (what lives where)
```
code/sat7/        core package (orbit, scheduler, prefilter, rle, dedup, real_workload, relay, optimum)
code/scripts/     one+ script per work package (wp0_* … wp20_*)
code/results/     ALL outputs — csv/json/log/png, FLAT on purpose (see §6)
code/tests/       86 pytest tests
code/data/        8.6 GB Airbus dataset   (gitignored, regenerable)
code/.venv312/    5.0 GB py3.12 + torch/CUDA  (gitignored) — detector/gate/ONNX work
code/.venv/       745 MB py3.14, no torch     (gitignored) — sim WPs + pytest
reports/          01–21 numbered write-ups (the current account) + README index
paper/            LaTeX + 3 PDFs + figures/ (figures copied from results/ for the build)
docs/             STATUS.md (deep state), START_HERE.md, ARCHITECTURE.md, GLOSSARY.md
```

## 5. Duplicate-file verdict (you asked about this)
**There is almost no real duplication.** What looked like duplicates isn't:
- "WP11 has 4 files" (`wp11_full_run.log`, `wp11_integration.json`, `wp11_per_tile.csv`,
  `wp11_run.log`) → these are **different artifacts** (run log vs data vs per-tile table),
  not copies. Same for every WP. Leave them.
- The only byte-identical copies are 5 PNGs intentionally mirrored into `paper/figures/`
  so LaTeX can build — **keep them**.
- One genuine stray: `Desktop/IASTAM_START_HERE.md` is an **exact copy** of
  `docs/START_HERE.md`, sitting loose outside the repo (see §8).

## 6. Organization verdict (you asked: put each WP's files in its own folder?)
**Recommendation: do NOT subfolder `code/results/`.** Reason you correctly predicted:
scripts read each other's outputs **by exact path**. Examples confirmed in the code:
- `wp16_swath_policies.py` and `wp18_campaign_runner.py` both read
  `results/wp15_manifest.json` and `results/wp15_ships.csv`.
- `wp17_energy_model.py` reads `results/passes.csv`.

Moving `wp15_*` into `results/wp15/` breaks ≥6 cross-WP references at once. The flat
`results/` dir is a deliberate shared data-bus between stages, right before a deadline it
is pure risk for zero scientific gain. **The code itself is already well-organized**
(package / scripts-by-WP / tests / numbered reports) — the thing you wanted is already done.
If you ever do want per-WP folders, it's a code change (every `ROOT/"results"` default +
each cross-reference), not a file move — defer to post-deadline.

## 7. What's left (Phase 3, highest value first)
1. **Paper + 2-min video.** ⚠ Rebuilding the PDFs will NOT fix them — `build_report.py`
   reads only `wp0_stats.json` and is written around the old synthetic workload. Making the
   PDF current means **rewriting content**, drawing from `reports/` (which are current).
2. WP8 escalation signal still keys on the classic pre-filter (gate can't localise) — a
   small localising head would drop the classic stage, worth 56 ms/tile. Phase-3 idea.
3. Edge-triggered re-inference for tile seams (matches SAHI at 61% cost) — when moving to
   wide swaths.
4. Small-ship recall — still the single largest loss (INT8, coastal recompress, seams all
   hit it first).

## 8. Cleanup candidates — ⚠ NOTHING DELETED YET, awaiting your OK
All of these are safe; none affect results or the pipeline. Pick which to remove:

| # | What | Size | What it is / why safe |
|---|------|------|-----------------------|
| A | `Desktop/IASTAM_START_HERE.md` | 11 KB | Exact copy of `docs/START_HERE.md`, loose on Desktop. Keep the repo one. |
| B | `code/results/*SMOKETEST*` + `*1epoch*` (4 files) | small | Verification scratch runs, STATUS says "do NOT cite". Not real results. |
| C | `__pycache__/`, `.pytest_cache/` (5 dirs) | small | Regenerated automatically on next run. Already gitignored. |

**Not recommended to delete:** both `.venv*` and `code/data/` are large but are your
working environment + dataset (already gitignored, not in the repo). `.venv` (py3.14, no
torch) _might_ be redundant if `.venv312` can run the sim WPs + tests — but verifying that
isn't worth the risk this close to the deadline. Leave them.

## 9. How to run
```bash
cd code
# torch work (detector, gate, ONNX):
.venv312/Scripts/python.exe -W ignore scripts/<wpX>.py
# sim WPs + tests:
.venv/Scripts/python.exe -m pytest tests -q
```

## 10. Known stale references (fix when touching docs)
`docs/STATUS.md` still links to `../code/results/wpN_report.md` for several WPs — those
`.md` files were moved into `reports/` and renamed `01–21`. The links are dead; the content
is in `reports/`. Low priority, but don't chase the old paths.

---

## 11. DEMO readiness — criteria to be "ready" (demo, NOT prototype)

**Framing (read first).** A *prototype* proves the pipeline runs on real satellite/edge
hardware — we can't afford that. A *demo* proves the **idea** works, on real data, with
honest numbers + the deliverable docs. The accepted paper is a methodology/measurement
paper (accepted with zero flown numbers), so a demo is the **expected** form, not a
shortcut. The only place "prototype" leaks in is the word **"onboard"**: every timing is a
laptop RTX 5060, so onboard execution is presented as **TARGET** (mapped to a published
processor spec [LIT]), never claimed as measured. That one honest framing is the whole
demo-vs-prototype bridge.

### Status of the criteria

| # | Criterion | Status | What's left |
|---|-----------|--------|-------------|
| **Science — the paper's contract (Phase 3)** |
| 1 | B0–B4 campaign run end-to-end | ✅ done (report 21, SIM) | confirm nothing regressed |
| 2 | SAHI + global-coord fusion (B2) | ✅ done (reports 15–16, synthetic swaths) | — |
| 3 | Per-stage energy model + ES ratio | ✅ done (report 17) | — |
| 4 | Inter-satellite relay / path choice (B4) | ✅ done (reports 19–21, SIM) | real ISL range is TARGET |
| 5 | Ablations the paper lists | 🟡 mostly done | confirm the few missing |
| 6 | Fold unreported work into paper §IV-B | ❌ open | writing task (scheduler, dark vessels, bounds) |
| **The demo artifact (what you present)** |
| 7 | One end-to-end run: raw→detect→encode→schedule→ground, with the 557× landing | ✅ done → `demo/` | `python demo/summary.py` (instant, no GPU) or `bash demo/run_demo.sh` (live); runbook `demo/DEMO.md` |
| 8 | Headline figures | ✅ exist in `code/results/` | — |
| **Deliverable documents** |
| 9 | Phase-3 results write-up / presentation vs the accepted paper | 🟡 **draft done** → `paper/PHASE3_RESULTS.md` | measured counterpart to the accepted paper (which stays FIXED), section-for-section with labels; map in `demo/PAPER_COVERAGE.md`. Next: team review, then render to LaTeX/slides |
| 10 | 2-min video | 🟡 script done (`paper/video_script.md`) | record + edit |
| 11 | Poster / presentation | 🟡 `presentation.tex` + guide exist | refresh with current numbers |
| **Credibility** |
| 12 | Tests green | ✅ 86 pass | re-run once post-cleanup |
| 13 | Purge superseded numbers everywhere | ❌ open, MUST before submit | kill 442×/511×→557×, 97.1%→0.685, +16.7→+6.3, 2.7×→2.26×, "INT8 lossless", "q40 free" (list in `docs/START_HERE.md §7`) |
| 14 | Onboard-feasibility = TARGET, mapped to a real processor [LIT] | ✅ done → `reports/22` | laptop compute+energy → Φ-sat-2's Myriad 2 (our baseline already runs onboard CNNs); duty-cycle + power-free ES_proc argument; prototype boundary stated |
| 15 | CI / multi-station / bursty arrivals | ❌ open | **skip for a demo** — nice-to-have only |

### Do these, in order (≈9 days, 4 people — split in `START_HERE §9`)
1. ~~Write the Phase-3 results/presentation~~ ✅ **draft done** → `paper/PHASE3_RESULTS.md`
   (review as a team, then render to LaTeX/slides). The accepted paper itself stays FIXED.
2. **Purge superseded numbers** from every doc/figure/slide (#13) — cheap, mandatory.
3. ~~Assemble the single end-to-end demo run~~ ✅ **done** — see `demo/` (`summary.py`, `run_demo.sh`, `DEMO.md`).
4. **Record + edit the video** (script is ready) (#10).
5. ~~Write the onboard = TARGET feasibility section~~ ✅ **done** — `reports/22-onboard-feasibility.md` (cite it in the paper + a demo slide).
6. Refresh the poster/presentation (#11); re-run tests + `rerun_all.sh` to confirm green (#12).

### ⚠ The 3 things judges will probe (defend these or lose points)
- **Recall without a cloud number** — always attach the assumed cloud fraction (0.685 @ 15%; band 0.40–0.82).
- **"Runs onboard"** — it's laptop-measured; say so, present onboard as TARGET.
- **SAHI** — report 13 shows a cut ship is usually still detected, so argue selective slicing honestly; don't quietly drop it.

_Note: I could not find an official Phase-3 / final grading rubric in the repo. The criteria
above are inferred from the accepted paper + `START_HERE`. If you have the official rubric,
share it and I'll map these to it exactly._
