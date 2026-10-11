# Reproducing this repository

What you need, what each level of reproduction costs, and the exact commands. Everything here was
run on the development laptop on 10 Oct 2026 (Windows 11, Git Bash; RTX 5060) unless a line says
otherwise; section 7 lists what was run and what was not.

No canonical number is restated in this file. The results are in `code/results/`, and the
documents that quote them are `README.md` and `paper/PHASE3_RESULTS.md`.

| You want to | You need | Time | Section |
|---|---|---|---|
| see the system run | Python ≥ 3.10, one `pip install` | 2 s after a few minutes of install | 2 |
| check the code | the analysis environment | 15 s | 3, tier 1 |
| regenerate every simulated result | the analysis environment | 8.5 min | 3, tier 2 |
| re-measure the detector | the Airbus split, the weights, the torch environment | minutes to hours | 3, tier 3; 5 |

---

## 1. Environments

Three environments exist on the development laptop. Only the first is needed to check and
re-simulate; only the second to run the demos.

| Environment | Python here | Install | Runs | Does not have |
|---|---|---|---|---|
| **analysis** `code/.venv` | 3.14.7 | `code/requirements.txt` (pinned) | the unit-test tier, every simulated experiment, `demo/summary.py` | a working torch, onnxruntime, streamlit |
| **demo** `code/.venv-demo` | 3.12.14 | `demo/requirements-dashboard.txt` (= the quickstart's requirements + streamlit) | the dashboard, the terminal quickstart, the fallback, the inference and page test tiers | torch, matplotlib |
| **detector** `code/.venv312` | 3.12.14 | torch from the PyTorch index, then `code/requirements-gpu.txt` | training, the learned gate, every script that runs the detector on the Airbus split; also the whole test suite except the three page tests | streamlit |

```bash
# analysis
python -m venv code/.venv
code/.venv/Scripts/python -m pip install -r code/requirements.txt

# demo (any Python >= 3.10; a virtual environment is optional)
python -m pip install -r demo/requirements-dashboard.txt

# detector (Python 3.12; adjust the index URL to your CUDA version)
py -3.12 -m venv code/.venv312
code/.venv312/Scripts/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
code/.venv312/Scripts/python -m pip install -r code/requirements-gpu.txt
```

On Linux or macOS the interpreter is `bin/python` instead of `Scripts/python`.

**Is the two-environment split still accurate?** Yes, as a description of the laptop: `.venv` is
Python 3.14 with no working torch, and `.venv312` is Python 3.12 with torch 2.11 and CUDA 12.8. It
is no longer the whole picture:

* The demo path does not need Python 3.12. From a fresh clone it installed and ran on Python 3.14.6
  and on 3.12 (onnxruntime 1.31.0, streamlit 1.65.0, numpy 2.5.3, opencv-python-headless 5.0.0.93).
* `.venv312` runs the unit tier too, so a machine with only that environment can run everything
  but the page tests.
* Not tested: whether torch now installs on Python 3.14, and whether the simulated experiments
  give byte-identical output under `.venv312` (they do under `.venv`).

`code/requirements-gpu.txt` is new. It records the versions in the working environment; it was not
installed into a clean one.

## 2. Running the demo: one command per path

`demo/launch.py` is standard library only and behaves the same in Git Bash, PowerShell and cmd. It
checks what the chosen path needs and names what is missing, with the command that provides it.

| Path | Command | Needs |
|---|---|---|
| **Full demo**: the dashboard, live on CPU | `python demo/launch.py` | the demo environment |
| **Fallback**: the same page on pre-generated results | `python demo/launch.py --fallback` | streamlit only |
| **Quickstart**: the chain in the terminal | `python demo/launch.py --quickstart` | the quickstart requirements |
| Committed results on one screen | `python demo/launch.py --summary` | any Python |
| Detector in the loop on the real split | `python demo/launch.py --gpu` (add `--full` for all 5,320 test tiles) | bash, `code/.venv312`, the split, `best.pt`, `gate.pt` |
| Environment and fallback check, nothing started | `python demo/launch.py --check` | any Python |

Add `--install` to install the chosen path's requirements into the current interpreter first, and
`--dry-run` to print the command instead of running it. What the launcher runs:

```bash
python -m streamlit run demo/dashboard.py        # full demo; opens http://localhost:8501
python demo/quickstart/run_demo.py               # quickstart
bash demo/run_demo.sh                            # detector in the loop (wp18 + wp11 + summary)
```

The dashboard falls back to `demo/fallback_assets/` by itself when the live pipeline cannot run.
The runbook, the labels and the pre-presentation checklist are in
[`demo/DASHBOARD.md`](../demo/DASHBOARD.md); the five-minute script is
[`demo/LIVE_DEMO_SCRIPT.md`](../demo/LIVE_DEMO_SCRIPT.md).

## 3. Tests and experiments: three tiers

They answer different questions and need different things. Keep them apart when you report what
you ran.

### Tier 1 — unit tests (`pytest`)

*Does the code do what it says?* 294 tests in `code/tests/`, each in one of three groups set in
`code/pytest.ini`:

| Marker | Tests | What they need | Without it |
|---|---|---|---|
| `unit` | 278 | the analysis environment only: no torch, no onnxruntime, no dataset, no GPU | — |
| `inference` | 16 | onnxruntime for 14 of them (on the bundled model and synthetic tiles; 3 of the 14 are the page tests), torch for the 2 gate tests (`gate.pt` for one) | skipped, with the reason |
| `page` | 3 (also `inference`) | streamlit | skipped |

Inference is optional by construction: a test that runs the detector skips itself when the package
or the weights are absent, so the basic suite passes anywhere.

```bash
cd code
.venv/Scripts/python -m pytest -m unit -q        # 278 passed            (analysis environment)
.venv/Scripts/python -m pytest -q                # 278 passed, 16 skipped
.venv312/Scripts/python -m pytest -q             # 291 passed, 3 skipped (the page tests)
.venv-demo/Scripts/python -m pytest -q tests/test_dashboard.py tests/test_host_power.py tests/test_quickstart.py tests/test_perception.py tests/test_detect_image.py
```

The last line is the demo environment, which has no matplotlib and so cannot import every test
file; those five files hold all the `inference` and `page` tests except the two gate tests.

The unit tier includes the checks of the committed result files against the documents
(`tests/test_validation.py`, `tests/test_docs.py`): they read results, they do not recompute them.

### Tier 2 — simulated experiments

*Do the committed results regenerate?* These scripts replay the committed detection catalogues
(`code/results/wp1_predictions.csv`, `wp6_tiles.csv`, `wp6_ships.csv`, …) through the encoder,
scheduler, orbit, energy and relay models. They need the analysis environment and nothing else.

```bash
cd code && bash rerun_simulations.sh             # 19 runs of 16 scripts, about 8.5 minutes
```

Run from a fresh clone with no dataset and no weights on 10 Oct 2026: every run exited 0, and
afterwards `git status` showed **no tracked file changed** — each committed CSV, JSON and figure
was regenerated byte for byte. The script ends by repeating that check.

**The paper's Table I levels (pure P0–P3).** The headline results use the level-of-detail ladder.
The same day and the same campaign under the paper's four levels are two tagged runs, which write
`*_paper` files next to the canonical ones and never over them (`--semantic priority` refuses to
run without `--tag`):

```bash
cd code
.venv/Scripts/python scripts/wp6_simulate_real.py --semantic priority --tag paper        # wp6_real_*_paper.csv / .png
.venv312/Scripts/python scripts/wp18_campaign_runner.py --semantic priority --tag paper  # wp18_campaign_paper.json
```

The first is a simulation like the others (analysis environment, 53 s; rerun on 11 Oct 2026 with
no tracked file changed). The second belongs to tier 3: it loads the detector, so it needs
`code/.venv312`, `best.pt` and the split. Neither is in `rerun_simulations.sh`; both are in
`rerun_all.sh`. They are read in `paper/PHASE3_RESULTS.md` §2b.

**Energy.** `scripts/wp17_energy_model.py` (in the script above) writes three files in one run:
`wp17_energy_model.json`, `wp17_energy_stages.csv` and `wp17_energy_edge.json`. The last is the
edge-class grid: the same model re-priced over assumed processor powers and slowdowns. There is no
separate command for it. It is read in `paper/PHASE3_RESULTS.md` §6b and §6c.

Two simulations are not in that script because they serialize real tiles and so need the Airbus
split (CPU only, no detector): `wp25_semantic_packets.py` (3–6 min) and
`wp27_comms_direct_vs_relay.py` (125 s recorded). Without the split they stop with
`FileNotFoundError`. `wp28_coast_tile_model.py --measure` re-encodes the coastal tiles; without
the flag it reads the committed size table.

### Tier 3 — full detector evaluation

*Do the measurements regenerate?* These run the detector, or the gate, on real tiles. They need
the split (section 5) and the weights (section 6); the torch ones need `code/.venv312`.

| Script | Produces | Needs beyond the split | Time recorded |
|---|---|---|---|
| `wp1_train_detector.py` | the detector and `wp1_metrics.json` | torch, a GPU (trained on a Kaggle T4, `kaggle/wp1_train_kaggle.ipynb`) | — |
| `wp1_export_int8.py` | ONNX FP32 / INT8 and `wp1_export.json` | torch, `best.pt` | — |
| `wp2_predict_split.py`, `wp2_calibrate.py` | calibration | torch, `best.pt` | — |
| `wp3_train_gate.py`, `wp3_score_tiles.py` | the learned gate | torch | — |
| `wp6_build_catalogue.py` | the catalogues tier 2 replays | `wp1_predictions.csv` | — |
| `wp11_integration_demo.py --tiles 5320` | `wp11_integration.json` | torch, `best.pt`, `gate.pt` | — |
| `wp16_swath_policies.py`, `wp12*`, `wp13*` | tiling and seam studies | torch, `best.pt`, the swaths | — |
| `wp18_campaign_runner.py` | `wp18_campaign.json` | torch, `best.pt` | — |
| `wp18_campaign_runner.py --semantic priority --tag paper` | `wp18_campaign_paper.json` (B3 / B4 under pure P0–P3) | torch, `best.pt` | — |
| `wp24_detection_eval.py --live-b1` | `wp24_detection_eval.json` | onnxruntime, `best.onnx`, the swaths | 316 s on CPU |
| `wp26_b0_b4.py` | `wp26_b0_b4.json` (B0–B4 on the same scenes) | onnxruntime, `best.onnx` | 208 s |

`cd code && bash rerun_all.sh` runs every dependent script of tiers 2 and 3 in order, with the
flags that produced the committed results; it is the record of those flags. `bash demo/run_demo.sh`
is the short live version (a 400-tile sample, 45 s when the audit ran it; `--full` for all 5,320 tiles) and writes
to `demo/_live/`, never to `code/results/`.

The swaths are rebuilt from the split with
`python scripts/wp15_build_swaths.py --regenerate results/wp15_manifest.json`.

## 4. Configuration

Every setting, with its default and the three that every experiment overrides, is in
[`docs/CONFIGURATION.md`](CONFIGURATION.md). The tables there are generated from the code:

```bash
python code/scripts/print_config.py --check docs/CONFIGURATION.md
```

## 5. The dataset — you must obtain it yourself

The imagery is the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)
dataset. It is not in this repository and cannot be.

1. Create a Kaggle account and accept the competition rules on the competition page.
2. Download the competition archive (about 30 GB), from the page or with
   `kaggle competitions download -c airbus-ship-detection`.
3. Build the leakage-free split, reading straight from the zip (about 35 minutes; about 8 GB on
   disk):

```bash
cd code && .venv/Scripts/python scripts/wp0_build_dataset.py --zip /path/to/airbus-ship-detection.zip --all-ships --workers 12
```

It writes `code/data/yolo_ships/` (53,195 tiles in train / val / test, YOLO labels, `split.csv`,
`duplicates.csv`). `code/data/` is git-ignored.

**Licence constraint.** The competition rules allow non-commercial use only (§7A) and forbid
publishing, redistributing or otherwise providing the Competition Data to anyone who has not
accepted the rules (§7B). In practice, for this repository:

* no real tile, crop, thumbnail or gallery cut from a real tile is committed; three such galleries
  are git-ignored (they remain in earlier commits, a decision still open);
* the bundled demo tiles are procedurally rendered stand-ins, and everything computed on them is
  labelled SYNTH;
* live outputs on real tiles go to `demo/_live/` and `demo/quickstart/_out/`, both git-ignored;
* whether a real tile may appear in a public video or slide is a question for the rules and is
  not settled here. `paper/video_script.md` plans one (shots 1 and 4); see the cross-check notes in
  `audit/AUDIT_UPDATE_2026-10-10.md`.

## 6. The weights

| File | In git | How to obtain it |
|---|---|---|
| `demo/quickstart/model/best.onnx` (12.3 MB, sha256 `5463fc72…22d4fd9`) | yes | nothing to do; it is the trained detector exported to ONNX FP32 |
| `code/runs/ships/weights/best.pt` | no (`code/runs/` is ignored) | train: `scripts/wp1_train_detector.py --epochs 12 --imgsz 768 --batch 8`, or the Kaggle notebook; ultralytics downloads the YOLOv8n starting weights by itself |
| `code/runs/ships/weights/best.onnx` | no | `scripts/wp1_export_int8.py` from `best.pt`; byte-identical to the tracked copy |
| `code/models/gate.pt` | no | `scripts/wp3_train_gate.py --epochs 4 --size 128` (about 2 minutes on the split) |

A retrained detector will not reproduce the committed detection numbers to the digit; the committed
catalogues are what tier 2 reproduces exactly.

**Licence of the weights.** They are fine-tuned from Ultralytics YOLOv8n, whose export metadata
carries the AGPL-3.0 licence string, and they were trained on Airbus Competition Data under the
non-commercial terms above. The repository's MIT licence covers the code, not these weights.

## 7. What was run for this document, and what was not

Run on 10 Oct 2026:

* the three test tiers in `.venv` and `.venv312`, and the five demo-side test files in `.venv-demo`
  (counts in section 3);
* `rerun_simulations.sh`, from a fresh clone without dataset or weights: 19 runs, exit 0, no
  tracked file changed;
* the dashboard and the quickstart from a fresh clone, in new environments on Python 3.14.6 and
  3.12, with a fresh user profile and nobody at the keyboard;
* every mode of `demo/launch.py` as a dry run; `--check`, `--quickstart` and `--summary` for real;
  the missing-dependency message from an environment without streamlit.

Run on 11 Oct 2026, for the pure P0–P3 and energy paragraphs of section 3:

* `wp6_simulate_real.py --semantic priority --tag paper` and `wp17_energy_model.py`, in a clean
  checkout with `.venv`: exit 0, no tracked file changed;
* the test suite in `.venv` (counts in section 3, unchanged).

Not run:

* `wp18_campaign_runner.py --semantic priority --tag paper` (tier 3); the command is the one
  recorded in `rerun_all.sh`;

* tier 3. No detector-side script was rerun for this document; their times above are the ones
  recorded in the result files, and the commands are from the scripts' own docstrings;
* creating the analysis and detector environments from scratch (both already exist here), and so
  a clean install from `code/requirements.txt` or `code/requirements-gpu.txt`;
* the dataset download and `wp0_build_dataset.py` (the 35 minutes is its recorded run time);
* anything on Linux or macOS, or on a second machine.
