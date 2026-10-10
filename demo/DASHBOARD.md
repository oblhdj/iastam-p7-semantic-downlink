# Demo dashboard — runbook

One interface, `demo/dashboard.py` (Streamlit): one image through B0–B4, live on a laptop CPU.
It needs no GPU, no torch and no dataset. If the live pipeline cannot run, the same page shows
pre-generated results from `demo/fallback_assets/`.

Canonical numbers are not written in the page or in this file. The page reads them from
`code/results/` when it starts; the authoritative statements are in `README.md` and
`paper/PHASE3_RESULTS.md`.

## Run it

```bash
python -m pip install -r demo/requirements-dashboard.txt   # once: quickstart deps + streamlit
streamlit run demo/dashboard.py
```

First load takes a few seconds (model load, one run on the bundled 3×3 swath).

| Path | Command | Needs |
|---|---|---|
| Dashboard, live | `streamlit run demo/dashboard.py` | `requirements-dashboard.txt`; the tracked `demo/quickstart/model/best.onnx` |
| Dashboard, static fallback | same command; it switches by itself when the pipeline cannot run. Rehearse it with `P7_DEMO_FORCE_FALLBACK=1 streamlit run demo/dashboard.py` | streamlit only |
| Terminal quickstart | `python demo/quickstart/run_demo.py` | `demo/quickstart/requirements.txt` |
| Summary of committed results | `python demo/summary.py` | any Python |
| Full demo on the real split | `bash demo/run_demo.sh` (`--full` for all 5,320 tiles) | `code/.venv312` (torch, GPU), the Airbus split, `gate.pt`, `best.pt` |

## What runs live, and what does not

Live, on the image in front of you (bundled sample or upload):

- input validation, dimensions, file and raw size
- the pre-filter's context per 768 px tile
- plain YOLO (one call on the whole image) and SAHI + fusion, each timed
- B0–B4: the real packet stream of each mode, its ground decode, payload and overhead bytes
- the simulated links: direct, policy-chosen and relay-only routes; latency against capture time
- per-image energy and latency budgets
- accuracy, only if you upload a YOLO label file with the image

Read from `code/results/`, never recomputed by the page: the data-reduction headline with its
measured and modeled shares, B1/B2/B3 recall on the same scenes and the two reference rows, the
simulated-day B0–B4 table, the six-day steady state, the sensitivity count, energy per day
(default and the labelled SAHI variant). Reproducing those needs the dataset, the weights and the
GPU environment (`bash demo/run_demo.sh --full`, and the `wp` scripts behind each file).

## Labels on the page

| Label | Meaning |
|---|---|
| SYNTH | computed on a bundled synthetic stand-in image: the chain runs, it is not a measurement |
| MEASURED (bytes) | the length of the serialized packets of this image |
| MEASURED (times) | wall clock on this machine: a laptop, not flight hardware |
| SIM | propagated orbits, geometric windows, link-budget rates |
| ESTIMATE | an assumed power × a measured or simulated time; no power was measured |
| REAL / SIM-over-REAL | as in the repo's results files |

## Static fallback

`demo/fallback_assets/` holds four pre-generated samples (the swath; a ships, a coast and a cloud
tile) from a live run at the default settings, with the commit, model fingerprint and the canon of
that day in `manifest.json`.

```bash
python demo/make_fallback_assets.py --check   # verify: loads, self-consistent, canon unchanged
python demo/make_fallback_assets.py           # regenerate (needs the live pipeline once)
```

Regenerate after any change to `code/results/` or to the encoder; `--check` and the test suite
both fail on a stale set. Uploads and the settings sliders need the live pipeline.

## Before a presentation

- [ ] `python demo/make_fallback_assets.py --check` prints `fallback assets OK`.
- [ ] `streamlit run demo/dashboard.py` opens; the sidebar's *Environment check* is all green.
- [ ] `P7_DEMO_FORCE_FALLBACK=1 streamlit run demo/dashboard.py` opens on the static results.
- [ ] No network is needed once the packages are installed.

## Files

| File | Role |
|---|---|
| `demo/dashboard.py` | the page (sections A–H) |
| `demo/demo_data.py` | standard library only: canon reader, environment report, fallback loader, exports |
| `demo/demo_pipeline.py` | the live run, through `sat7.campaign`, `sat7.semantic`, `sat7.comms` |
| `demo/make_fallback_assets.py` | writes and checks `demo/fallback_assets/` |
| `demo/requirements-dashboard.txt` | quickstart requirements + streamlit |
| `code/tests/test_dashboard.py` | 17 tests: canon, pipeline, routes, bad inputs, fallback, the page itself |
