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
python demo/launch.py                                      # checks the environment, then starts the page
```

`python demo/launch.py` runs `python -m streamlit run demo/dashboard.py` after checking that what it
needs is installed; `streamlit run demo/dashboard.py` still works. The launcher behaves the same in
Git Bash, PowerShell and cmd, and has one flag per path: `--fallback`, `--quickstart`, `--summary`,
`--gpu`, `--check` (start nothing, report the environment), `--install`.

First load takes a few seconds (model load, one run on the bundled 3×3 swath). The browser opens
on `http://localhost:8501` by itself.

`demo/.streamlit/config.toml` (read by Streamlit because it sits next to the script) does three
things for a first run on a new machine: it skips Streamlit's "Email:" prompt, which otherwise
holds the terminal until someone presses Enter; it listens on this machine only, so Windows shows
no firewall dialog; and it sends no usage statistics. The page makes no request outside
`localhost`.

| Path | Command | Needs |
|---|---|---|
| Dashboard, live | `python demo/launch.py` | `requirements-dashboard.txt`; the tracked `demo/quickstart/model/best.onnx` |
| Dashboard, static fallback | same command; it switches by itself when the pipeline cannot run. Rehearse it with `python demo/launch.py --fallback` | streamlit only |
| Terminal quickstart | `python demo/launch.py --quickstart` | `demo/quickstart/requirements.txt` |
| Summary of committed results | `python demo/launch.py --summary` | any Python |
| Detector in the loop on the real split | `python demo/launch.py --gpu` (`--full` for all 5,320 tiles) | `code/.venv312` (torch, GPU), the Airbus split, `gate.pt`, `best.pt` |

## What runs live, and what does not

Live, on the image in front of you (bundled sample or upload):

- input validation, dimensions, file and raw size
- the pre-filter's context per 768 px tile
- plain YOLO (one call on the whole image) and SAHI + fusion, each timed
- B0–B4: the real packet stream of each mode, its ground decode, payload and overhead bytes
- the simulated links: direct, policy-chosen and relay-only routes; latency against capture time
- per-image energy and latency budgets
- accuracy, only if you upload a YOLO label file with the image. The label file carries the
  image's name (`00113a75c.jpg` with `00113a75c.txt`); an empty file means the tile has no ships;
  the page warns when the two names differ, because labels left over from the previous image
  would score the new one against the wrong ships

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

- [ ] In the sidebar's *Environment check*, the last line reads `Windows power throttling — opted
      out`. Windows slows down the processes of an app that is in the background or minimised
      (power throttling, which Windows calls EcoQoS), and the terminal that starts the demo is in
      the background whenever the browser is in front. The demo processes ask Windows not to do
      that to them, at start-up (`demo/host_power.py`). See *Windows power throttling* below.
- [ ] `python demo/make_fallback_assets.py --check` prints `fallback assets OK`.
- [ ] `streamlit run demo/dashboard.py` opens; in the sidebar's *Environment check* nothing is ❌.
      (Two ➖ rows, the Airbus split and the GPU environment, are normal on a clone: only the
      full demo uses them.)
- [ ] `python demo/launch.py --fallback` opens on the static results. (It sets
      `P7_DEMO_FORCE_FALLBACK=1`, which is written differently in every shell.)
- [ ] No network is needed once the packages are installed. Installing them does need it:
      a few minutes, about 500 MB on disk.

## Windows power throttling

What happened on the development laptop on 10 Oct 2026, with the terminal quickstart:

| State of the process | Detector | Quickstart |
|---|---|---|
| Normal | about 30 ms per call | about 2 s |
| The slow period first observed (the app that launched it was minimised or hidden; on battery) | about 700 ms per call | 22 s |
| Throttle forced on a test process, on battery and on mains alike | about 800 ms per call | 27 s |
| Throttle forced on, with the opt-out active | 27 ms per call | 1.6 s |

The slow period was not caused by the battery or the power plan as such: later, still on battery,
the speed was normal, and forced, the throttle is as slow on mains power. What fits is Windows'
treatment of background processes: forcing that state reproduces the slowdown, and the slow period
coincided with the launching app being minimised. That Windows was throttling the process during
that first period is an inference, not a measurement. `demo/host_power.py` opts each demo
process out of it (one attribute of that process; no system setting changes), and the dashboard,
the quickstart and `make_fallback_assets.py` all call it at start-up. `P7_DEMO_ALLOW_THROTTLING=1`
switches the opt-out off, which is how the table's third row was measured against the fourth.

One thing was not reproduced: Windows applying the throttle by itself, which needs the launching
window in the background. The opt-out is the documented way to exclude a process from it and was
verified against the forced state. A one-minute check on the demo laptop settles it: on battery,
start the dashboard, put the terminal behind the browser, load the bundled swath, and read the
SAHI *Time* tile in section C. About half a second is right; several seconds means the throttle
got through.

## Files

| File | Role |
|---|---|
| `demo/dashboard.py` | the page (sections A–H) |
| `demo/demo_data.py` | standard library only: canon reader, environment report, fallback loader, exports |
| `demo/demo_pipeline.py` | the live run, through `sat7.campaign`, `sat7.semantic`, `sat7.comms` |
| `demo/make_fallback_assets.py` | writes and checks `demo/fallback_assets/` |
| `demo/launch.py` | one command per path, with the environment checked first |
| `demo/LIVE_DEMO_SCRIPT.md` | where the demo sits in a presentation, and the five-minute script |
| `demo/requirements-dashboard.txt` | quickstart requirements + streamlit |
| `demo/host_power.py` | standard library only: opts the demo process out of Windows power throttling |
| `demo/.streamlit/config.toml` | first-run settings: no email prompt, localhost only, no usage statistics |
| `code/tests/test_dashboard.py` | 23 tests: canon, pipeline, routes, bad inputs, uploads and label files, fallback, the page itself |
| `code/tests/test_host_power.py` | 5 tests: the opt-out is requested, overrides a throttled state, and every entry point calls it |
