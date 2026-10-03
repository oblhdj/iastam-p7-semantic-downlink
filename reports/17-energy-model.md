# 17 — The per-stage energy model, and where the joules actually go

> ### ⚠ REFRESHED 3 Oct — now reads every figure live from its source file (+ a sanity gate)
> The model had **hardcoded** its stage times and MB/day. Per the project rule ("the `.csv`/`.json`
> files always win", and these numbers have moved before — MB/day went 117.9 → 108.0 after
> `conf_high` 0.670), `wp17_energy_model.py` now reads each one live: detector GPU from
> `wp1_metrics.json`, detector CPU from `wp1_export.json`, the learned gate from `wp3_gate.json`,
> the classic pre-filter from `wp11_integration.json`, and MB/day from `wp6_real_table.csv`. A
> **sanity gate** now runs before any ES is reported (same pattern as [report 16](16-swath-policies.md)'s
> whole-policy gate): it asserts the data-reduction factor it derives (`D_raw/D_sent`) reproduces
> `wp6_real_table.csv`'s own `data_reduction_x` (557×, rel err 0), and that the effective link rate
> cannot exceed the model's zenith peak (2.53 ≤ 6.40 Mbps). `wp17_energy_model.json` is authoritative
> over the tables below. **Gate scope, stated plainly:** both checks validate the SIM *downlink* side
> (`ES_total`, `E_comm`); the *processing* energy (`ES_proc`) has **no independent cross-check** — it
> is definitional arithmetic over the live-read stage times, so its trust rests on the source-file
> provenance (recorded per-figure in the JSON), not on a second measurement.
>
> **What changed, old (hardcoded) → new (live), and why:** the two **GPU** figures had been taken
> from `wp11_integration.json`'s per-tile *stage* timing — **learned gate 0.154 → 0.052 ms**
> (`wp11`'s 0.154 bundles dataloader/harness overhead that this model already counts in the
> preprocess stage; `wp3_gate.json`'s 0.052 is the pure forward pass) and **detector GPU
> 11.1 → 10.71 ms** (`wp1_metrics.json`). Consequently **ES_proc (GPU budget) 68.8% → 69.8%**
> (E_base 2.276 → 2.252 J, E_prop 0.709 → 0.680 J; the GPU *time* ratio 67.4 → 11.3 ms (6.0×)
> becomes 67.0 → 10.8 ms (6.2×), i.e. ~3.3× cheaper in energy), and the §5 `ES_proc_gpu` sweep
> shifts by ~1 point.
>
> **What did NOT change — the headlines hold:** the default **CPU onboard build ES_proc = 57.6%**
> (a power-free time ratio; its stage times — classic 56.3→56.28, gate_cpu 0.94, ONNX 38.6 — did not
> move) and **ES_total ≈ 98.2%** vs a bent pipe (MB/day 108.0→107.95 and 60,124→60,124.3 are
> rounding; the detector cancels in both ratios). **No figure label changed** (times REAL, downlink
> SIM, powers ASSUMPTION). Two figures are **not in any results file** and are now flagged as such
> rather than silently trusted — preprocess/JPEG-decode (0.71 ms, report 03's ~1,400 tiles/s, never
> emitted) and the semantic stage (0.5 ms, ASSUMPTION); both are shared stages that cancel in ES_proc.
> The body below is otherwise unedited.
>
> **Measurement-scope caveat (raised by the energy track, [report 19](19-relay-energy.md) §5).** The
> live-read stage times are **not all measured under the same conditions**: the gate (0.052 ms,
> `wp3_gate.json`) is a **pure-forward, batch-32** figure (tensor resident on the GPU, no I/O or
> pre/post-processing), while the detector (10.71 ms, `wp1_metrics.json`) is **end-to-end
> `model.predict` over a file list** (JPEG decode + letterbox + NMS + match loop, effectively
> batch-1). So the §1/§3 GPU per-stage times mix amortised and unamortised scopes and should be read
> as *indicative*, not like-for-like. **This does not change the ES_proc conclusion:** the gate is
> two-plus orders of magnitude below the 56.3 ms classic pre-filter it *replaces*, so the gating-stage
> saving holds whatever the gate's fair end-to-end cost; and `ES_total` is downlink-volume-driven,
> untouched. A fair end-to-end gate timing (decode + forward + post, batch-1) would tighten the GPU
> absolutes and is the honest next measurement if those absolutes are ever quoted on their own.

Builds the energy model the accepted paper promised and `../docs/START_HERE.md` §5 lists as
unbuilt: `E_proc = Σ Pₖ·Tₖ` across preprocess / tile / detect / fuse / semantic, `E_comm =
P_tx·D_tx/R_tx`, and the energy-saving ratio `ES = 1 − E_prop/E_base`. Script:
`../code/scripts/wp17_energy_model.py`. Data: `../code/results/wp17_energy_model.json`,
`wp17_energy_stages.csv`. No detector is run and no torch is imported — every stage *time* is
reused from an earlier measurement, the downlink is the orbit sim's own output, and this script
does only the arithmetic.

**Labels, stated up front because the honesty of the result depends on them:**
* **Stage times Tₖ — REAL.** Measured in earlier reports, each on a **laptop RTX 5060 + its CPU**.
  START_HERE §5/§7 flags that this is *not flight hardware*; so do we, throughout.
* **Downlink capacity, R_tx, D_tx — SIM.** The SGP4 orbit/link model (`sat7.orbit`, `results/passes.csv`)
  and the scheduler's real-detection result (108.0 MB/day, [report 05](05-scheduler-on-real-detections.md)).
* **Stage powers Pₖ — ASSUMPTION, and swept** (the project rule: every invented constant is swept,
  cf. [report 11](11-sensitivity.md)). The link model carries SNR and data rate but **no transmit
  power**, so `P_tx` is the one genuinely missing quantity — taken as representative of published
  S-band cubesat transmitters (~8–30 W DC while keying) and swept, *not* cited from a datasheet.

## 1. The stages, and the times we already had

| stage | runs on | time/tile | label | source |
|---|---|---|---|---|
| preprocess (JPEG decode) | CPU | 0.71 ms | REAL | ~1,400 tiles/s warm ([report 03](03-confidence-calibration.md)) |
| **gate/pre-filter** — classic CV | CPU | **56.3 ms** | REAL | every tile ([report 12](12-end-to-end-integration.md)) |
| **gate/pre-filter** — learned gate | GPU 0.154 / CPU 0.94 ms | | REAL | 47k-param CNN ([report 12](12-end-to-end-integration.md) / [03](03-confidence-calibration.md)) |
| detect — YOLO 768 px | GPU 11.1 / CPU-ONNX 38.6 ms | | REAL | surviving tiles ([report 12](12-end-to-end-integration.md) / [02](02-detector-and-quantisation.md)) |
| fuse (global-coord + NMS) | CPU | 0.0 ms (default) | — | single-tile frame; B2/SAHI adds it ([report 16](16-swath-policies.md)) |
| semantic (LoD + schedule) | CPU | 0.5 ms | ASSUMPTION | sub-ms, swept |

The two numbers that move the result are the **gating stage** (56.3 ms classic → 0.154/0.94 ms
learned) and the **detector**, which is identical in baseline and proposed, so it cancels in the
processing ratio.

## 2. The powers we did not have

| power | value | range swept | label | basis |
|---|---|---|---|---|
| `P_cpu` | 28 W | 15–45 | ASSUMPTION | mobile CPU package, active |
| `P_gpu` | 60 W | 35–115 | ASSUMPTION | active draw within the **RTX 5060 Laptop TGP envelope 35–115 W** (a real spec bound; the laptop every Tₖ was taken on) |
| `P_tx` | 15 W | 8–30 | ASSUMPTION | representative S-band cubesat transmitter DC while keying; the link model has rate but no power |

None is a datasheet citation, so all are **ASSUMPTION** and all are swept in §5.

## 3. Processing energy — and why the energy saving is smaller than the time saving

Per tile, detector counted once in both pipelines (so the gate's *extra* skipping of empty tiles
is an uncounted additional win — [report 04](04-onboard-gate.md)):

| config | E_base (J/tile) | E_prop (J/tile) | **ES_proc** | time ratio |
|---|---|---|---|---|
| **DEFAULT — gate + ONNX FP32, all CPU** (the onboard build) | 2.691 | 1.141 | **+57.6%** | 94.9 → 39.5 ms (2.4×) |
| GPU budget (report 12: GPU gate + GPU detector) | 2.276 | 0.709 | **+68.8%** | 67.4 → 11.3 ms (6.0×) |

Two things worth saying plainly:

* **For the default onboard build the saving needs no power assumption at all.** Every stage runs
  on the CPU, so `P_cpu` cancels and `ES_proc = 1 − T_prop/T_base = 57.6%` exactly, whatever the CPU
  draws. It is also a **lower bound**: the learned gate additionally drops 40.2% of empty tiles that
  the classic stage keeps, removing detector work this per-tile accounting does not credit.
* **The energy saving is smaller than the headline time saving.** On the GPU budget the gate swap is
  6.0× *faster* but only 3.2× *cheaper in energy* (68.8%), because the stage it removes (classic
  pre-filter, CPU, 28 W) draws less than the stage it keeps (detector, GPU, 60 W). Time and energy
  are not the same lever, and the paper should quote them separately.

## 4. Comms energy — and the thing that is really being bought

Reusing the orbit sim's own day (`results/passes.csv`): **5 passes, 34.2 min contact, 649 MB/day
capacity**, effective rate **2.53 Mbps** averaged over contact (the model's zenith rate is 6.40
Mbps; low-elevation passes pull the average down). The semantic downlink sends **108 MB = 16.6% of
capacity**. A bent pipe sending the raw 60,124 MB would need **93× the daily contact — it does not
just cost more energy, it does not fit.**

Per day at 40,000 tiles:

| quantity | value |
|---|---|
| E_proc, proposed | **45.6 kJ** |
| E_proc, baseline (classic + detector) | 107.6 kJ |
| E_comm, proposed (send 108 MB) | **5.1 kJ** |
| E_comm, bent pipe (send 60,124 MB) | 2,851 kJ |
| **proc : comm ratio, proposed** | **8.9 : 1** |

**Once you stop sending pixels, onboard compute is the dominant energy cost, not the radio** (8.9
to 1). That reframes where energy optimisation belongs for Phase 3: the detector and the gating
stage, not the transmitter. And against a bent pipe the whole system saves

> **ES_total ≈ 98.2%** — spend ~51 kJ/day of compute-plus-radio to avoid ~2,851 kJ/day of raw
> transmission — a saving dominated by the 557× data cut, and underwritten by the fact that the raw
> volume could not have been sent at all.

## 5. Sensitivity — the conclusions survive the invented powers

Each `Pₖ` swept to its range ends (one at a time):

| swept | ES_proc (GPU budget) | ES_total vs bent pipe |
|---|---|---|
| `P_cpu` 15→45 W | 0.546 → 0.776 | 0.990 → 0.973 |
| `P_gpu` 35→115 W | 0.786 → 0.540 | 0.982 (flat) |
| `P_tx` 8→30 W | 0.689 (flat) | 0.968 → 0.990 |

* **ES_total stays 97–99%** across every power assumption — the bent-pipe comparison is driven by
  the data-volume cut, not by any Pₖ.
* **ES_proc for the default CPU build is invariant at 57.6%** (it is a time ratio). Only the GPU
  budget's ES_proc moves, and only with the `P_cpu/P_gpu` ratio — which is exactly the quantity
  flight-hardware selection would pin down. Even at its worst (0.54) the gate still saves >half the
  processing energy.

## 6. Limits

* **All times are laptop RTX 5060, not flight hardware** (START_HERE §5/§7). Absolute joules are a
  laptop figure. The *ratios* (ES_proc, ES_total) are far more robust than the absolutes — ES_proc
  for the default build is power-free, and ES_total is volume-driven — but a real energy budget in
  watt-hours waits on an actual onboard board being chosen.
* **`P_tx` is the weakest number** — invented, because the link model never needed transmit power to
  compute a rate. It only enters E_comm, which is already 9× smaller than E_proc, so the final
  picture barely depends on it (the `P_tx` sweep moves ES_total by ~2 points).
* **Single ground station, one day** — the orbit sim's Sfax day. More stations would raise capacity
  and lower the bent-pipe infeasibility factor, not change the processing side.
* **`tiles/day = 40,000` is an ASSUMPTION** (imager duty cycle); per-day joules scale linearly with
  it, ES ratios do not.
* **`fuse` is 0 and `semantic` is 0.5 ms** for the current single-tile default. When B2/SAHI lands
  ([report 16](16-swath-policies.md)), the detect stage carries SAHI's 1.56× multiplier and fuse
  becomes non-zero — both plug straight into this model, and both raise E_proc, reinforcing §4's
  point that compute, not comms, is where the energy now lives.
