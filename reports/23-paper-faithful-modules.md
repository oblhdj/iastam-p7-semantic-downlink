# 23 — Paper-faithful modules: first-class SAHI, P0–P3, the joint program, calibratable energy

Phase-3 housekeeping against the accepted paper. `PAPER_COVERAGE.md` named four places where the
code expressed the paper's intent through a *substitute* rather than the paper's own construct:

* §4 — the joint objective `min αE + βD + γT s.t. A≥A_min, I≥I_min` was **approximated** by
  value-per-byte + the relay's λ path-choice, never solved as one program.
* §6 — the ablations *without tile overlap* and *without detection fusion* were not isolated.
* §8.1 — SAHI ran only on Airbus-stitched swaths; no loader for the paper's recommended large-scene
  sets (xView / VisDrone / DOTA / HRSID).
* Table I — the P0–P3 priority levels existed on paper; the code shipped the LoD ladder instead.

This WP adds the paper's own constructs as first-class, testable `sat7` modules, **without
displacing** the measured LoD pipeline. Everything here is infrastructure + one head-to-head
comparison; **no new campaign headline is minted**, and the LoD path still reproduces the 108 MB /
557× / 0.680 headline exactly (checked below).

> Labels unchanged (START_HERE §8). New code emits no REAL number of its own: the slicing/fusion and
> the P0–P3 sizes are the existing REAL/SIM quantities reused; the joint program's `D` and `T` are
> SIM, its `E` is **TARGET** until calibrated powers are injected; all new thresholds and weights are
> **ASSUMPTION**, swept.

## What was added

| module | paper construct | role |
|---|---|---|
| `sat7/perception.py` | B1 vs B2 as a *mode* | `detect_image(img, detect_fn, PerceptionConfig(mode=…))` unifies whole-image and SAHI; `overlap=0` and `fuse=False` are the two isolated ablations; carries the YOLO `detect_fn` |
| `sat7/priority.py` | **Table I P0–P3** | `encode_priority` emits ordinary scheduler `Item`s, so the *same* scheduler/`simulate`/`metrics`/optimum run over it; `encode_semantic(mode="lod"\|"priority")` is the switch |
| `sat7/joint.py` | **eq 20**, solved as one program | `solve()` over per-object levels: unconstrained optimum (a true lower bound) + greedy constraint repair, with a validity bracket like `optimum.py`; `pareto()` sweeps (α,β,γ) |
| `sat7/energy.py` | **§V eqs 8–19**, calibratable | per-stage `E_proc`/`E_comm`/`E_relay`/ES, with `from_measurements(P_k, T_k, R_tx)` to inject real edge-hardware numbers; supplies `relay.route`'s energy callables |
| `sat7/orbit.py` (+) | multi-ground-station | `find_passes_multi(sat, stations, …)` merges a station network onto one timeline; `max_gap_h` reports the latency-dominating gap |
| `sat7/datasets.py` | §VIII datasets | DOTA / YOLO / COCO(HRSID) annotation parsers + `evaluate_scenes` running the SAHI path by size bucket |
| `sat7/relay.py` (+) | deadline-aware `min(J)` | `route_item(…, deadline_s=…)`: disqualify paths that miss the deadline, then `min(J)`; flag `deadline_missed` if none makes it |

43 new tests (`tests/test_{perception,priority,joint,energy,datasets,multistation,relay_deadline}.py`);
full suite **137 passed, 2 skipped**. Each module has a `python -m sat7.<name>` self-check.

## The switch, in one line

```python
from sat7.priority import encode_semantic
items_lod      = encode_semantic(wl, mode="lod",      lod=lod)            # the repo ladder
items_priority = encode_semantic(wl, mode="priority", lod=lod, cfg=PriorityConfig())  # paper P0–P3
```

Both feed the identical `simulate(...)` + `metrics(...)`. The campaign runner takes it too:
`wp18_campaign_runner.py --semantic priority` (the B3 traceability gate, which pins to the LoD
headline CSV, runs for `lod` only).

## Head-to-head (SIM-over-REAL, 40k tiles, link_share 0.25) — `wp23_semantic_compare.py`

```
scheme                    recall   dark  MB sent    val  lat_med_h
LoD ladder (repo)         0.680    0.679   108.5   0.727     4.49     <- reproduces the headline
P0–P3 (paper Table I)     0.617    0.626    49.9   0.903     4.50
P0–P3 level histogram (detections): P0 4620 · P1 5927 · P2 4450 · P3 2656
```

Read it honestly: **P0–P3 is the leaner, more selective scheme** — it sends 54% fewer bytes at a
higher value density, but **−6.3 points of recall**, because P0 discards sub-threshold detections and
the scheme has no blanket-thumbnail safety net (that net is a LoD-specific recovery path; P0–P3
replaces it with P3 context only on *uncertain* real detections). Which is "better" depends on the
operating point the paper's constraints pin — which is exactly what the joint program decides. This
is a trade, not a win, and is reported as one (the project rule).

## The joint program, actually solved

`sat7.joint.solve` minimises `αE + βD + γT` over each detected object's level choice, subject to
`Accuracy ≥ A_min` (weighted fraction reported at all) and `InformationPreservation ≥ I_min`
(richness retained vs all-P3). It is bracketed, like the scheduler's optimality gap:

* `J_lower_bound` — the per-object unconstrained argmin. Constraints only force levels **up**, so no
  feasible assignment costs less: a true lower bound. `JointResult.valid` asserts `J ≥ J_lower`.
* the repair is two heap phases (accuracy, then information), O(n log n) — 13k objects solve in <2 s.

```python
from sat7.joint import build_objects, solve, JointWeights, JointConstraints
objs = build_objects(wl, lod)                      # one object per detected ship
r = solve(objs, JointWeights(alpha=1, beta=1, gamma=1), JointConstraints(a_min=0.9, i_min=0.5))
r.J, r.accuracy, r.info_preservation, r.feasible, r.gap   # gap = price the constraints impose
```

Units are stated and commensurable: **E in joules, D in MB, T in hours** (set them so a typical item
contributes order-1 to each term). One modelling note: the joint `T` term is per-item *transmission
time* (`t_proc + bytes/R_tx`), a documented lower bound on latency; the **end-to-end delivered
latency is still owned by the scheduler + relay** (reports 20–21), which model queueing and windows.
So use `joint` for the encoding-policy trade, and `simulate`/`route` for delivered latency.

## Calibrating the energy model with real hardware

`wp17` hardcoded assumption powers. `sat7.energy` injects measured ones through one constructor —
the §7 "clean interface for injecting measured P_k and T_k" ask:

```python
from sat7.energy import EnergyModel
em = EnergyModel.from_measurements(
    powers_W={"cpu": 5.1, "gpu": 9.0, "tx": 11.3},        # e.g. Jetson Orin Nano + S-band radio
    stage_times_ms={"detect_cpu_onnx": 95.0, "gate_cpu": 1.2},   # partial calibration is fine
    r_tx_bps=1.0e6, provenance={"powers_W": "REAL (bench-measured, Orin Nano)"})
em.e_proc_per_tile_J("cpu_onnx"); em.e_comm_direct_J(d); em.e_comm_relay_J(d)   # eqs 8–19
route(plan, links, lam_E=…, lam_T=…,                     # calibrated model drops into the relay
      direct_energy=em.direct_energy_fn(), relay_energy=em.relay_energy_fn())
```

`EnergyModel.from_results(results_dir)` reads the REAL measured stage times live from the WP files
(the wp17 provenance), keeping the assumption powers until real ones are injected. The all-CPU
`ES_proc` stays power-independent (wp17's robustness claim) — a unit test pins it.

## How to evaluate the new components against B0–B4

1. **B2 / SAHI first-class + ablations.** Run `evaluate_scenes` (or `wp18 --b2-window {512,768}`) with
   `PerceptionConfig(mode="sahi", overlap∈{0.2,0.0}, fuse∈{True,False})`. Report four rows — SAHI,
   no-overlap, no-fusion, and `mode="whole"` (no-SAHI) — by size bucket. This closes the two 🟡
   ablations in `PAPER_COVERAGE.md §6` as isolated rows rather than "implicit in B1 vs B2".
2. **B3 / P0–P3 vs LoD.** `wp23_semantic_compare.py` is the head-to-head at the canonical load; sweep
   `--day-tiles` and `--link-share` to trace recall-vs-bytes for both schemes on one plot, beside the
   LoD curve from report 06. State the cloud caveat on every recall.
3. **Joint program trade-off curve.** `joint.pareto(objs, constraints, grid=…)` over a fine (α,β,γ)
   grid gives the `E`–`D`–`T` surface the paper's eqs 32–35 ask for, each point annotated feasible or
   not. Compare the joint optimum's byte total to what the online value-greedy scheduler actually
   sends — the gap is the price of solving online rather than with full foresight (cf. report 09).
4. **Large-scene B2.** Point `datasets.load_dota_dir` / `load_hrsid` at a downloaded DOTA or HRSID
   split and run `evaluate_scenes` with the trained `detect_fn`; this is the large-scene B2 strengthen
   flagged in `PAPER_COVERAGE.md §8.1` (and a SAR cross-check via HRSID). No retrain needed to *run*
   it; recall will be low until the detector is fine-tuned on the new domain — report that honestly.
5. **Multi-station latency.** `find_passes_multi(sat, [sfax, svalbard, …])` → re-run B3/B4; report
   `max_gap_h` and the p90 latency vs the single-station baseline. This is the §8 multi-station gap.
6. **Deadline-aware relay.** Re-run B4 with per-item `deadline_s`; report the fraction of items that
   meet their deadline direct-only vs with the relay, and the `deadline_missed` count.

None of these change the accepted paper's *claims*; they make the paper's own constructs runnable and
directly comparable to the measured LoD pipeline, which is what Phase 3 is for.
