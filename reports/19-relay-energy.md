# 19 — Relay energy: `E_relay = E_ISL + E_GS`, and what it is still blocked on

Builds the inter-satellite relay's energy terms that `../docs/START_HERE.md` §5 item 3 and
[report 18](18-campaign-runner-spec.md) §3 need, so `route(plan, links)` can weigh
`J_relay = λ_E·E + λ_T·T` against the direct path. Script: `../code/scripts/wp19_relay_energy.py`.
Data: `../code/results/wp19_relay_energy.json`. It builds the relay **energy** terms, in [report 17](17-energy-model.md)'s
`E = P·D/R` structure, reads every reused figure **live** from its source file (same discipline as
report 17), imports no torch, and — now that the path-choice `route()` exists ([report 20](20-relay-path-choice.md))
— folds the real relay fraction into the energy-saving ratio rather than leaving it a TARGET.

## 1. The two legs, and why relay is never the cheap option

The relay path replaces one ground hop with two: the originating satellite crosslinks the payload
to a relay satellite that has a ground pass sooner.

| leg | energy | hardware |
|---|---|---|
| `E_ISL` originating sat → relay sat | `P_isl · D / R_isl` | the inter-satellite crosslink (new) |
| `E_GS` relay sat → ground | `P_tx · D / R_gs` | the **same ground-link radio** as the direct path |
| **`E_relay = E_ISL + E_GS`** (constellation-total, per START_HERE 5.3) | | strictly **>** direct `E_comm = P_tx·D/R_gs` |

Adding a hop can only add joules, so **relay always costs more energy than direct** — the model
asserts this as a sanity invariant (below). Relay is never chosen to save power; it is chosen to
save **time**. Sfax is a single ground station with an 11.4 h max gap ([report 09](09-whole-day-bound.md)),
so routing through a satellite that sees the ground sooner cuts latency. That energy-versus-latency
trade is `route()`'s to make via `J = λ_E·E + λ_T·T`; this report only supplies the joules each path
costs, which is the half of `J` that belongs to the energy track.

## 2. The numbers — reused where they existed, flagged where they are new

Following report 17's rule, nothing is hardcoded that a result file already owns:

| quantity | value | label | source |
|---|---|---|---|
| `D` payload sent | 108.0 MB/day | **SIM** | live from `wp6_real_table.csv` (via `wp17.load_downlink`) |
| `R_gs` ground rate | 2.53 Mbps | **SIM** | the orbit sim's `passes.csv` (capacity ÷ contact), as report 17 |
| `P_tx` ground tx power | 15 W | **ASSUMPTION** | **reused from report 17**, not re-invented; swept 8–30 W |
| `P_isl` crosslink tx power | 12 W | **ASSUMPTION — new** | no repo figure exists; swept 5–30 W |
| `R_isl` crosslink rate | 2.53 Mbps (= ground, neutral) | **ASSUMPTION — new** | no repo figure exists; swept 1–50 Mbps |

`P_isl` and `R_isl` are the **only genuinely new constants**, because the ISL is different hardware
from the ground downlink and the repo had no crosslink figure — `sat7.orbit` computes `find_passes`
/ `rate_bps` for the **ground** link only. They are **ASSUMPTION and swept**, and become **LIT** only
when sourced from a crosslink datasheet. The parallel path-choice track ([report 20](20-relay-path-choice.md),
`sat7.relay`) now supplies the **real ISL windows** (SIM): 46 windows over 36 h, 656 min total, min
range **1252 km**, mean 2695 km (`results/wp20_isl_windows.csv`) — enough window time to pass the
daily payload ~115× over (feasibility checked). That range could **derive** a real `R_isl` from a
crosslink link budget if a datasheet lands; until then `R_isl` stays ASSUMPTION. *(The track adopted
this report's §4 interface verbatim — see §4. A min-range figure of 458 km that had leaked into
report 20's prose — a stale number from an earlier RAAN+30° probe — was flagged against the CSV and
corrected to 1252 km; the CSV, code, and the energy here were right throughout, as range is unused
in the energy.)*

## 3. Per-path energy, and the sanity gate

At `D = 108 MB/day`, `R_isl = R_gs` (neutral), `P_isl = 12 W`:

| path | energy/day | vs direct |
|---|---|---|
| direct (`E_comm`) | **5.12 kJ** | 1.00× |
| `E_ISL` | 4.09 kJ | |
| `E_GS` | 5.12 kJ | |
| **relay (`E_ISL + E_GS`)** | **9.21 kJ** | **1.80×** |

Three independent checks must pass before any number is reported (report 16/17 pattern):

* **(a) one source of truth for `E_comm`:** the direct path computed here equals report 17's
  `E_comm` to a relative 3e-4 — the two energy models cannot disagree on the same downlink.
* **(b) the hop invariant:** `E_relay > E_direct` (here 1.80×); if it ever came out ≤ 1, a sign or
  definition error has flipped the model, and it refuses to report.
* **(c) rate bound:** `R_gs` 2.53 Mbps ≤ the model's zenith peak 6.40 Mbps.

## 4. ES folds into report 17 — now computed, via `route()` with our energy injected

`ES = 1 − (E_proc + E_comm_mix) / E_bent_pipe`, where `E_comm_mix = (1−f)·E_direct + f·E_relay` and
`f` is the relay byte-fraction. `f` is `route()`'s output, not ours — so rather than invent it, we
**inject our two energy functions into [report 20](20-relay-path-choice.md)'s `sat7.relay.route()`**
(which adopted this report's §4 interface verbatim) and read the `f` it returns over the workload:

| | ES vs bent pipe |
|---|---|
| all-direct (`f = 0`, = report 17's figure) | 0.9822 |
| all-relay (`f = 1`) | 0.9808 |
| **weighted, `f = 0.44`** (latency-optimal, `λ_E = 0`) | **0.9816** |

The whole-day ES spread is **only 0.14 points**, so **ES is essentially insensitive to how much is
relayed**: relay energy (9.2 kJ) is a small rider on a processing-dominated budget ([report 17](17-energy-model.md):
proc : comm ≈ 9 : 1). **Relay is a latency buy, not an energy one** — report 20 measures it cutting
worst-case latency 11.53 h → 5.78 h; it barely moves the energy-saving ratio either way.

**Drop-in validated.** The injection is a true drop-in: at `λ_E = 0` the choice is energy-free, so it
must reproduce report 20's latency-optimal relay fraction — and it does, **`f = 0.440` exactly**,
with real joules now flowing through `route()` (9.2 kJ/relay-item) in place of its placeholder units.

**The interface `route()` adopted** (the energy primitives it calls per path for `J = λ_E·E + λ_T·T`):

```
direct_energy(d_bytes, p_tx, r_gs_bps)                  -> {E_comm_J, tx_time_s}
relay_energy (d_bytes, p_tx, p_isl, r_gs_bps, r_isl_bps) -> {E_ISL_J, E_GS_J, E_relay_J, isl_time_s, gs_time_s}
```

Latency `T` is **not** computed here: energy is window-independent (`P·D/R`), while `T` needs the
link windows, which are report 20's domain (SIM). We supply `E`; `route()` supplies `T`.

### Sensitivity of the new assumptions

Swept one at a time, `E_relay` ranges **1.04×–3.0×** the direct path (P_isl 5→30 W, R_isl 1→50 Mbps).
The *structure* and the *hop invariant* hold throughout; the *magnitude* is what the link-window
track's real `R_isl`/`P_isl` will pin down. The energy-track conclusion is robust regardless: relay
is a latency buy, and its joules are a small rider on a processing-dominated budget.

## 5. A consistency check done first (task prerequisite), and its verdict

Before building this, a flagged open question in report 17 was resolved: are its gate and detector
times measured the same way? **They are not.** The gate figure (0.052 ms, `wp3_gate.json`) is a
**pure-forward, batch-32** number — a random tensor already resident on the GPU, no I/O or
pre/post-processing. The detector figure (10.71 ms, `wp1_metrics.json`) is an **end-to-end
`model.predict` over a file list** — JPEG decode + letterbox + NMS + the Python match loop, effectively
batch-1. So "0.052 vs 10.71" is pure-forward-amortised against full-pipeline-unamortised, not a
like-for-like comparison.

* **It does not affect anything in this report.** `E_ISL`/`E_GS`/`E_direct` are transmit energies
  (`P·D/R`) and use neither forward time — the inconsistency structurally cannot propagate here.
* **It does not change report 17's ES_proc conclusion**: the gate is negligible against the 56.3 ms
  classic pre-filter it replaces, so the gating-stage saving holds whatever the gate's fair
  end-to-end cost is. Report 17's owner has since annotated its banner with the gate side of this
  (its 0.052 ms is the pure forward pass); the matching caveat on the detector side — its 10.71 ms is
  end-to-end `predict`, batch-1 — is the other half of why the two are not a like-for-like pair.

## 6. Limits and what stays blocked

* **`P_isl`, `R_isl` are invented** (ASSUMPTION, swept), pending a crosslink datasheet (→ LIT); the
  real ISL **range** now exists (report 20) and could derive `R_isl` via a link budget once one does.
  Magnitudes provisional; the structure and the hop invariant are firm.
* **The weighted ES is now computed** (`f = 0.44` → ES 0.9816) via report 20's `route()`, no longer a
  bound-only TARGET — but its *value* barely moves with `f` (0.14-pt spread), so the number that
  matters for the relay is latency (report 20), not this ES.
* **Latency `T` and the ISL windows** are report 20's (SIM); this report reuses its windows for an
  ISL feasibility check only.
* **Full B0–B4** is no longer blocked on missing pieces — `route()` (report 20) and these energy
  terms both exist. What remains is wiring [report 18](18-campaign-runner-spec.md)'s B0–B3 runner to
  `route()` for the B4 leg; that is plumbing, not a missing capability.
* All energy is **laptop-hardware-derived** on the processing side (report 17's caveat); the relay
  terms are pure link arithmetic and inherit only `P_tx`/`R_gs`, which are a sat-radio assumption and
  the orbit sim, not the laptop.
