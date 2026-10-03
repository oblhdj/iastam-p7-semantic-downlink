# 20 — Relay path choice: ISL windows, `min(J_direct, J_relay)`, and the direct-only gate

Builds the other half of the optional inter-satellite relay (B4): the **link windows** and the
**path-choice** `route(plan, links)` that [report 18](18-campaign-runner-spec.md) §3 reserved and
START_HERE §5 item 3 needs. The companion [report 19](19-relay-energy.md) supplies the energy
`E`; this one supplies the inter-satellite visibility, the latency `T`, and the decision rule that
weighs them. Code: `../code/sat7/relay.py`, `../code/scripts/wp20_relay_path_choice.py`. Data:
`../code/results/wp20_relay_path_choice.json`, `wp20_isl_windows.csv`.

**Labels, because this report deliberately mixes them:**
* **ISL windows, ground windows, latency `T`** — **SIM**: real orbit propagation. The relay is a
  second satellite built *the same way as the primary* — SGP4 via `sat7.orbit.make_satellite`, a
  reproducible Keplerian orbit in a RAAN-offset plane, no external TLE and **no new orbital
  mechanics**. Mutual-visibility windows fall straight out of the two propagated position tracks.
* **Per-item energy `E`** — **TARGET**: supplied by report 19 through `route()`'s `direct_energy`/
  `relay_energy` callables. This track runs with clearly-labeled placeholders so the logic is
  exercised and gated; they are not joules.
* **`λ_E`, `λ_T`** — **ASSUMPTION**: invented trade-off weights, swept.

## 1. The second satellite and the ISL windows (SIM)

The relay is `OrbitConfig(epoch=START, raan_deg=Δ)` — same altitude/inclination as the primary,
offset into another orbital plane — propagated by the same SGP4 code. Inter-satellite visibility is
a line-of-sight test on the two tracks: the relay sees the primary whenever the segment between them
clears the Earth plus a 100 km atmospheric margin.

The geometry has one finding worth stating before any numbers, because it decides whether a relay is
worth having at all:

| relay RAAN offset | ISL co-visibility | ground track vs primary | use as a relay |
|---|---|---|---|
| small (≤ ~20°) | **continuous** (always in view) | nearly the **same** | **useless** — it reaches Sfax at the same times the primary does |
| large (~90°) | **intermittent** (46 windows / 36 h) | **complementary** | **valuable** — covers Sfax when the primary cannot |

So a relay's value is **coverage complementarity, not ISL availability**. A co-planar companion is
always reachable and never helpful; a cross-plane companion is only intermittently reachable but
actually fills the gaps. The default models the useful case (Δ = 90°): **6 primary ground passes, 6
relay ground passes, 46 ISL windows** over the 36 h day.

## 2. The decision rule

`route()` chooses, per item, `min(J_direct, J_relay)` with `J = λ_E·E + λ_T·T`:

* **direct**: wait for the primary's next Sfax pass. `T_direct` = time until it (SIM); `E_direct`
  from report 19's `direct_energy(d, p_tx, r_gs)`.
* **relay**: wait for the next ISL window, cross-link to the relay, then wait for the relay's next
  Sfax pass. `T_relay` = time until *that* reaches the ground (SIM); `E_relay = E_ISL + E_GS` from
  report 19's `relay_energy(d, p_tx, p_isl, r_gs, r_isl)`.

The split is clean and matches report 19's: **energy is window-independent** (`P·D/R`, theirs),
**latency needs the windows** (mine). `route()` calls their two energy primitives unchanged — the
signatures in `sat7.relay` match report 19 §4 exactly, so the real functions drop in with no reshape.

## 3. What the relay buys — latency (SIM, energy-free)

With `λ_E = 0` (pure latency, no energy assumption needed), over 432 arrival times across the day:

| policy | median | p90 | **max (worst case)** |
|---|---|---|---|
| direct only | 4.78 h | 9.75 h | **11.53 h** |
| direct + relay | 1.95 h | 4.43 h | **5.78 h** |

**The relay roughly halves worst-case latency — 11.53 h → 5.78 h (−5.75 h)** — and the median from
4.78 to 1.95 h, choosing relay for **44 %** of arrivals. The 11.53 h direct worst case is the
single-station gap [report 09](09-whole-day-bound.md) measured (Sfax, ~11.4 h max); a complementary
relay is precisely the thing that closes it. This number is REAL orbit geometry and does not depend
on any energy assumption.

## 4. When is relay *worth* it — the energy trade (TARGET)

Latency says "relay often"; energy says "relay costs more" (report 19: `E_relay` ≈ 1.8× direct,
never cheaper — an added hop only adds joules). `route()` resolves it through `λ_E`. Sweeping the
energy weight (with report 19's placeholder `E`, `λ_T` = 2.78e-4/s):

| `λ_E` | 0 | 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| relay chosen | 44 % | 44 % | 44 % | **23 %** | **0 %** | 0 % |

The rule behaves correctly: when energy is cheap relative to time, relay wins wherever it is faster;
as energy is weighted more heavily the crossover arrives and relay is abandoned. **Where that
crossover actually sits is TARGET** — it depends on report 19's real `E_relay` (its `P_isl`/`R_isl`
are themselves ASSUMPTION-and-swept, 1.04×–3.0×) and on the operator's `λ_E/λ_T`. So the *mechanism*
is delivered and tested; the *operating point* lands when report 19's numbers are no longer
placeholder.

## 5. Sanity gate — direct-only reproduces the ground-only baseline exactly

Same discipline as [report 16](16-swath-policies.md): the path choice must collapse to the existing
no-relay simulation when the relay is removed, or no relay number means anything. With `isl_enabled =
False`, `route()` must reproduce the ground-station-only opportunity set and latency that
WP5b/WP6/WP8/wp18 use — `find_passes(primary, Sfax, 2026-09-18, 36 h)`.

**It does, exactly:** all **6** primary passes reproduced (same rise times, same capacities to < 1 B),
and the direct-only latency equals an independently-computed next-pass wait at **all 432** arrival
times. The relay layer adds nothing to — and subtracts nothing from — the baseline when disabled.

## 6. Limits and what stays blocked

* **Single relay, single ground station.** The benefit is specifically complementary Sfax coverage;
  with more stations the direct gaps shrink and the relay's latency win with them. Not generalised
  beyond one Sfax day.
* **The relay geometry (Δ = 90°) is an ASSUMPTION**, chosen to model the *useful* (complementary)
  regime; §1 reports the small-offset regime where a relay is useless, so the result is not cherry-
  picked — it is conditional on the constellation actually providing complementary coverage.
* **`E` is TARGET** (report 19 placeholders here); the final `J` decision and the `λ_E` crossover are
  provisional until report 19's `P_isl`/`R_isl` are sourced. The **latency** numbers (§3) are not —
  they are SIM and final.
* **`R_isl` as an ISL *rate*** is report 19's ASSUMPTION; this track supplies the ISL *window* and
  *range* (`wp20_isl_windows.csv`, min range 1252 km at Δ 90°), from which a real `R_isl` could later
  be derived with a crosslink budget — not done here.
* **`direct_latency_s` parameter (added for the campaign).** `route_item` gained an optional
  `direct_latency_s` (backward-compatible, default None = the next-pass-wait behaviour above, so every
  number in this report is unmoved). When supplied it overrides `T_direct` with a caller-provided
  direct latency — [report 21](21-b0-b4-campaign.md) passes B3's *actual* mid-pass delivered latency
  through it so the campaign's direct-only baseline reproduces B3 exactly. Note the resulting
  asymmetry: `T_direct` then reflects mid-pass delivery while `T_relay` is still the window estimate
  (relay's next pass *rise*), so the path choice is slightly **conservative against relay**.
* **Full B0–B4 end-to-end** is now wired ([report 21](21-b0-b4-campaign.md)): windows+choice here +
  energy in report 19, into `wp18_campaign_runner.py`. B4's recall/MB are B3's (reroute); latency is
  SIM; **energy stays TARGET** until report 19's `P_isl`/`R_isl` are sourced.
