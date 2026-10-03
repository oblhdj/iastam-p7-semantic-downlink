# 18 — B0–B4 campaign runner: a composition spec (decision doc, not code)

> ### ✅ EXERCISED 3 Oct — spec validated against real entry points (B0–B3), B4 confirmed blocked
> `../code/scripts/wp18_campaign_runner.py` now builds the B0–B3 runner against this spec and ran it
> small-scale: B0 60,124 MB/day [SIM], B1 recall 0.782 [REAL], B2 recall 0.745 @ 1.56× [REAL], B3
> recall 0.691 and **560× vs B0** [SIM-over-REAL] — the 560× matches report 05's 557×, a sanity
> check that the chain is wired right. **B4 is a stub** (`route(plan, links)`) that raises
> `NotImplementedError`, so full B0–B4 end-to-end stays BLOCKED, exactly as §4 states. **One thing
> the spec glossed, now corrected below (§1):** B1/B2 emit *raw detections* but B3 consumes the
> *per-tile catalogue frames*, and B2's unit is a *swath* while the catalogue is per *tile* — the
> catalogue-build **adapter** is a step, not an arrow. Spec is the source of truth, so it is fixed
> here rather than worked around in code. Data: `../code/results/wp18_campaign.json`.

**This is a planning document, not a sprint to working code.** It fixes how the accepted paper's
B0→B4 configs compose into one campaign runner, using the module entry points that exist *today*,
so the runner can be built once and not redesigned when the relay (B4) lands. No runner is built
here. `../docs/START_HERE.md` §5 item 4 asks to "run the B0–B4 campaign end to end"; this decides
its shape first and is explicit about what cannot execute yet.

**Status in one line.** B0–B3 are buildable from existing code (**B2 done** — `sat7/b2_sahi_fusion.py`,
[report 16](16-swath-policies.md); B1/B3 are the `wp11_integration_demo` chain over
`sat7/real_workload.py` + `sat7/scheduler.py`; B0 is a raw-volume figure). The energy column comes
from [report 17](17-energy-model.md). **B4 does not exist.**

## 1. The configs, their I/O, and today's entry points

Each config adds one stage: **B0 ⊂ B1 ⊂ B2 ⊂ B3 ⊂ B4**. The data spine is
`tiles → [B1 detector | B2 SAHI+detector+fusion] → detections → [B3 catalogue → scheduler] →
downlink plan → [B4 routing]`.

| config | onboard | input | output | code entry point today | status |
|---|---|---|---|---|---|
| **B0** raw reference | none | tiles (raw bytes) | full-image downlink volume | raw tile byte totals (60,124 MB/day, [report 05](05-scheduler-on-real-detections.md)); no module needed | buildable |
| **B1** YOLO on tile | detector | 768 px tile | per-tile detections (boxes+conf) | `ultralytics.YOLO(weights).predict(tiles, imgsz=768)` — as `wp11_integration_demo.py:154`, wp12, wp16; **no sat7 wrapper yet** | buildable (a thin `detect_fn` wrapper would centralise it — and B2 already needs that callable) |
| **B2** SAHI+YOLO+fusion | detector + SAHI | wide swath + `detect_fn` | fused swath-global detections | `sat7.b2_sahi_fusion.run_sahi(swath, detect_fn, SahiConfig(window=768))` | **DONE**, window-parameterised, cross-checked byte-identical to wp16 |
| **B3** +semantic policy | +LoD +scheduler | detections → catalogue (`tiles`,`ships` frames) | scheduled downlink: `Item`s, `Result`, `metrics` (recall, bytes, what-the-sat-knew) | `sat7.real_workload.workload_from_catalogue` → `sat7.scheduler.encode_lod` → `simulate(…, ValueGreedy())` → `metrics`; + `onboard_ceiling` (`wp6_simulate_real`). Exactly the wp11 chain. | buildable (`wp11_integration_demo` already wires B1→B3 on single tiles) |
| **B4** +adaptive comms / relay | +path choice | B3 plan + link windows | direct-vs-relay routing, `E_relay` | **NONE — does not exist** | **not buildable** |

The detector is shared across B1/B2/B3; B2 only swaps the per-tile `predict` for `run_sahi` over a
swath.

**⚠ The catalogue-build adapter is a step, not an arrow (corrected 3 Oct after wiring B0–B3).** The
spine above draws `B1/B2 detections → B3 catalogue` as one hop. It is not: B1/B2 emit *raw
detections* `(cx,cy,w,h,conf)`, while B3 consumes the per-tile **catalogue frames** (`tiles`: context,
cloud_frac, n_ships, false-alarm counts; `ships`: matched found/conf). Converting one to the other is
a required stage with its own entry points — `sat7.prefilter.run_prefilter` (context/cloud), plus
`wp6_build_catalogue.flag_predictions` / `count_false_alarms` and the greedy GT match — i.e. exactly
**wp11 stage 4 / `wp6_build_catalogue`**. The runner must treat this adapter as a named stage between
perception (B1/B2) and policy (B3), not a cast.

**⚠ B2's unit is a swath; B3's catalogue is per tile.** B1 is per-768px-tile, so it maps to the
catalogue one-to-one. B2 produces *swath-global* detections, which must be attributed back to their
source tiles before they can populate a per-tile catalogue — or B3 must be re-expressed at swath
granularity. The spine hid this because B1 and B2 were both drawn as "→ detections".

**What this run did about it:** B3 was driven from the *stored* catalogue (`wp6_tiles.csv` /
`wp6_ships.csv`) — the proven path, since [report 12](12-end-to-end-integration.md) showed live
inference reproduces it to 1e-4 — and B1/B2 were validated as perception front-ends separately. A
live B1→B3 / B2→B3 campaign still needs the adapter above built and (for B2) the swath→tile
attribution; both are plumbing with a known target, not new measurements. This spec is the source of
truth, so the adapter and the unit mismatch are recorded here rather than hidden in the runner.

## 2. What one campaign run reports per config — and the labels it must carry

A run emits one row per config. A campaign necessarily **mixes figure sources**, so every cell keeps
its REAL/SIM/LIT/TARGET/ASSUMPTION label (the project rule):

| column | label and source |
|---|---|
| ship recall (overall / small / med / large) | **REAL** for B1/B2 (detector on Airbus). For B3 it is **SIM-over-REAL** (scheduler sim over real detections, report 05) and **must state the cloud fraction** — never bare (START_HERE §7). |
| data sent, reduction vs B0 | **SIM** (orbit sim) over **REAL** detection volumes |
| compute (slices/swath, ×regular tiling) | **REAL** geometry; B2 carries SAHI's **1.56×** at window 768 (report 16) |
| `E_proc` / `E_comm` / `ES` | from [report 17](17-energy-model.md): `Tₖ` **REAL** (laptop RTX 5060, *not* flight hw), `Pₖ` **ASSUMPTION** (swept), downlink **SIM**. Quote **ratios**, not absolute joules. |
| B4 routing / relay energy | **does not exist → TARGET at best**; any B4 cell is incomplete until built, and everything downstream of B4 inherits that |

A run that cannot fill B4 leaves it labelled **TARGET**, never blank-implied-done.

## 3. Where B4 plugs in — to be agreed with the B4/energy track, not decided here

B4 adds a routing decision *after* B3's downlink plan: `min(J_direct, J_relay)`, `J = λ_E·E + λ_T·T`,
with `E` from report 17's model and link windows from `sat7.orbit`. The runner should treat B4 as a
post-B3 function `route(plan, links) -> (path, E_relay, T)` with a **null (direct-only)
implementation today**, so B0–B3 run now and the real relay slots into that one function without
reshaping the runner. **The exact interface** — what `E_relay`/`E_ISL`/`E_GS` the energy track
exposes, how link windows are passed — **must be agreed with whoever owns the B4/relay + energy
track.** This spec deliberately does not fix it solo; it only reserves the seam so nothing has to be
rebuilt.

## 4. What Phase 3 §5 item 4 depends on that doesn't exist yet

"Run the B0–B4 campaign end to end" (START_HERE §5 item 4) is **partially blocked today**:

* **B0, B1, B2, B3 can run** — code exists, or is a thin wrapper over existing modules.
* **B4 cannot** — no relay, no path choice, no `E_relay` (START_HERE §5 item 3). The energy *model*
  exists (report 17) but the relay it would price does not.

So the executable scope **today is a B0–B3 campaign**, with B4 reported as TARGET. The honest plan:
build the runner for B0–B3 now with the B4 seam from §3, and defer the "end-to-end B0–B4" claim
until the relay lands — so the campaign is never reported as complete while a fifth of it is a stub.

**Non-goals:** this document builds no runner, changes no module, and does not fix the B4 interface.
It is the shape to build against.
