# IASTAM P7 — DEMO runbook

**What this demo proves:** a satellite can send *information about ships* instead of pixels —
**341× less data** (coastal tiles at their measured JPEG size; thumbnails and crops still
modeled), 0.685 ship recall (@15% cloud), scheduling provably near-optimal — all on
**real Airbus imagery with models we trained**. It is a **demo, not a prototype**: the pipeline
runs end-to-end on a laptop; "runs onboard" is presented as a TARGET, not a flown system.

---

## Three ways to run it

### Q. Live on CPU — no GPU, no dataset, ~2 seconds (the recommended live demo)
```bash
python -m pip install -r demo/quickstart/requirements.txt   # once: numpy, opencv, onnxruntime, sgp4, skyfield
python demo/quickstart/run_demo.py
```
The real `sat7` chain runs **live**: pre-filter → trained YOLOv8n (ONNX, CPU) → SAHI + fusion on a
3×3 swath → P0–P3 decision per detection → semantic packet vs raw bytes → value-greedy downlink
order. It ends on the measured canon read from `code/results/`. The tiles are **synthetic
stand-ins**, because the Airbus rules forbid redistributing the real ones. Everything computed on
them is labelled **SYNTH** (it shows the chain runs; it is not a measurement). Open
`demo/quickstart/_out/swath_annotated.jpg` to show the boxes coloured by P-level. Details and the
licence reasoning: [`quickstart/README.md`](quickstart/README.md).

### A. Instant — no GPU, no data, 1 second (use this if unsure, or on the projector laptop)
```bash
python demo/summary.py
```
Prints one screen: the problem → our headline → the **B0→B4 table** → the end-to-end integrity
check → the honesty labels. Every number is read live from the committed results in
`code/results/` — nothing is hard-coded. **This is your backup if anything else fails.**

### B. Live — runs the real pipeline (needs the GPU env `code/.venv312` + the dataset)
```bash
bash demo/run_demo.sh          # small sample, fast (~1 min) — for a live audience
bash demo/run_demo.sh --full   # full 5320 tiles — reproduces the headline exactly (~2–3 min)
```
This runs, in order: **(1)** the B0→B4 campaign (`wp18`), **(2)** real JPEGs through the real
chain prefilter→gate→detector→LoD→scheduler→ground with an integrity cross-check (`wp11`), then
**(3)** `summary.py`. If the GPU env is missing it falls back to **A** automatically.

> **Live runs write to `demo/_live/` (scratch, gitignored) — they never overwrite the
> authoritative results in `code/results/`.** The live run proves the pipeline *executes* on a
> sample; `summary.py` always reports the committed **full-run** numbers. A small sample's
> absolute recall is not comparable to the full run (a rare context can be one tile resampled
> thousands of times) — the integrity claim that carries over is **decision flips @conf_high = 0**.
> To reproduce the headline numbers exactly, use `--full`.

---

## Pre-flight checklist (do this before you walk in)

- [ ] `python demo/quickstart/run_demo.py` finishes in a few seconds on the projector laptop and
      writes `demo/quickstart/_out/swath_annotated.jpg` (install `demo/quickstart/requirements.txt`
      first; it needs no GPU and no dataset).
- [ ] `python demo/summary.py` prints the full screen **with no dataset and no GPU** — this is the
      guaranteed backup. (It reads only `code/results/*.csv|json`; the 7.7 GB Airbus set is **not**
      needed.) Confirm the B0→B4 table and the honesty block appear.
- [ ] Slide deck opens: double-click **`paper/slides/index.html`** in a browser; `←/→` navigate,
      `N` toggles speaker notes (the 90-s track), `F` full-screen. Works offline.
- [ ] **`demo/QA_CARD.md`** is on the podium (0.685+cloud, why 341× not 557×, B1 0.339, onboard
      TARGET, SAHI cost and its negative result, the sensitivity failures, six days over capacity,
      power not measured, Airbus-not-DOTA, relay=latency, novelty, INT8).
- [ ] If demoing live: `code/.venv312` exists and `bash demo/run_demo.sh` completes (~1 min). If not,
      you present from the slide deck + `summary.py` — nothing is lost.
- [ ] Numbers to have cold: **341×** (coastal tiles measured, thumbnails/crops modeled; the earlier
      all-modeled estimate was 557×), **0.685 @15% cloud (band 0.40–0.82)**, **+6.3 pts** fair
      baseline at nominal load, **B1 0.339 → B2 0.717** on the same scenes, **35.2 h → 10.9 h**
      relay (worst case; per-item median 7.9 → 3.4 h), and the scheduler under pressure: **six
      days at 130% of the link share — recall 0.677–0.688 every day with our scheduler, while
      FIFO's latency goes 9.4 h → 42.9 h**. Everything else: "the CSV is authoritative."

---

## Click-by-click (live demo)

1. Open a terminal.
2. `cd` into the project:
   ```bash
   cd "C:/Users/Mega-PC/Desktop/IASTAM_Problem7"
   ```
3. Run the backup summary first so a good screen is already up:
   ```bash
   python demo/summary.py
   ```
4. Then run the live pipeline:
   ```bash
   bash demo/run_demo.sh
   ```
5. Talk over it using the 90-second track below. When it finishes, the final block on screen is
   the same clean summary from step 3 — end on that.

---

## The 90-second talk track (say this, let the numbers sit)

- **[1] The problem.** "A small satellite images sixty gigabytes of ocean a day and can only send
  down two hundred megabytes. So ninety-nine-point-seven percent is thrown away."
- **[2] The idea.** "We stopped sending pictures. The satellite finds the ships itself, then
  spends the link on *information* about them — a confident ship costs forty bytes, an uncertain
  one earns a picture."
- **[3] The result — point at the headline.** "Three hundred forty-one times less data, and we
  still recover sixty-eight-point-five percent of the ships — *at an assumed fifteen percent
  cloud*. That is six-point-three points better than a fair Phi-sat-2 baseline."
- **[4] B0→B4.** "This is the progression our paper promised: raw image; one detector call on the
  whole scene, which finds a third of the ships; SAHI on the same scenes, which finds seventy-two
  percent; our full semantic pipeline; and an inter-satellite relay that cuts worst-case latency
  from thirty-five hours to eleven without changing what's delivered."
- **[+] If you have ten more seconds — the scheduler under pressure.** "A day of data is thirty
  percent more than our share of the link can carry. We ran six days back to back: our scheduler
  kept recovering the same sixty-eight percent of the ships every day, in about five hours. A
  first-in-first-out queue fell a further seven hours behind each day."
- **[5] It's honest.** "Every stage you just saw ran on *real* data. The one thing we do *not*
  claim is flight hardware — every timing is a laptop, so 'runs onboard' is a target, not a
  prototype."

---

## The questions judges will ask — and your answers

The full set is in **`demo/QA_CARD.md`** — keep it on the podium. The five that always come:

| They ask | You say |
|---|---|
| "Is 0.685 the real recall?" | "At an assumed **15% cloud**. The band is **0.40–0.82** over 0–50% cloud — cloud fraction moves it more than anything else, so we never quote it bare." |
| "Does this actually run on a satellite?" | "Not yet — this is a **demo, not a prototype**. Every timing is a laptop RTX 5060. We map that compute+energy onto a flight processor as a **TARGET** (report 22), and we say so." |
| "You rely on SAHI — isn't it expensive?" | "We measured it: a cut ship is usually still detected (report 13), so blanket SAHI is a poor trade. We argue **selective** slicing. It costs 1.56× compute, not 2.25× (report 16). Against one call on the whole scene it lifts recall **0.339 → 0.717**; against a plain tile-by-tile pass it does **not** win on our stitched scenes (0.717 vs 0.730) — no ship crosses a seam between independent tiles — and we report that." |
| "What if the satellite produces more than the link can carry?" | "It already does, and that is where the scheduler earns its place. A day offers **130%** of what our 25% link share sustains (134.5 MB per 24 h). Over **six consecutive days** value-greedy recovers **0.677–0.688** of the ships every day, first report in a median 4.7–5.6 h — it sheds bytes, not ships. FIFO on the same days: latency **9.4 h → 42.9 h**, and it ends having delivered 0.424 of day six's ships. One assumption sits under that: a progressively truncated item counts as delivered once it clears `min_fraction` (10% of its bytes)." |
| "Your earlier material said 557×." | "That was an all-modeled estimate: it charged every coastal tile a flat 32 kB, a size measured on open-sea tiles. We then measured all 1,415 coastal tiles — median 66.5 kB — and the figure is **341×**. Thumbnails and ship crops are still model sizes, and we say so next to the number." |

---

## Where the numbers come from (traceability)
- Headline table & reduction → `code/results/wp6_real_table.csv` (row *Ours: LoD + value-greedy*).
- B0→B4 → `code/results/wp18_campaign.json` (sanity gates assert B4 reproduces B3); its B1 and B2
  are read from the same-input run `code/results/wp26_b0_b4.json`.
- Coastal tile sizes behind the 341× → `code/results/wp28_coast_tile_model.json` (and the earlier
  557× estimate → `code/results/wp6_real_table_modeledcoast.csv`).
- End-to-end integrity → `code/results/wp11_integration.json` (decision flips @conf_high = 0).
- Units, energy, latency, fairness and the document-vs-results ledger →
  `code/results/wp29_validation.json` (37 checks, none failing; five stated gaps).
- Dataset / leakage-free split → `code/results/wp0_stats.json`.
- Full narrative per work package → `reports/01`–`reports/23`; Phase-3 write-up → `paper/PHASE3_RESULTS.md`.

## If something breaks
- No GPU / no dataset → `python demo/summary.py` alone (it's the whole story, from committed data).
- Weird numbers in old PDFs → ignore them; see the superseded-numbers list at the foot of
  `paper/PHASE3_RESULTS.md` (current to 10 Oct 2026; `docs/START_HERE.md §7` stops earlier).
- The `.csv`/`.json` files are always authoritative over any prose or slide.
