# IASTAM P7 — DEMO runbook

**What this demo proves:** a satellite can send *information about ships* instead of pixels —
**557× less data**, 0.685 ship recall (@15% cloud), scheduling provably near-optimal — all on
**real Airbus imagery with models we trained**. It is a **demo, not a prototype**: the pipeline
runs end-to-end on a laptop; "runs onboard" is presented as a TARGET, not a flown system.

---

## Two ways to run it

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
- **[3] The result — point at the headline.** "Five hundred fifty-seven times less data, and we
  still recover sixty-eight-point-five percent of the ships — *at an assumed fifteen percent
  cloud*. That is six-point-three points better than a fair Phi-sat-2 baseline."
- **[4] B0→B4.** "This is the progression our paper promised: raw image, detector, SAHI, our full
  semantic pipeline, and an inter-satellite relay that cuts worst-case latency from eleven to six
  hours without changing what's delivered."
- **[5] It's honest.** "Every stage you just saw ran on *real* data. The one thing we do *not*
  claim is flight hardware — every timing is a laptop, so 'runs onboard' is a target, not a
  prototype."

---

## The 3 questions judges will ask — and your answers

| They ask | You say |
|---|---|
| "Is 0.685 the real recall?" | "At an assumed **15% cloud**. The band is **0.40–0.82** over 0–50% cloud — cloud fraction moves it more than anything else, so we never quote it bare." |
| "Does this actually run on a satellite?" | "Not yet — this is a **demo, not a prototype**. Every timing is a laptop RTX 5060. We map that compute+energy onto a flight processor as a **TARGET** (report 17), and we say so." |
| "You rely on SAHI — isn't it expensive?" | "We measured it: a cut ship is usually still detected (report 13), so blanket SAHI is a poor trade. We argue **selective** slicing. It costs 1.56× compute, not 2.25× (report 16)." |

---

## Where the numbers come from (traceability)
- Headline table & reduction → `code/results/wp6_real_table.csv` (row *Ours: LoD + value-greedy*).
- B0→B4 → `code/results/wp18_campaign.json` (sanity gates assert B4 reproduces B3).
- End-to-end integrity → `code/results/wp11_integration.json` (decision flips @conf_high = 0).
- Dataset / leakage-free split → `code/results/wp0_stats.json`.
- Full narrative per work package → `reports/01`–`reports/21`.

## If something breaks
- No GPU / no dataset → `python demo/summary.py` alone (it's the whole story, from committed data).
- Weird numbers in old PDFs → ignore them; see the superseded-numbers list in `docs/START_HERE.md §7`.
- The `.csv`/`.json` files are always authoritative over any prose or slide.
