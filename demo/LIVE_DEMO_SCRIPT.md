# Presentation outline and five-minute live-demo script

Two things: where the live demo sits in a presentation (part 1), and what to click and say in the
five minutes (part 2). The demo is the dashboard, `python demo/launch.py`, on the bundled
synthetic swath, at the default settings.

**Two kinds of number appear, and the script keeps them apart out loud.**

* *On screen, computed live:* what the page shows for the bundled swath. These are labelled
  **SYNTH** on the page. They show that the chain runs; they are not measurements. The values
  quoted below are the ones stored in `demo/fallback_assets/swath/`; a test fails if they drift.
* *Canonical, read from `code/results/`:* the numbers to leave the room with. A test compares every
  one quoted here with the result files (`code/tests/test_docs.py`).

---

## Part 1 — presentation outline

The slide deck is `paper/slides/index.html` (nine slides). The live demo replaces nothing; it goes
between slide 4 and slide 5, so the audience has seen the idea before the page and sees the
measured results after it.

| # | Slide (as in the deck) | Point to make | Time |
|---|---|---|---|
| 1 | Title | what the project is | 0:20 |
| 2 | A satellite sees far more than it can send | the link, not the camera, is the limit | 0:40 |
| 3 | Send the packet, not the picture | detect onboard, send information graded by confidence | 0:40 |
| 4 | Everything runs onboard, before the radio | the chain: pre-filter, detector, SAHI, policy, encoder, scheduler | 0:40 |
| — | **Live demo** (part 2) | the chain running, then the committed results | 5:00 |
| 5 | All five configurations ran | B0 to B4; B1 and B2 on the same scenes | 0:40 |
| 6 | Less data, more mission | 341×, with what is measured and what is modeled | 0:40 |
| 7 | Compute, not the radio, is the cost | energy: estimates, no power measured | 0:30 |
| 8 | Honesty & limitations | the negative results and the stated gaps | 0:50 |
| 9 | Run it yourself | one command | 0:20 |

A little over ten minutes. If the slot is five minutes, give part 2 alone; its first beat is slide
2's point. Questions: `demo/QA_CARD.md` stays on the podium.

Slide 9 of the deck predates the dashboard: it shows `demo/summary.py` and `bash demo/run_demo.sh`.
Say the command aloud instead: `python demo/launch.py`.

---

## Part 2 — the five-minute script

About 550 spoken words: four minutes of speech at a calm pace, which leaves a minute for the
clicks and scrolls. Do not add sentences. If you run long, cut the second paragraph of 2:00–2:45
first, then the last two sentences of 2:45–3:45; never cut the negative result in 1:15–2:00 or the
last beat.

### Before you start

- [ ] `python demo/launch.py --check` ends with `live dashboard: ready` and `static fallback: ready`.
- [ ] The page is open, full screen, on **A · Overview**; sample = the 3×3 synthetic swath;
      sidebar at its defaults (cut 0.25, window 768, overlap 0.20, capture time 2.00).
- [ ] In the sidebar's *Environment check* the last line reads `Windows power throttling — opted out`.
- [ ] Click through the eight tabs once, so every section is already drawn.
- [ ] `demo/QA_CARD.md` is on the podium.

The tabs are used left to right, once each. Nothing in the script needs the network, a GPU or the
dataset.

### 0:00 – 0:30 · The communication problem — tab **A · Overview**

> **On screen:** the opening paragraph and the architecture diagram.

"A small satellite images about sixty gigabytes of ocean a day and can send down about two hundred
megabytes. Almost everything it sees is thrown away. The usual answer is to compress the pictures
harder. Ours is to stop sending pictures: find the ships onboard, and send information about them."

### 0:30 – 1:15 · Load an image, show detection — tabs **B · Image**, then **C · Detection**

> **On screen, B:** the swath, 2304 × 2304 px, raw size 15.93 MB. Source: SYNTH.
> **On screen, C, left panel:** plain YOLO, 12 detections, 1 detector call.

"This scene is nine tiles, fifteen megabytes raw. These tiles are synthetic stand-ins: the Airbus
licence does not let us redistribute real ones. So what is computed here shows the chain running.
The measurements come later, and those are on real tiles."

*Switch to C.*

"This is the detector we trained, on this laptop's CPU. One call on the whole scene finds twelve.
Green boxes are above our onboard threshold; grey ones are never sent."

### 1:15 – 2:00 · SAHI and fusion — tab **C · Detection**, right panel, then scroll to the table

> **On screen, right panel:** SAHI, 14 detections, 16 detector calls.
> **On screen, table:** *Which recall belongs to which configuration*.

"The scene is three times wider than the detector's input, so one call has to shrink it, and small
ships disappear. SAHI cuts it into overlapping windows and fuses the boxes: fourteen, for sixteen
calls."

*Scroll to the table.*

"On a hundred and fifty scenes stitched from real tiles, one call finds 0.339 of the ships and SAHI
finds 0.717. And here is a result we did not want: against a plain tile-by-tile pass SAHI does not
win, 0.717 against 0.730, for more compute. No ship crosses a seam in our scenes, so the overlap
has nothing to recover. We report that."

### 2:00 – 2:45 · Semantic records and ROIs — tab **D · Semantic packet**

> **On screen:** boxes coloured by level; 7 at P1, 4 at P2, 3 at P3; packet 45.7 kB;
> the decoded records; the ROI crops.

"Now the idea itself. Each detection gets a level. Confident: a record of a few dozen bytes, no
picture. Uncertain: the record plus a small crop, so the ground can check. On a coast, where the
detector is weakest: the context as well. The cloudy tile sends nothing."

"These are real bytes: the page built the packets, counted them and decoded them again as a ground
station would. Forty-five kilobytes for this scene, payload and overhead shown separately."

### 2:45 – 3:45 · B0 to B4, direct against relay — tabs **E · Experiment**, then **F · Communication**

> **On screen, E:** the five-row table. Select **B4 — semantic + relay**; route stays *Policy-chosen*.
> **On screen, F:** route *Policy-chosen*; capture to ground 2.82 h; the latency chart.

"Our paper promised five configurations; they are one pipeline with three switches. B0 sends the
raw image. B1 adds the detector, B2 adds SAHI, B3 the semantic policy you just saw, and B4 lets a
second satellite relay the packet."

*Select B4, switch to F.*

"This part is a simulation: propagated orbits, a link budget, assumed powers. For this image the
policy sent every item through the relay: on the ground in 2.8 hours, against 9.6 direct. The chart
shows it depends on when the image is taken. The relay buys time. It does not save energy, and we
do not claim it does."

### 3:45 – 4:30 · The link-capacity finding, then data reduction, latency, energy — **F**, then **G · Results**

> **On screen, F, lower half:** *One image always fits. A whole day does not.*
> **On screen, G:** this image's tiles, then *The measured results*.

*Scroll down on F.*

"One image always fits. A day does not: a nominal day offers 130 percent of what our share of the
link carries. So we ran six days back to back. Our scheduler recovered 0.677 to 0.688 of the ships
every day: it sheds bytes, not ships. A first-in-first-out queue went from nine hours behind to
forty-three."

*Switch to G, scroll to "The measured results".*

"The headline: 341 times less data. Three quarters of those bytes are measured as real JPEGs; a
quarter are still model sizes. Latency is simulated. Energy is an estimate: we never measured
power."

### 4:30 – 5:00 · Findings and limitations — stay on **G**

> **On screen:** the energy table, default and SAHI variant.

"So: less data can carry more of the mission, and the scheduler matters most when the link is
full. What we have not shown: this ran on a laptop, not a satellite; no power was measured; a
quarter of the bytes are modeled; and our lead over a fair baseline holds in 74 of 80 settings,
failing under heavy load. Every number here is labelled with what it is."

---

## What the script leaves out, and why

Nothing above depends on a feature that is not reliable on a presenter's laptop.

| Not in the script | Why |
|---|---|
| Uploading an image live | it works, and was tested with real tiles and label files, but two file dialogs cost twenty seconds and the pre-filter can misjudge an unfamiliar image. Use it for a question, not in the five minutes |
| The detector-in-the-loop run on the real split (`--gpu`) | needs the dataset, the weights and the GPU environment, and takes about 45 s |
| The learned gate | a torch model; it is not in the CPU demo |
| Accuracy on the bundled swath | there is no ground truth for synthetic tiles; accuracy is quoted from the committed results only |
| Moving the sidebar sliders | each change recomputes the scene; it works, but the numbers in this script are the defaults |

**If someone asks for a real tile** and the dataset is on the laptop: tab B, *Upload*, choose a
test tile and its `.txt` label file of the same name. Section C then shows precision and recall
against the labels, and the page warns if the two names do not match.

## If something goes wrong

| Symptom | Do |
|---|---|
| the page shows "The live pipeline cannot run here" | carry on: it is showing the same sections from pre-generated results, and says so at the top |
| the page will not open at all | `python demo/launch.py --fallback`; if streamlit itself is broken, `python demo/launch.py --summary` in the terminal |
| detection takes many seconds | the laptop is throttling the process; check the *Environment check* line, plug in, and keep talking over the committed table in section C |
| a section shows "could not be rendered" | the other sections still work; move on and use the committed results in section G |
| a number is questioned | "the CSV is authoritative", and `demo/QA_CARD.md` |
