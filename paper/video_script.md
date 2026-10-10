# 2-minute video — script and shot list (Phase-3)

**Target: 2:00.** The narration below is **~300 words ≈ 120 s at a calm 150 wpm**. Do not add
sentences; if you must cut, cut from §4 (B0→B4) first — never §6 (the self-correction).

**Delivery.** Read slowly, short sentences, let the numbers sit. Do not read the on-screen text
aloud — the viewer reads it. **Put `341×` and the band `0.40–0.82` on screen as text** (numbers
heard once are lost). Under `341×`, in small type: *coastal tiles measured; thumbnails and crops
modeled.* Every figure a card shows is labelled REAL / SIM / LIT / TARGET.

---

## 0:00 – 0:12 · The problem
> **ON SCREEN:** a slow zoom on one Airbus tile. Two bars: **60,124 MB imaged · ~210 MB downlink.**

A small satellite photographs far more ocean than it can send home.
In one day ours images sixty gigabytes. The radio link carries two hundred megabytes.
So ninety-nine point seven percent of what it sees is thrown away.
The usual answer is to compress the pictures harder.

## 0:12 – 0:30 · Packet, not picture
> **ON SCREEN:** `docs/architecture.png`, revealed left to right. Big text: **send the packet, not the picture.**

We stopped sending pictures.
The satellite finds the ships itself, then sends a small **packet** about each one.
A ship it is sure about costs forty bytes. One it is unsure about earns a cropped picture.
One it may have missed on a coast earns the surrounding context.
And the relay, when it helps, forwards the **packet** — never the image.

## 0:30 – 0:48 · One real tile through the pipeline
> **ON SCREEN:** one tile, then `code/results/wp4_lod_sizes.png` (the 40 B / 900 B / 2.5 kB ladder).

Here is one real tile.
The gate decides there is something worth a closer look.
The detector finds the ships. SAHI stitches the detections into one map.
The encoder turns each ship into a priority packet — P0 to P3.
The scheduler picks what fits the next pass, by value per byte.

## 0:48 – 1:08 · B0 → B4 *(cut here first if long)*
> **ON SCREEN:** a five-row card B0→B4 (from `demo/summary.py` / `wp18_campaign.json`).

Our paper promised a progression, and we measured all of it.
Raw image. Then one detector pass over the whole scene — a third of the ships.
Add SAHI on the same scenes — seventy-two percent, at one-and-a-half times the compute.
Add the semantic policy — this is the full system.
And an optional inter-satellite relay.

## 1:08 – 1:28 · The numbers
> **ON SCREEN:** `code/results/wp6_real_sweep.png`, then big: **341×** (small type: *coastal tiles
> measured; thumbnails and crops modeled*) and **0.685 @15% cloud (0.40–0.82)**.
> Then a text card **35.2 h → 10.9 h** (worst case, with the relay).

Three hundred and forty-one times less data than sending the pictures.
Sixty-eight-point-five percent of the ships — at an assumed fifteen percent cloud —
and that is one hundred percent of what the satellite still knew. The link is no longer the
bottleneck; the detector is. The relay cuts worst-case latency from thirty-five hours to eleven.

## 1:28 – 1:52 · The honest part *(never cut)*
> **ON SCREEN:** split screen "we claimed" / "we measured", strike through the left;
> then `code/results/wp10_tornado.png` — point at the cloud bar.

We also broke our own results on purpose.
Our first baseline ignored coastal ships, which flattered us; fixed, our lead fell from sixteen
points to six. We called an image setting free; re-running the detector showed it costs four points
on small ships. And more than half of everything we report as lost traces to twenty-one cloudy
tiles — so we quote recall as a **band**, not a number.
Every timing is a laptop: running onboard is a **target**, not a prototype.

## 1:52 – 2:00 · Close
> **ON SCREEN:** `code/results/wp11_funnel.png`, then the repo URL + QR.

The whole chain runs end to end on real images and reproduces our simulation to four decimals.
Every figure is labelled. It is all open.

> **HOLD 3 s:** `github.com/oblhdj/iastam-p7-semantic-downlink`

---

## Shot list (all sources exist in the repo)

| # | time | source | note |
|---|---|---|---|
| 1 | 0:00 | slow zoom on one Airbus tile (static) | avoid unlicensed footage — a static tile is fine |
| 2 | 0:06 | title card: 60,124 MB vs ~210 MB | build the two bars |
| 3 | 0:12 | `docs/architecture.png` | reveal one stage at a time |
| 4 | 0:30 | one tile + `code/results/wp4_lod_sizes.png` | the 40 B / 900 B / 2.5 kB ladder |
| 5 | 0:48 | B0→B4 card (`demo/summary.py` block or `wp18_campaign.json`) | five rows, labels visible |
| 6 | 1:08 | `code/results/wp6_real_sweep.png` | ours vs FIFO vs fair baseline |
| 7 | 1:12 | big number card **341×** + **0.685 @15% cloud (band 0.40–0.82)** | hold 3 s; small type under 341×: coastal tiles measured, thumbnails and crops modeled |
| 8 | 1:20 | text card **35.2 h → 10.9 h** (source `wp18_campaign.json`, B4) | the relay; `sim_latency.png` is the older synthetic day and no longer matches |
| 9 | 1:28 | claimed-vs-measured split screen | make the strike-through visible |
| 10 | 1:40 | `code/results/wp10_tornado.png` | point at the cloud bar |
| 11 | 1:52 | `code/results/wp11_funnel.png` → repo URL + QR | hold to 2:00 |

## If you overrun
Cut in this order: the last sentence of §2 ("And the relay…"), then the SAHI line in §4.
**Never cut §6** — the self-correction is the most distinctive thing in the project.

## Recording tips
- Record narration **first**, then cut visuals to it.
- One take per section, not one take for the whole thing.
- Phone voice memo in a small carpeted room beats a laptop mic in a big room.
- Put `341×` and the band `0.40–0.82` **on screen as text** — the two numbers judges must leave with.
- No slide/plot exists for energy by design — the 2-min cut stays on data reduction + recall + relay;
  the energy story lives in the slide deck (`paper/slides/`) and `PHASE3_RESULTS.md §6`.
