# 2-minute video — script and shot list

**Target: 2:00 exactly.** The narration below is **295 words ≈ 118 s at a calm 150 wpm**, which
leaves ~2 s of air. Do not add sentences; if you must, cut from §4 first.

**Delivery notes.** Read it slowly. Short sentences on purpose. The numbers are the point — say
them clearly and let them sit. Do not read the on-screen text aloud; the viewer can read.

---

## 0:00 – 0:15 · The problem

> **ON SCREEN:** a satellite passing over ocean; then a progress bar that fills only 1%.
> Big text: **60,124 MB imaged — 210 MB of downlink.**

A small satellite photographs far more ocean than it can ever send home.
In one day ours images sixty gigabytes. The radio link carries two hundred megabytes.
So ninety-nine point seven percent of what it sees is thrown away.
The usual answer is to compress the pictures harder.

---

## 0:15 – 0:35 · The idea

> **ON SCREEN:** `docs/architecture.png` — the pipeline, animated left to right.
> Big text: **stop sending pictures. Send what you learned.**

We stopped sending pictures.
The satellite finds the ships itself, then spends the link on **information about them**.
A ship it is sure about costs forty bytes. One it is unsure about earns a picture.
A coastal tile it may have misread is sent whole.
Then a scheduler picks what goes down each pass, by value per byte.

---

## 0:35 – 1:05 · What it achieves

> **ON SCREEN:** `code/results/wp6_real_sweep.png`, then the headline number large:
> **557× less data.**

Five hundred and fifty-seven times less data than sending the pictures.
Sixty-eight percent of the ships delivered — and that is **one hundred percent of what the
satellite still knew**. The link is no longer the bottleneck. The detector is.
When the link does saturate, we deliver two-point-three times more ships than first-come
first-served.
Every number is measured on fifty-three thousand real Airbus tiles.

---

## 1:05 – 1:40 · The honest part *(this is the section that wins)*

> **ON SCREEN:** split screen. Left: "we claimed". Right: "we measured". Strike through the left.

We also broke our own results on purpose.
Our first baseline ignored coastal ships, which flattered us. Fixed, our advantage fell from
sixteen points to six.
We adopted an image setting as free. Re-running the detector showed it costs four points of
small ships.
And more than half of everything we report as lost traces back to twenty-one cloudy tiles.
So we quote recall as a **band**, not a number.

---

## 1:40 – 2:00 · Close

> **ON SCREEN:** `code/results/wp11_funnel.png`, then the repo URL and QR code.

The whole chain runs end to end on real images, and reproduces our simulation to four decimal
places. Ninety-six tests. Every figure labelled real, simulated, or from the literature.
It is all open.

> **HOLD 3 s:** `github.com/oblhdj/iastam-p7-semantic-downlink`

---

## Shot list (what to actually record)

| # | time | source | note |
|---|---|---|---|
| 1 | 0:00 | stock orbit clip **or** a slow zoom on one Airbus tile | avoid unlicensed footage — a static tile is fine |
| 2 | 0:08 | title card, 60,124 MB vs 210 MB | build the two bars |
| 3 | 0:15 | `docs/architecture.png` | reveal one stage at a time |
| 4 | 0:25 | `code/results/wp4_lod_sizes.png` | the 40 B / 900 B / 2.5 kB ladder |
| 5 | 0:35 | `code/results/wp6_real_sweep.png` | ours vs FIFO vs baseline |
| 6 | 0:50 | big number card: **557×** | hold 3 s |
| 7 | 1:05 | claimed-vs-measured split screen | make the strike-through visible |
| 8 | 1:25 | `code/results/wp10_tornado.png` | point at the cloud bar |
| 9 | 1:40 | `code/results/wp11_funnel.png` | |
| 10 | 1:52 | repo URL + QR | hold to 2:00 |

## If you overrun

Cut in this order: the last sentence of §2 ("Then a scheduler…"), then the FIFO sentence in §3.
**Never cut §4** — the self-correction is the most distinctive thing in the project and no other
team will have it.

## Recording tips

- Record narration **first**, then cut visuals to it. Far easier than the reverse.
- One take per section, not one take for the whole thing.
- Phone voice memo in a small carpeted room beats a laptop mic in a big room.
- Put the 557× and the band (0.40–0.82) **on screen as text**; numbers heard once are lost.
