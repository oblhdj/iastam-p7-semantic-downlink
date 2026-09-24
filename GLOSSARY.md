# Glossary — IASTAM 6.0, Problem 7

Plain-language definitions of every term used in our project, our slides and our code.
Keep this open while reading the code; reuse the wording in the paper and in the jury Q&A.

---

## General

**Downlink / uplink** — Downlink = satellite → ground. Uplink = ground → satellite. The uplink is
much slower, so we only upload small things (thresholds, an AIS list).

**Buffering** — Keeping data in memory to send later instead of dropping it. Our satellite waits up
to 11 hours between passes, so without buffering everything not sent immediately is lost.

**Scheduler** — The part that decides *what to send and in which order* during a pass. Like packing a
suitcase with a weight limit.

**Encoder vs policy** (in `sat7/scheduler.py`)
- *Encoder*: how a ship becomes bytes (raw image / small crop / just coordinates).
- *Policy*: the sending order (oldest first, most valuable first…).
Separating them lets us test "same data, different order", which is how we showed the scheduler alone
raises delivered ships from 68.5 % to 93.0 % (simulation).

**Load sweep** — Repeating the experiment at different data volumes (5 000 → 160 000 tiles/day) to
find where each method breaks.

**Regression test** — A test that checks a fixed bug never comes back (we have one for the
"cloudy tile dropped with visible ships" bug).

---

## WP0 — Data integrity

**Dedup (de-duplication)** — Removing duplicates. **Split** — dividing data into train / validation / test.

**Why it matters** — Airbus tiles are crops of larger scenes and neighbouring crops overlap, so the
same ship can appear in two images. If one is in training and the other in test, the model has already
seen the answer and the score is inflated.

**pHash (perceptual hash)** — A short fingerprint of what an image *looks like*: shrink to 32×32 grey,
apply a DCT (frequency transform), keep the low frequencies, compare each to the median → 64 bits.

**Why not MD5/SHA?** Cryptographic hashes change completely if one pixel changes; they only find
*identical* files. Our duplicates are *similar but not identical* (shifted crop, different brightness).
pHash gives similar fingerprints for similar images.

**Hamming distance** — Number of differing bits between two fingerprints.
`1011 0110` vs `1011 0010` → distance 1. Small distance = near-duplicate.

**LSH banding** — Splitting each 64-bit hash into 4 bands of 16 bits and only comparing images that
share a band. Avoids comparing every pair with every other pair.

**Union–Find** — A simple structure to merge things into groups: if A~B and B~C then A, B, C form one
group. Each group goes entirely into one split.

**ORB + RANSAC (geometric verification)** — ORB finds distinctive points in two images and matches
them; RANSAC checks whether the matches fit one consistent shift. This *proves* two tiles show the
same scene, instead of just looking similar.

---

## WP1 — Detector

**YOLO** — "You Only Look Once": a detector that produces all boxes in a single pass (fast).
**v8** = version 8, **n** = nano, the smallest and fastest variant.

**FP32 / FP16 / INT8** — How numbers inside the model are stored: 32-bit float (default),
16-bit float (≈2× smaller), 8-bit integers (≈4× smaller, 1.5–3× faster, small accuracy loss).
Small objects suffer first under INT8, so we measure accuracy by ship size.

**ONNX** — A universal model file format: train in PyTorch, export to ONNX, run anywhere without
PyTorch. Like exporting a document to PDF.

**Tile vs frame** — A *frame* is one full camera image (e.g. 10 000 × 10 000 px); a *tile* is a small
square cut from it (768 × 768). Detectors resize their input to a fixed size, so a whole frame would
shrink a 40 px ship into nothing. Tiles keep ships at full size (better accuracy), frames are faster.
Airbus already ships 768 px tiles.

**IoU (Intersection over Union)** — Overlap between a predicted box and the true box:
shared area ÷ combined area. 0 = no overlap, 1 = perfect. "IoU ≥ 0.5" is the usual rule for
"this detection is correct".

**Precision / recall** — Precision = of the boxes we reported, how many were real ships.
Recall = of the real ships, how many we found. **mAP** summarises both over all thresholds.

**NMS (non-maximum suppression)** — Removes duplicate boxes on the same object, keeping the
highest-scoring one. It struggles when ships are very close together.

---

## WP2 — Confidence calibration

**Confidence** — The detector's own score for a box (0…1).

**Calibrated** — A score of 0.9 really means "90 % of such boxes are ships". Detectors are usually
**over-confident**, and our level-of-detail rule depends on the score being meaningful.

**Reliability diagram** — Chart of predicted confidence (x) against observed correctness (y).
Perfect calibration = the diagonal.

**ECE (Expected Calibration Error)** — One number for that chart: the average gap between confidence
and reality, computed over bins of confidence.

**Temperature scaling ("fit temperature")** — The simplest fix: divide all scores by one learned
number *T*, chosen on the validation set. T > 1 lowers over-confident scores. No retraining needed.

---

## WP3 — Classic pre-filter

**HSV** — Colour representation: **H**ue (which colour), **S**aturation (how vivid), **V**alue (how
bright). Clouds = bright + weakly coloured, which is easier to express in HSV than in RGB.

**Morphology** — Shape operations on images: *erosion* shrinks bright areas, *dilation* grows them,
*opening* = erosion then dilation (removes small bright things), *closing* = the opposite.

**SE (structuring element)** — The shape used by morphology (e.g. a disk of N pixels).
Rule: the SE must be **bigger than a ship**, or the top-hat will erase ships.

**Top-hat** — Image minus its opening. Keeps small bright objects (ships), removes the slowly varying
sea background (sun glint, colour gradients).

**Otsu thresholding** — Picks a threshold automatically by best separating the histogram into two
groups. Weakness: it always splits, even on an empty sea.

**Canny** — Classic edge detector. Many edges = coast / port / land; almost none = open sea.

**Connected components** — Groups touching white pixels into objects, giving each one a box and area.

**Watershed** — Splits blobs that touch (two ships side by side) using the distance to the border.

**Tune on val, report on test** — Choose settings on the validation set, then measure the final number
once on the untouched test set. Otherwise the score is optimistic.

**"It only saves compute"** — The pre-filter's job is to reduce processing, not to decide what a ship
is. Dropped tiles still leave a thumbnail so nothing disappears silently.

---

## WP4 — Level of detail with measured values

**ROI (region of interest)** — The cropped rectangle around one ship.

**Level of detail (LoD)** — How much we spend on one detection: L0 = coordinates only (~40 B),
L1 = small crop, L2 = larger progressive ROI, plus tile and thumbnail options.

**Encoding at several qualities** — The crop is compressed as JPEG / JPEG2000 at different quality
settings; quality is the knob that trades bytes against clarity.

**bytes(ℓ)** — The *measured* file size at level ℓ (not an estimate).

**g(ℓ)** — The *measured* probability that the ground can still detect the ship after decoding the
payload sent at level ℓ. We get it by re-running the detector on the decompressed crop.

Example table (numbers to be measured):

| Level | bytes(ℓ) | g(ℓ) |
|---|---|---|
| Coordinates only | 40 B | 1.00 |
| Crop, quality 20 | 900 B | 0.71 |
| Crop, quality 60 | 2.4 KB | 0.95 |
| Full ROI | 30 KB | 0.99 |

**Why it matters** — The scheduler currently uses guessed values; with these tables it uses measured
ones, so its decisions improve. And a jury can check a measurement, not a guess.

**Progressive coding (JPEG2000 / CCSDS 122)** — A file that can be cut at any point and still be
usable, just blurrier. This is what lets the scheduler fill the last free bytes of a pass.

---

## WP5 — Scheduler as an optimisation problem

**Knapsack problem** — A bag with a weight limit and objects with weight and value: maximise total
value. Our pass is the bag, bytes are the weight, information is the value.

**The objective**

    maximise   sum_i  v_i * h_i(x_i)
    subject to sum_i  b_i * x_i  <=  C_pass

- `x_i` — how much of item *i* is sent (0 = nothing, 1 = all, 0.5 = half)
- `b_i` — its size in bytes
- `v_i` — its value (a dark vessel counts more)
- `C_pass` — bytes the pass can carry
- `h_i` — how usefulness grows with the fraction sent; **concave**, because the first half of an image
  is worth more than the second half

In words: *get the most useful information into the bytes available.*

**Greedy** — Simple rule: always take the best value-per-byte first. Very fast, usually good, not
guaranteed optimal. This is what the satellite runs.

**DP (dynamic programming)** — Finds the *exact* best combination by reusing sub-results. Slow, used
offline on small cases to measure how good greedy is.

**LP (linear programming)** — Allows fractions and solves with a standard solver; gives an upper bound
nobody can beat.

**Optimality gap** — The distance between greedy and the exact optimum, e.g. "within 3 % of optimal
and 1000× faster". This is the result that makes the scheduler defensible.

**Aging** — Increasing an item's priority as it waits, so old data is not starved forever.

---

## Mission / space terms

**Pass (contact window)** — The minutes while the satellite is visible from the ground station.
Ours: 5 passes/day over Sfax, 34 min total, up to 11.4 h between passes.

**Elevation angle** — How high the satellite is above the horizon. Low elevation = longer distance =
weaker signal = lower data rate.

**Link budget** — The calculation of achievable data rate from power, distance and bandwidth.

**AIS (Automatic Identification System)** — Radio system where ships broadcast their identity and
position. **Dark vessel** = a ship seen in the image with no matching AIS signal — which may mean
hiding, *or* simply a reception gap. It raises priority; it proves nothing by itself.

**COTS (commercial off-the-shelf)** — Normal consumer hardware (e.g. Raspberry Pi) used in space
instead of expensive space-grade parts.

**SEU (single event upset)** — A bit flipped by radiation. None were observed on BUPT-1's Raspberry Pi
in 6 months at ~500 km.
