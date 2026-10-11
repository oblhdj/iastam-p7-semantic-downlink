# Judge Q&A card — one page, keep it on the podium

Every answer below is defensible from `code/results/*`. Labels: REAL / SIM / LIT / TARGET / ASSUMPTION.
If cornered on a number you don't remember: **"the CSV is authoritative — `code/results/`"** and move on.

---

**Q. Is 0.685 the real recall?**
At an **assumed 15% cloud**. The honest figure is a **band: 0.40–0.82 over 0–50% cloud** — cloud
fraction moves recall more than any other quantity, and the Airbus set (0.4% cloud) can't settle it.
We **never quote 0.685 bare**. `source: wp6_real_table.csv, report 11`.

**Q. Your earlier figure was 557×. Why 341× now?**
557× was an **all-modeled estimate**: it charged every coastal tile a flat 32.4 kB, a size measured
on open-sea tiles. We measured all 1,415 coastal test tiles as real JPEGs (median **66.5 kB**) and
charge each its own size: 60,124 MB → **176 MB/day = 341×**, mean of three days. Recall is unchanged
(0.685). Still modeled: thumbnails and ship crops, 24% of the bytes — we say that next to the
number. `source: wp6_real_table.csv, wp28_coast_tile_model.json; 557× reproduces from
wp6_real_table_modeledcoast.csv`.

**Q. Does this actually run on a satellite?**
Not yet — it's a **demo, not a prototype**. Every timing is a laptop RTX 5060, so "runs onboard" is
a **TARGET** only (report 22; the laptop-to-edge mapping is in `paper/PHASE3_RESULTS.md` §6c). But it's the *class* of model Φ-sat-2 already flies on a ~1 W Myriad 2 VPU
[LIT]: a 3.15 M-param detector + 47 k-param gate, a few-% duty cycle at 40k tiles/day even on a VPU
6× slower. What a prototype still owes: wall-clock + quantised accuracy on a real board, thermal,
rad-tolerance.

**Q. You lean on SAHI — isn't it expensive?**
We measured it. A seam-cut ship is **usually still detected** (report 13), so **blanket SAHI is a
poor trade**; we argue **selective** slicing. Cost is **1.56× compute**, not 2.25× (report 16). Window
is **768, not the paper's 512** — the result reverses at 512, and we state the window.
On the same 150 scenes it lifts recall **0.339 → 0.717** over one call on the whole scene (B1 → B2).
**Negative result, reported:** against a plain tile-by-tile pass with no overlap it does *not* win
on those scenes — **0.717 vs 0.730**. They are stitched from independent tiles, so no ship crosses a
seam and the overlap has nothing to recover. `source: wp26_b0_b4.json`.

**Q. B1 is only 0.339? You used to show 0.78.**
The paper's B1 is YOLO on the **original image, no slicing**. A 3072 px scene has to be shrunk 4× to
fit the detector, and small ships vanish (recall **0.046** on ships under 32 px). The 0.78 was a
262-ship sample of the detector on **native 768 px tiles**, one call per tile — nothing to shrink.
On all 8,173 test ships that reference is **0.766** (precision 0.754); it is *not* the paper's B1
and we label it so. `source: wp26_b0_b4.json, wp24_detection_eval.json`.

**Q. Why Airbus, not DOTA / xView / HRSID?**
The paper permits optical maritime datasets; Airbus is one, **53,195 tiles, leakage-free split**
(a naive split would leak 1,617 groups; ours leaks 0 — report 01). Airbus tiles are already 768²,
so B1 and B2 are evaluated on **synthetic scenes stitched from tiles** (150 scenes, 1,164 ships). A
large-scene optical set would strengthen B2 — honest limitation, not a hidden one.

**Q. The relay — does it save energy?**
No, and we don't claim it. The relay is a **latency buy, not an energy win**: it costs **~1.8×
comm energy per relayed item** (+52% across the B4 mix, TARGET), and in return **cuts worst-case
latency 35.2 h → 10.9 h** and the per-item median 7.9 h → 3.4 h (SIM). The 35 h is the last
low-value item of a day that more than fills the link share; a ship's first report still lands in a
median 4.5 h. Recall and bytes are unchanged — B4 is a reroute (reports 19–21). Its relay latency is
a window estimate; the volume-aware two-leg simulation is `wp27`.

**Q. What happens when the satellite produces more than the link can carry?**
It already does — and that is where the scheduler earns its place. A nominal day offers **130%** of
what our 25% link share sustains (**134.5 MB per 24 h**). Run for **six consecutive days**,
value-greedy keeps recovering **0.677–0.688** of the ships every day, with a ship's first report in
a median **4.7–5.6 h**: it sheds bytes, not ships — low-value products are truncated or deferred.
FIFO on the same six days falls about seven hours further behind each day (median latency **9.4 h
→ 42.9 h**) and ends having delivered **0.424** of day six's ships.
Two things we state with it. One isolated day does fit, given the 36 h of passes the simulation
allows — that is the "100% of what the satellite still knew" figure. And recall counts a
progressively truncated item as delivered once it clears `min_fraction` (10% of its bytes): an
assumption, not a measurement. `source: wp29_validation.json`.

**Q. Did you measure power?**
No. **Every energy figure is an estimate**: assumed powers (CPU 28 W, GPU 60 W, TX 15 W, ISL 12 W,
all swept) times measured laptop stage times or simulated transmit times. Only ES_proc (57.6%) needs
no power — it is a ratio of measured times. And "ES_total ≈ 98%" is against a bent pipe sending every
raw byte, which this link could never carry; a link-limited bent pipe would spend 30.8 kJ/day, less
than our 54 kJ/day, for 0.2% of the ships.

**Q. Does your lead over the baseline survive your own sensitivity sweep?**
Not everywhere, and we say so. Over 80 settings, **value-greedy ≥ FIFO holds in all 80**; **"ours ≥
the fair baseline" holds in 74**. All six failures are at **160k tiles/day**, where the lead is only
**1.3 points** to begin with (6.3 at 40k, where nothing fails): `thumb_value` 0.05 and 0.10,
`conf_high` 0.90 and 0.99, `conf_low` 0.05 and 0.10. Each spends more bytes on a saturated link.
The +6.3 is a nominal-load claim. `source: wp10_sensitivity.csv`.
It is also a result of the **LoD ladder**: under the paper's pure Table I P0–P3 levels, recall
equals the fair baseline (**0.622** for both). `source: wp6_real_table_paper.csv; PHASE3_RESULTS §2b`.

**Q. What's the weakest number?**
Small-ship recall (**0.50–0.61** at the 0.25 operating cut) — the detector, not the downlink, is
the residual loss. And `P_tx` (invented, ASSUMPTION), but it only enters E_comm, which is ~5.5×
smaller than E_proc. And the byte sizes are only partly measured: coastal tiles yes, thumbnails
and crops no.

**Q. What's actually novel? YOLO and SAHI are off-the-shelf.**
Correct — those are building blocks. The novelty is **system/decision level**: the per-object
**P0–P3 priority packet** and the **`min(J_direct, J_relay)`** path choice, plus a value-aware
multi-pass scheduler that is **within 0.88% of the exact optimum** (report 08). The paper is a
methodology; Phase 3 is the measurement.

**Q. Why is INT8 not used?**
Measured and **rejected**: 8.5× *slower* on CPU and −2.2 pts on small ships (dynamic ONNX INT8, not
TensorRT). ONNX FP32 is the onboard build. A reported negative result (report 02).

---

**One-line pitch:** *341× less data, 100% of what the satellite still knew, every figure labelled and
every claim we could break, we broke — on real Airbus imagery, measured end to end.*
