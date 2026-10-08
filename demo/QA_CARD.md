# Judge Q&A card — one page, keep it on the podium

Every answer below is defensible from `code/results/*`. Labels: REAL / SIM / LIT / TARGET / ASSUMPTION.
If cornered on a number you don't remember: **"the CSV is authoritative — `code/results/`"** and move on.

---

**Q. Is 0.685 the real recall?**
At an **assumed 15% cloud**. The honest figure is a **band: 0.40–0.82 over 0–50% cloud** — cloud
fraction moves recall more than any other quantity, and the Airbus set (0.4% cloud) can't settle it.
We **never quote 0.685 bare**. `source: wp6_real_table.csv, report 11`.

**Q. Does this actually run on a satellite?**
Not yet — it's a **demo, not a prototype**. Every timing is a laptop RTX 5060, so "runs onboard" is
a **TARGET** (report 22). But it's the *class* of model Φ-sat-2 already flies on a ~1 W Myriad 2 VPU
[LIT]: a 3.15 M-param detector + 47 k-param gate, a few-% duty cycle at 40k tiles/day even on a VPU
6× slower. What a prototype still owes: wall-clock + quantised accuracy on a real board, thermal,
rad-tolerance.

**Q. You lean on SAHI — isn't it expensive?**
We measured it. A seam-cut ship is **usually still detected** (report 13), so **blanket SAHI is a
poor trade**; we argue **selective** slicing. Cost is **1.56× compute**, not 2.25× (report 16). Window
is **768, not the paper's 512** — the result reverses at 512, and we state the window.

**Q. Why Airbus, not DOTA / xView / HRSID?**
The paper permits optical maritime datasets; Airbus is one, **53,195 tiles, leakage-free split**
(a naive split would leak 1,617 groups; ours leaks 0 — report 01). Airbus tiles are already 768²,
so SAHI is evaluated on **synthetic swaths stitched from tiles**. A large-scene optical set would
strengthen B2 — honest limitation, not a hidden one.

**Q. The relay — does it save energy?**
No, and we don't claim it. The relay is a **latency buy, not an energy win**: it costs **~1.8×
comm energy per relayed item** (+34% across the B4 mix, TARGET), and in return **halves worst-case
latency, 11.6 h → 6.2 h** (SIM). Recall and bytes are unchanged — B4 is a reroute (reports 19–21).

**Q. What's the weakest number?**
Small-ship recall (**0.50–0.61** at the 0.25 operating cut) — the detector, not the downlink, is
the residual loss. And `P_tx` (invented, ASSUMPTION), but it only enters E_comm, which is ~9× smaller
than E_proc, so it barely moves anything.

**Q. What's actually novel? YOLO and SAHI are off-the-shelf.**
Correct — those are building blocks. The novelty is **system/decision level**: the per-object
**P0–P3 priority packet** and the **`min(J_direct, J_relay)`** path choice, plus a value-aware
multi-pass scheduler that is **within 0.251% of the exact optimum** (report 08). The paper is a
methodology; Phase 3 is the measurement.

**Q. Why is INT8 not used?**
Measured and **rejected**: 8.5× *slower* on CPU and −2.2 pts on small ships (dynamic ONNX INT8, not
TensorRT). ONNX FP32 is the onboard build. A reported negative result (report 02).

---

**One-line pitch:** *557× less data, 100% of what the satellite still knew, every figure labelled and
every claim we could break, we broke — on real Airbus imagery, measured end to end.*
