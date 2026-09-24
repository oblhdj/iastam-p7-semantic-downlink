# 15 — Re-keying the thumbnail gate onto the learned gate

Script: `../code/scripts/wp14_gate_signal.py` (+ `../code/scripts/wp3_score_tiles.py` to score the
catalogue). Data: `../code/results/wp14_gate_signal.json|.csv`, `wp3_gate_scores_test.csv`.
**REAL** — all 5,320 catalogue tiles scored by the trained gate, 3 seeds, 2 loads.

## The problem

[Report 06](06-byte-budget.md) skips the safety-net thumbnail when the onboard software is
confident a tile is empty. Two opinions had to agree: the **classic** pre-filter reports
`empty_sea`, *and* the detector fired nothing. Thumbnails are ~24% of the byte budget, so this is
a real lever.

But the classic half of that test is the weakest and most expensive component in the pipeline:
**0.645 ship recall** ([report 04](04-onboard-gate.md)) at **56 ms/tile**
([report 12](12-end-to-end-integration.md)). The learned gate reaches **0.988** at **0.15 ms**.
The gating decision was being made by the part we had already proven worst.

## What each signal actually calls "empty"

| signal | tiles called empty | of those, truly empty | ships sitting on them |
|---|---|---|---|
| classic pre-filter | 177 | 91.5% | 15 |
| **learned gate** (threshold 0.126, chosen on val) | **513** | 84.8% | 100 |

The gate is willing to call **2.9× more tiles empty**, at slightly lower precision. That is the
trade: more bytes saved, more ships resting on the detector's opinion alone.

## The result: −3.8% of all bytes, at zero measured recall cost

Same catalogue, same day, same seeds — only the signal changes.

| load | signal | thumbnails skipped | MB offered | ship recall | latency h |
|---|---|---|---|---|---|
| 40,000 | classic | 5,123 | 107.95 | 0.6851 | 4.38 |
| 40,000 | **learned gate** | **10,680** | **103.94** | **0.6851** | 4.38 |
| | *change* | **+5,557** | **−4.01 (−3.7%)** | **±0.0000** | ±0.00 |
| 160,000 | classic | 20,265 | 427.65 | 0.6631 | 4.89 |
| 160,000 | **learned gate** | **42,638** | **411.31** | **0.6631** | 4.89 |
| | *change* | **+22,373** | **−16.34 (−3.8%)** | **±0.0000** | ±0.00 |

Recall is identical to four decimal places at both loads. The gate doubles the number of
thumbnails safely skipped and returns ~3.8% of the entire downlink budget for it.

## Say the caveat as well

**100 ships sit on tiles the gate is willing to call empty, against 15 for the classic filter.**
Measured recall does not move, because the gating rule still requires the detector to have fired
nothing, and in practice those ships are either detected or recovered elsewhere. But the safety
margin is genuinely thinner, and it is thinner in the place the whole thumbnail exists to
protect: tiles where the detector was wrong. If the operating point ever needs tightening, the
threshold is one number (`GATE_EMPTY_THR = 0.126` in `../code/sat7/real_workload.py`) and the
full trade-off curve is in `wp3_gate_tradeoff.csv`.

## What was *not* re-keyed, and why

[Report 07](07-queue-aware-encoding.md)'s control law keys on **unconfirmed candidates** — bright
objects the classic stage localised that the network did not confirm. The learned gate cannot
supply that signal: it is a whole-tile classifier and produces one score, not a set of blob
positions. The two stages need different things from the cheap pass:

* **the gating decision** needs a good *whole-tile verdict* → the learned gate wins outright
* **the escalation signal** needs *localisation* → only the classic filter provides it

So the classic pre-filter cannot simply be deleted; it is still the only source of the WP8
signal. Replacing that too means designing a new escalation signal and **recalibrating the four
fitted parameters** of the control law, which is Phase-3 work, not an input swap. Stated plainly
because "swap one for the other" was the obvious-looking move and it is only half right.

## Recommendation
1. **Adopt the learned gate for thumbnail gating.** −3.8% of total bytes, no measured recall cost,
   and it removes a 0.645-recall component from a safety decision.
2. Keep the classic pre-filter running **only** for the WP8 escalation signal, and note in the
   compute budget that it is now paying for one thing rather than two.
3. Revisit the WP8 signal in Phase 3; a small localising head on the gate would let the classic
   stage be dropped entirely, which [report 13](13-tile-seams-and-sahi.md) shows is worth 56 ms/tile.
