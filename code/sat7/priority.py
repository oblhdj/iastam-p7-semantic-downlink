"""Paper Table I semantic priority levels P0-P3, as a drop-in alternative to the LoD ladder.

The accepted paper specifies an explicit four-level priority scheme (Table I):

    P0  No transmission / discard
    P1  Structured semantic metadata only           (lat, lon, length, heading, confidence)
    P2  Metadata + compressed ROI                    (P1 + a crop of the object)
    P3  Metadata + ROI + contextual image            (P2 + the surrounding context / tile)

The repo's own encoder (`sat7.scheduler.encode_lod`) grew a richer but differently-named ladder
(L0/L1/L2/tile/thumbnail). This module expresses the paper's exact P0-P3 scheme instead, and emits
ordinary `sat7.scheduler.Item`s, so the SAME scheduler, `simulate`, optimum bounds and `metrics`
run over it unchanged -- only the encoding differs. Pick the scheme with `encode_semantic(wl,
mode="priority"|"lod")`; both are first-class and directly comparable on one workload.

How a detection's level is chosen (paper section IV "adaptive semantic priority"): by the project's
own confidence-aware principle, not inverted. A *confident* ship needs only its report (P1); an
*uncertain* one earns a visual crop so the ground can verify (P2); a ship *on a coast*, where the
detector is weakest and a miss is most likely, earns the surrounding context too (P3); and anything
below the detector's own operating threshold is not a detection at all -> P0, discarded.

The coastal rule is `PriorityConfig.coast_escalation`. The default "all" escalates EVERY coastal
detection that cleared the cut -- confident ones included -- because the context tile exists to
recover the ships the detector MISSED next to them, not to verify the one it saw (this is what the
code always did and what wp23 measured; an earlier version of this docstring said "uncertain"
only, which was wrong -- audit C3). "uncertain" escalates only the coastal detections below
p1_conf, sending fewer context tiles. A dark (no-AIS) vessel is bumped one level (capped at P3).
This reuses the WP2-calibrated `conf_high=0.670` as the P1/P2 boundary so the two schemes share the
same operating point and the comparison is fair.

Labels (project rule): the *sizes* are the WP4-measured LoD sizes reused as-is (REAL where the WP4
crops back them, the SizeModel power law where a real length is known); the *level thresholds* and
the ROI/context incremental *values* are ASSUMPTION -- invented, physically-motivated, and meant to
be swept exactly like every other invented constant (report 11). This module measures nothing on
its own.

    from sat7.priority import PriorityConfig, encode_semantic
    items = encode_semantic(wl, mode="priority", lod=LoDConfig(size_model=sm))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scheduler import (Item, LoDConfig, Ship, Workload, _Ids, _l1, _l2,
                        _n_false_alarms, _w, coast_tile_size, encode_lod)

# Level names, in the paper's order. P0 emits no item at all (discard).
P0, P1, P2, P3 = 0, 1, 2, 3
LEVEL_NAME = {P0: "P0-discard", P1: "P1-meta", P2: "P2-roi", P3: "P3-context"}


@dataclass
class PriorityConfig:
    """Thresholds and incremental values for the P0-P3 scheme. All ASSUMPTION, all swept.

    The sizes come from the LoDConfig passed to `encode_priority` (the WP4 measurements / SizeModel),
    so this config holds only the *policy*: where the level boundaries sit and how much extra value a
    ROI or a context image is worth beyond the metadata report that already revealed the ship.
    """
    # --- level boundaries (on calibrated confidence; share the LoD operating point) --------------
    p1_conf: float = 0.670     # >= this: confident -> P1 metadata only (== LoDConfig.conf_high)
    # between det_thr and p1_conf: uncertain -> at least P2 (metadata + ROI)
    coast_to_p3: bool = True   # coastal detections escalate to P3 (which ones: coast_escalation)
    coast_escalation: str = "all"   # "all": every coastal detection (the measured default) |
                                    # "uncertain": only coastal detections below p1_conf
    dark_bump: bool = True     # a dark (no-AIS) vessel escalates one level (capped at P3)
    # --- incremental values (value ALREADY credited to the P1 report is the ship's weight) -------
    #     kept small and in step with the LoD extras (thumb_value, dark_wake_value) so value_frac
    #     stays recall-aligned and the two schemes are comparable under sat7.scheduler.metrics.
    roi_value: float = 1.0     # a verified crop is worth ~one ship of extra evidence
    context_value: float = 0.5 # the surrounding context adds less again (diminishing returns)
    # --- coastal tile context: the P3 "contextual image" that recovers a detector miss ----------
    #     mirrors the LoD coastal tile -- on a coast the context is the whole tile, the only thing
    #     that can reveal a ship the detector never fired on.
    coast_context_tile: bool = True
    # --- what to do with a tile the onboard gate calls empty ------------------------------------
    gate_discard: bool = True  # honour Workload.gate_empty: a gated-empty tile is pure P0 (no audit
                               # thumbnail -- that is a LoD safety net the P-scheme replaces with P3
                               # context on uncertain real detections)


def classify(ship: Ship, ctx: str, cfg: PriorityConfig, lod: LoDConfig) -> int:
    """Priority level P0-P3 for one detection (paper Table I).

    `ctx` is the tile context ("ships" | "coast" | ...). A detection below the detector's operating
    threshold (`lod.conf_low`, overridden to 0.25 onboard) never fired, so it is P0.
    """
    if ship.confidence < lod.conf_low:
        return P0
    level = P1 if ship.confidence >= cfg.p1_conf else P2
    if cfg.coast_to_p3 and ctx == "coast" and (cfg.coast_escalation == "all" or level == P2):
        level = P3
    if cfg.dark_bump and ship.dark:
        level = min(P3, level + 1)
    return level


def encode_priority(wl: Workload, lod: LoDConfig | None = None,
                    cfg: PriorityConfig | None = None) -> list[Item]:
    """Encode a whole workload under the paper's P0-P3 scheme.

    Emits `sat7.scheduler.Item`s the existing scheduler/simulate/metrics consume unchanged:
      * P1 -> one metadata item (size `lod.l0_bytes`, value = ship weight, not progressive)
      * P2 -> P1 plus a ROI crop (size = SizeModel/WP4 ROI, `cfg.roi_value`, progressive)
      * P3 -> P2 plus a contextual image (per-ship context crop, or the whole tile on a coast)
      * P0 -> nothing (discarded onboard)
    False alarms the detector raised are sent as zero-value P1 reports: they cost metadata bytes
    (the ground cannot know they are false until it looks), which keeps byte accounting honest.
    """
    lod = lod or LoDConfig()
    cfg = cfg or PriorityConfig()
    nid, items, rng = _Ids(), [], np.random.default_rng(wl.cfg.seed + 1)
    for idx in range(len(wl.tiles)):
        items += encode_tile_priority(wl, idx, lod, cfg, nid, rng)
    return items


def encode_tile_priority(wl: Workload, idx: int, lod: LoDConfig, cfg: PriorityConfig,
                         nid: "_Ids", rng) -> list[Item]:
    """Everything the P-scheme produces for one tile."""
    items: list[Item] = []
    t, ctx, ids = wl.tiles[idx]
    if ctx == "cloud":
        return items                                   # cloud tile discarded upstream (P0 by context)
    gated = (cfg.gate_discard and wl.gate_empty is not None and wl.gate_empty[idx])

    coastal_context_ids: list[int] = []
    if ctx in ("ships", "coast"):
        for i in ids:
            s = wl.ships[i]
            level = classify(s, ctx, cfg, lod)
            if level == P0:
                continue
            w = _w(s, lod)
            # P1: the structured metadata report (reveals the ship; carries its full weight)
            items.append(Item(nid(), t, lod.l0_bytes, w, "P1", (i,)))
            # off a coast, a dark vessel escalated to P3 gets the L2 (ROI + wake) crop as its ROI --
            # and the P3 context is that same L2 footprint. Sending both shipped identical pixels
            # twice (0.77 MB of the 49.9 MB day, measured 9 Oct 2026): one item carries both now.
            wake_is_context = s.dark and level >= P3 and not (ctx == "coast" and cfg.coast_context_tile)
            if level >= P2:
                # P2: a compressed ROI crop. L1 (tight chip) for a plain ship, L2 (ROI + wake) when
                # we are escalating for a dark vessel -- the extra evidence the dark flag warrants.
                roi_bytes = _l2(s, lod) if (s.dark and level >= P3) else _l1(s, lod)
                value = cfg.roi_value + (cfg.context_value if wake_is_context else 0.0)
                items.append(Item(nid(), t, roi_bytes, value, "P2", (i,), progressive=True))
            if level >= P3 and not wake_is_context:
                if ctx == "coast" and cfg.coast_context_tile:
                    coastal_context_ids.append(i)       # fold into one tile-wide context below
                else:
                    # P3: a wider contextual crop around this object (L2 footprint reused as its size)
                    items.append(Item(nid(), t, _l2(s, lod), cfg.context_value, "P3", (i,),
                                      progressive=True))
    # P3 context on a coast = the whole tile, the only thing that recovers a ship the detector
    # missed. One tile-wide item covering every escalated ship, like the LoD coastal tile.
    if ctx == "coast" and cfg.coast_context_tile and coastal_context_ids and not gated:
        items.append(Item(nid(), t, coast_tile_size(wl, idx, lod),
                          cfg.context_value * len(coastal_context_ids), "P3",
                          tuple(coastal_context_ids), progressive=True))
    # false alarms: metadata reports the ground cannot pre-filter -> zero-value P1 cost
    if ctx != "coast":
        for _ in range(_n_false_alarms(wl, idx, ctx, rng)):
            items.append(Item(nid(), t, lod.l0_bytes, 0.0, "P1", ()))
    return items


def level_histogram(wl: Workload, lod: LoDConfig | None = None,
                    cfg: PriorityConfig | None = None) -> dict:
    """Count detections assigned to each P-level -- the headline the P-scheme reports.

    Over real detections this is a measured distribution (SIM-over-REAL: real confidences, assumed
    thresholds); it answers "how much of the day is metadata-only vs needs a ROI vs needs context",
    which is the byte-budget story the paper's Table I frames.
    """
    lod = lod or LoDConfig()
    cfg = cfg or PriorityConfig()
    hist = {P0: 0, P1: 0, P2: 0, P3: 0}
    for t, ctx, ids in wl.tiles:
        if ctx == "cloud":
            continue
        for i in ids:
            hist[classify(wl.ships[i], ctx, cfg, lod)] += 1
    return {LEVEL_NAME[k]: v for k, v in hist.items()}


def encode_semantic(wl: Workload, mode: str = "lod", lod: LoDConfig | None = None,
                    cfg: PriorityConfig | None = None) -> list[Item]:
    """Select the semantic encoder: the repo's LoD ladder, or the paper's P0-P3 scheme.

    One switch so the B0-B4 campaign and every sweep can run either scheme on the same workload and
    compare them head to head under the identical scheduler + metrics.
    """
    if mode == "lod":
        return encode_lod(wl, lod)
    if mode == "priority":
        return encode_priority(wl, lod, cfg)
    raise ValueError(f"mode must be 'lod' or 'priority', got {mode!r}")


if __name__ == "__main__":
    # Self-check without any real data: a tiny hand-built workload exercises every level.
    from .scheduler import WorkloadConfig

    ships = [
        Ship(0, 0.0, dark=False, confidence=0.90, coastal=False),   # confident      -> P1
        Ship(1, 0.0, dark=False, confidence=0.40, coastal=False),   # uncertain      -> P2
        Ship(2, 0.0, dark=False, confidence=0.40, coastal=True),    # uncertain+coast-> P3
        Ship(3, 0.0, dark=True,  confidence=0.90, coastal=False),   # confident+dark -> P2 (bump)
        Ship(4, 0.0, dark=False, confidence=0.10, coastal=False),   # below thr      -> P0
    ]
    tiles = [(0.0, "ships", (0, 1, 3, 4)), (0.0, "coast", (2,))]
    wl = Workload(ships, tiles, WorkloadConfig())
    lod = LoDConfig(conf_low=0.25)
    cfg = PriorityConfig()
    want = {0: P1, 1: P2, 2: P3, 3: P2, 4: P0}
    for i, lv in want.items():
        got = classify(ships[i], "coast" if i == 2 else "ships", cfg, lod)
        assert got == lv, f"ship {i}: got {LEVEL_NAME[got]}, want {LEVEL_NAME[lv]}"
    print("classify levels OK:", {i: LEVEL_NAME[want[i]] for i in want})
    items = encode_priority(wl, lod, cfg)
    kinds = sorted(it.kind for it in items)
    print("emitted item kinds:", kinds)
    assert "P1" in kinds and "P2" in kinds and "P3" in kinds
    # the P0 ship (id 4) must appear in NO item
    assert all(4 not in it.ships for it in items), "P0 ship was transmitted"
    print("level histogram:", level_histogram(wl, lod, cfg))
    print("priority self-check OK")
