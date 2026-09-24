"""Multi-pass, value-aware downlink scheduling (Contribution A).

Pieces
------
* ``Item``          one thing waiting onboard: metadata (L0), chip (L1), ROI/tile (L2),
                    thumbnail, raw tile, ... with a size, a value and the ships it covers.
* ``Workload``      synthetic onboard output for a day (until the real detector is plugged
                    in). All proportions are ASSUMPTIONS listed in ``WorkloadConfig``.
* Encoders          turn true ships into items: raw bent pipe, Phi-sat-2-style fixed
                    patch, and our confidence-aware level of detail (LoD).
* Policies          Strategy pattern: NoBuffer, FIFO, ValueGreedy (ours).
* ``simulate``      replays arrivals + contact windows and reports ground-side metrics.

Value model: an item's value is the sum of the weights of the ships it reveals
(normal ship = 1, dark vessel = ``dark_weight``). A progressive item (JPEG2000 /
CCSDS-122 style) may be cut: sending a fraction f keeps value * sqrt(f)
(diminishing returns). ValueGreedy sends items in decreasing value-per-byte,
with aging so old items are not starved.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .orbit import Pass

# ----------------------------------------------------------------------------- data


@dataclass
class Item:
    id: int
    created_s: float                   # seconds since simulation start
    size: float                        # bytes
    value: float
    kind: str                          # "L0", "L1", "L2", "tile", "thumb", "raw", "patch", "fp"
    ships: tuple[int, ...] = ()        # ids of true ships this item reveals
    progressive: bool = False

    @property
    def density(self) -> float:
        return self.value / self.size if self.size else 0.0


@dataclass
class Ship:
    id: int
    created_s: float
    dark: bool
    confidence: float                  # detector confidence (synthetic, or REAL from WP6)
    coastal: bool
    size_px: float = 0.0               # REAL ship length (WP6); 0 = unknown -> median payload


@dataclass
class WorkloadConfig:
    """Synthetic onboard workload. Tune, and report the values you use."""
    hours: float = 24.0
    tiles_per_day: int = 40_000        # 768x768 tiles imaged over the area of interest
    p_cloud: float = 0.15              # tile contexts (Airbus-like mix)
    p_ships: float = 0.20
    p_coast: float = 0.05              # rest = empty sea
    ships_per_ship_tile: float = 1.8   # mean (>=1)
    ships_per_coast_tile: float = 3.0  # mean
    p_dark: float = 0.10               # ships without AIS
    p_false_positive: float = 0.02     # per empty-sea tile
    conf_alpha: float = 5.0            # detector confidence ~ Beta(alpha, beta)
    conf_beta: float = 2.0
    seed: int = 0


@dataclass
class Workload:
    ships: list[Ship]
    tiles: list[tuple[float, str, tuple[int, ...]]]   # (time, context, ship ids)
    cfg: WorkloadConfig
    false_alarms: list[int] | None = None   # REAL detector false alarms per tile (WP6).
                                            # None -> draw them with cfg.p_false_positive.
    gate_empty: list[bool] | None = None    # onboard "nothing here" verdict per tile (WP7):
                                            # classic pre-filter says empty sea AND the
                                            # detector fired nothing. None -> gate everything on.
    unconfirmed: list[int] | None = None    # per tile (WP7): bright objects the cheap classic
                                            # stage found that the network did NOT confirm.
                                            # High = the network probably missed something.
    source: str = "synthetic"               # "synthetic" | "real detections (WP6)"


def generate_workload(cfg: WorkloadConfig | None = None) -> Workload:
    cfg = cfg or WorkloadConfig()
    rng = np.random.default_rng(cfg.seed)
    n_tiles = int(cfg.tiles_per_day * cfg.hours / 24)
    times = np.sort(rng.uniform(0, cfg.hours * 3600, n_tiles))
    contexts = rng.choice(["cloud", "ships", "coast", "empty"], size=n_tiles,
                          p=[cfg.p_cloud, cfg.p_ships, cfg.p_coast,
                             1 - cfg.p_cloud - cfg.p_ships - cfg.p_coast])
    ships, tiles = [], []
    for t, ctx in zip(times, contexts):
        ids: list[int] = []
        if ctx in ("ships", "coast"):
            mean = cfg.ships_per_ship_tile if ctx == "ships" else cfg.ships_per_coast_tile
            k = 1 + rng.poisson(mean - 1)
            for _ in range(k):
                s = Ship(len(ships), float(t), bool(rng.random() < cfg.p_dark),
                         float(rng.beta(cfg.conf_alpha, cfg.conf_beta)), ctx == "coast")
                ships.append(s)
                ids.append(s.id)
        tiles.append((float(t), str(ctx), tuple(ids)))
    return Workload(ships, tiles, cfg)


# ----------------------------------------------------------------------------- encoders

RAW_TILE_BYTES = 768 * 768 * 3        # 1.77 MB uncompressed RGB tile


@dataclass
class SizeModel:
    """Payload cost of one ship as a function of its length: bytes = a * px^b.

    Fitted on the WP4 measurements (scripts/wp6_fit_size_model.py). Without it every
    ship costs the WP4 median, which hides the very trade-off the scheduler exploits:
    a 300 px cargo ship costs ~30x a 12 px fishing boat at the same level of detail.
    """
    l1_a: float = 137.5                # crop tight q40 : R2 0.76
    l1_b: float = 0.57
    l2_a: float = 43.3                 # crop wake q80  : R2 0.89
    l2_b: float = 1.14
    min_bytes: float = 120.0           # JPEG header floor

    def l1(self, px: float) -> float:
        return max(self.min_bytes, self.l1_a * px ** self.l1_b)

    def l2(self, px: float) -> float:
        return max(self.min_bytes, self.l2_a * px ** self.l2_b)


@dataclass
class LoDConfig:
    """Our confidence-aware level of detail.

    Sizes MEASURED on 551 real Airbus ship crops and 300 tiles (scripts/wp4_measure_lod.py),
    medians; the p90 is several times larger because big ships cost more:
      tight crop q40 899 B | context crop q80 1.5 kB | wake crop q80 2.5 kB
      whole tile q60 42.9 kB | 96 px thumbnail 997 B
    When ``size_model`` is set and the ship's real length is known (WP6), the chip and
    ROI sizes come from the fitted power law instead of these medians.
    """
    l0_bytes: float = 40.0             # lat, lon, length, heading, confidence
    l1_bytes: float = 900.0            # small image chip      (measured: tight crop, q40)
    l2_bytes: float = 2_500.0          # ROI incl. wake        (measured: wake crop, q80)
    coast_tile_bytes: float = 32_420.0 # compressed coastal / port tile. WP7 moved this from
                                       # q60 (42.4 kB, PSNR 40.3 dB) to q40 (32.4 kB, 38.1 dB):
                                       # -24% of the single largest item in the budget, no
                                       # recall cost, +1.2 points under congestion. q30 is
                                       # 25.9 kB / 36.8 dB and tested, but adopting it needs
                                       # a detector-on-recompressed-tiles check first.
    coast_mode: str = "tile"           # "tile"    = whole coastal tile, catches ships the
                                       #             detector missed (62% of the budget)
                                       # "mosaic"  = ROI crops around detected ships only
                                       #             (WP7: 2.3 kB/tile, 14x cheaper, but it
                                       #             gives up the undetected-ship safety net)
                                       # "adaptive"= the level-of-detail idea applied to the
                                       #             tile itself: mosaic normally, whole tile
                                       #             only where the cheap classic stage saw
                                       #             objects the network did not confirm
    coast_mosaic_bytes: float = 2_259.0    # measured: ROI mosaic x1.5 q60, 400 real tiles
    coast_escalate: int = 11           # "adaptive": unconfirmed candidates needed to send the
                                       # whole tile. MEASURED on the real coastal tiles: at 11+
                                       # (50% of them) sit 75% of the ships the network missed,
                                       # while tiles with none hold 5%.
    # --- "queue" mode: the escalation threshold follows how full the buffer is, which is the
    #     one thing the satellite knows for free and we were not using. Generous while the
    #     link can still drain the queue, stingy once it cannot.
    #     Calibrated against the measured per-regime optimum (WP8). Median pressure and the
    #     best fixed policy, per load: 40k -> 0.28, whole tile | 80k -> 0.83, whole tile |
    #     160k -> 1.62, escalate >= 11 | 320k -> 3.27, mosaic only. So the law must stay at
    #     0 up to ~0.9, pass through ~11 at ~1.6, and saturate by ~3.5.
    #     Four parameters fitted to five measured points: see the held-out check in WP8 §3.
    esc_min: int = 0                   # plenty of capacity: every coastal tile goes whole
    esc_max: int = 40                  # hopeless backlog: effectively mosaic-only
    pressure_lo: float = 0.9           # queued bytes / capacity before the queue can drain
    pressure_hi: float = 3.5
    thumb_bytes: float = 1_000.0       # safety-net thumbnail  (measured: 96 px q60 984 B;
                                       # WP7: 128 px q30 is 1,037 B and *more* ships stay
                                       # visible, 0.643 vs 0.615 -- resolution beats quality
                                       # here because the JPEG header floor is ~700 B)
    thumb_value: float = 0.01
    # --- WP2: the detector is UNDERCONFIDENT (mean score 0.352, actually right 0.408 of the
    #     time), so a raw 0.9 was far stricter than "90% certain" and the cheap 40 B rung fired
    #     on only 3.1% of predictions. Isotonic calibration (ECE 0.0878 -> 0.0099, fitted on val,
    #     evaluated on test) puts the score that really means P=0.9 at **0.670**, where the rung
    #     serves 21.7%. Measured strictly better at every load (WP10 sweep): at 160k tiles/day
    #     +1.4 pts recall, -8.1% bytes, -0.13 h latency. Adopted 24 Sept 2026; was 0.9.
    conf_high: float = 0.670
    # NOTE: every script overrides this with the onboard detection threshold (0.25) -- the
    # bottom of the ladder IS "did the detector see it". This 0.4 default is therefore dead
    # in practice; do not read it as the operating point. See WP2 report section 4.
    conf_low: float = 0.4
    dark_weight: float = 5.0
    size_model: SizeModel | None = None
    # --- WP9: a missing AIS match must raise PRIORITY and nothing else.
    #     "replace"   (the original) swapped the chip for an L2 ROI+wake, so the flag changed
    #                 the payload as well as the value. Because L2 grows as px^1.14 and L1 as
    #                 px^0.57, above 128 px the extra bytes outweighed the 5x weight and a
    #                 dark vessel ranked BELOW the same ship with a clean AIS track -- the
    #                 opposite of the intent, on 13.6% of real ships (the largest ones).
    #     "decoupled" (now the default) gives a dark vessel exactly the same base product as
    #                 any other ship, so the weight alone does the prioritising and the claim
    #                 holds at every size by construction. The ROI+wake becomes a separate,
    #                 lower-priority follow-up, sent when there is capacity for it.
    dark_mode: str = "decoupled"
    dark_wake_value: float = 1.0       # ASSUMPTION: the wake/context is worth about one
                                       # ordinary ship of extra evidence, on top of the
                                       # report that the base chip already delivered
    # --- WP7: measured on the real workload, 86% of our bytes are coastal tiles (62%) and
    #     blanket thumbnails (24%), not ship information (13%). These two knobs control
    #     that spend; both cost recall, so both are swept rather than assumed.
    gate_thumbnails: bool = True       # skip the thumbnail when the onboard gate calls the
                                       # tile empty (needs Workload.gate_empty; a synthetic
                                       # workload has none, so nothing changes there).
                                       # WP7: -3.6% bytes, no measured recall cost. Small
                                       # because the classic pre-filter clears only 21% of
                                       # empty tiles -- a learned gate (WP3) would raise it.
    thumb_audit: float = 0.02          # fraction of gated-empty tiles thumbnailed anyway,
                                       # so silent domain shift stays detectable


def _w(ship: Ship, lod: LoDConfig) -> float:
    return lod.dark_weight if ship.dark else 1.0


def _l1(ship: Ship, lod: LoDConfig) -> float:
    """Bytes for a level-1 chip of this ship (real length when we have it)."""
    return lod.size_model.l1(ship.size_px) if (lod.size_model and ship.size_px > 0) else lod.l1_bytes


def _l2(ship: Ship, lod: LoDConfig) -> float:
    """Bytes for a level-2 ROI (ship + wake) of this ship."""
    return lod.size_model.l2(ship.size_px) if (lod.size_model and ship.size_px > 0) else lod.l2_bytes


class _Ids:
    def __init__(self):
        self.n = 0

    def __call__(self) -> int:
        self.n += 1
        return self.n


def encode_raw(wl: Workload, lod: LoDConfig | None = None) -> list[Item]:
    """Bent pipe: every non-cloud tile downlinked raw."""
    lod = lod or LoDConfig()
    nid, ships = _Ids(), wl.ships
    return [Item(nid(), t, RAW_TILE_BYTES, sum(_w(ships[i], lod) for i in ids), "raw", ids)
            for t, ctx, ids in wl.tiles if ctx != "cloud"]


def _n_false_alarms(wl: Workload, idx: int, ctx: str, rng) -> int:
    """How many false detections this tile produces.

    REAL workload: the measured count for this very tile (WP6). Synthetic: a coin
    flip with ``p_false_positive``, empty-sea tiles only.
    """
    if wl.false_alarms is not None:
        return wl.false_alarms[idx]
    return int(ctx == "empty" and rng.random() < wl.cfg.p_false_positive)


def encode_fixed_patch(wl: Workload, lod: LoDConfig | None = None, patch_bytes: float = 1_000.0,
                       coastal: bool = True) -> list[Item]:
    """Phi-sat-2 style: one fixed-size patch per detected ship (conf >= low).

    ``coastal=True`` (default) runs the baseline over coastal scenes too, which is the
    fair comparison: nothing in a fixed-patch design forbids coastal imaging, so denying
    it there would hand us a third of the ships for free. With ``coastal=False`` it images
    open ocean only, reproducing the weaker baseline used in the interim report -- the
    self-criticism table calls that out as weakness #2, and this flag is the fix.
    """
    lod = lod or LoDConfig()
    nid, items, rng = _Ids(), [], np.random.default_rng(wl.cfg.seed + 1)
    ship_ctx = ("ships", "coast") if coastal else ("ships",)
    for idx, (t, ctx, ids) in enumerate(wl.tiles):
        if ctx in ship_ctx:
            for i in ids:
                s = wl.ships[i]
                if s.confidence >= lod.conf_low:
                    items.append(Item(nid(), t, patch_bytes, _w(s, lod), "patch", (i,)))
        if ctx != "coast" or coastal:   # if it images the coast, it pays for coastal false alarms
            for _ in range(_n_false_alarms(wl, idx, ctx, rng)):
                items.append(Item(nid(), t, patch_bytes, 0.0, "fp"))
    return items


def encode_lod(wl: Workload, lod: LoDConfig | None = None) -> list[Item]:
    """Ours: confidence-aware level of detail + coastal tiles + thumbnails.

    Batch form: every tile is encoded with the same settings, which is what an offline
    comparison needs. ``simulate_online`` calls ``encode_tile`` directly instead, so the
    coastal decision can react to how full the buffer is at that moment.
    """
    lod = lod or LoDConfig()
    nid, items, rng = _Ids(), [], np.random.default_rng(wl.cfg.seed + 1)
    for idx in range(len(wl.tiles)):
        items += encode_tile(wl, idx, lod, nid, rng)
    return items


def encode_tile(wl: Workload, idx: int, lod: LoDConfig, nid: "_Ids", rng,
                escalate: int | None = None) -> list[Item]:
    """Everything the satellite produces for one tile.

    ``escalate`` overrides ``lod.coast_escalate`` for this tile, which is how the
    queue-aware policy tightens or relaxes the coastal decision as the buffer fills.
    """
    items: list[Item] = []
    t, ctx, ids = wl.tiles[idx]
    if ctx == "cloud":
        return items
    # The thumbnail is the safety net against a detector miss, so it is only worth
    # skipping where the onboard gate is confident nothing is there -- and even then a
    # small audit sample still goes down, to catch silent domain shift.
    gated = (lod.gate_thumbnails and wl.gate_empty is not None and wl.gate_empty[idx]
             and rng.random() >= lod.thumb_audit)
    if not gated:
        items.append(Item(nid(), t, lod.thumb_bytes, lod.thumb_value, "thumb"))
    if ctx == "coast":           # detectors struggle here (recall 0.784 vs 0.895 at sea)
        mode = lod.coast_mode
        if mode in ("adaptive", "queue"):
            # escalate to the whole tile only where the cheap stage disagrees with the
            # network; everywhere else the detections are all there is to say
            thr = lod.coast_escalate if escalate is None else escalate
            unconf = wl.unconfirmed[idx] if wl.unconfirmed is not None else 0
            mode = "tile" if unconf >= thr else "mosaic"
        if mode == "mosaic":
            # only the detected ships: 14x cheaper, but a ship the detector missed is
            # now as lost on the coast as it is at sea
            seen = tuple(i for i in ids if wl.ships[i].confidence >= lod.conf_low)
            if seen:
                items.append(Item(nid(), t, lod.coast_mosaic_bytes,
                                  sum(_w(wl.ships[i], lod) for i in seen), "mosaic", seen,
                                  progressive=True))
        else:                    # send the tile itself: the only thing that recovers a
                                 # ship the detector never saw
            items.append(Item(nid(), t, lod.coast_tile_bytes,
                              sum(_w(wl.ships[i], lod) for i in ids), "tile", ids,
                              progressive=True))
    elif ctx == "ships":
        for i in ids:
            s = wl.ships[i]
            if s.confidence < lod.conf_low:
                continue         # missed by the detector (only the thumbnail remains)
            if s.dark and lod.dark_mode == "replace":
                # original behaviour, kept for comparison: the flag swaps the product
                items.append(Item(nid(), t, _l2(s, lod), _w(s, lod), "L2", (i,), progressive=True))
                continue
            # base product: chosen by confidence and size only, never by the AIS flag, so a
            # dark vessel always costs the same bytes and carries 5x the value -- strictly
            # higher value-per-byte than the same ship with a clean track, at every size
            if s.confidence > lod.conf_high:
                items.append(Item(nid(), t, lod.l0_bytes, _w(s, lod), "L0", (i,)))
            else:
                items.append(Item(nid(), t, _l1(s, lod), _w(s, lod), "L1", (i,)))
            if s.dark:
                # the ROI + wake is now extra evidence rather than a substitute: lower
                # priority than the report itself, and truncatable if the pass runs out
                items.append(Item(nid(), t, _l2(s, lod), lod.dark_wake_value, "L2", (i,),
                                  progressive=True))
    if ctx != "coast":           # on coastal tiles a false alarm rides inside the tile
        for _ in range(_n_false_alarms(wl, idx, ctx, rng)):
            items.append(Item(nid(), t, lod.l1_bytes, 0.0, "fp"))
    return items


def escalation_for_pressure(lod: LoDConfig, pressure: float) -> int:
    """Turn buffer pressure into a coastal escalation threshold.

    ``pressure`` = bytes already queued / link capacity left before the queue could drain.
    Below ``pressure_lo`` the link will clear the backlog anyway, so we escalate almost
    every coastal tile (threshold -> ``esc_min``) and keep the safety net. Above
    ``pressure_hi`` the bytes are not going to fit, so whole tiles would only crowd out
    ship reports, and the threshold rises to ``esc_max``. Linear in between.
    """
    lo, hi = lod.pressure_lo, lod.pressure_hi
    f = 0.0 if hi <= lo else (pressure - lo) / (hi - lo)
    f = min(1.0, max(0.0, f))
    return int(round(lod.esc_min + f * (lod.esc_max - lod.esc_min)))


# ----------------------------------------------------------------------------- policies


@dataclass
class Sent:
    item: Item
    fraction: float
    delivered_s: float


class Policy(ABC):
    """Strategy interface: choose what to send during one pass."""
    name = "policy"
    buffers = True                     # keep unsent items for later passes?

    @abstractmethod
    def order(self, queue: list[Item], now_s: float) -> list[Item]:
        """Return the queue in sending order."""

    truncate = False                   # may cut progressive items to fill the pass?

    def plan(self, queue: list[Item], capacity: float, now_s: float) -> list[tuple[Item, float]]:
        out, left = [], capacity
        for it in self.order(queue, now_s):
            if it.size <= left:
                out.append((it, 1.0))
                left -= it.size
            elif self.truncate and it.progressive and left >= 0.1 * it.size:
                out.append((it, left / it.size))
                left = 0.0
            elif self.stop_when_full:
                break
            if left <= 0:
                break
        return out

    stop_when_full = False


class FIFO(Policy):
    """Oldest first, whole items only, head-of-line blocking (classic store-and-forward)."""
    name = "FIFO"
    stop_when_full = True

    def order(self, queue, now_s):
        return sorted(queue, key=lambda it: it.created_s)


class NoBuffer(FIFO):
    """Only data captured since the previous pass; the rest is lost (no buffering)."""
    name = "No buffering"
    buffers = False


class ValueGreedy(Policy):
    """Ours: highest value-per-byte first, with aging, progressive truncation."""
    name = "Value-greedy (ours)"
    truncate = True

    def __init__(self, aging_per_hour: float = 0.5):
        self.aging = aging_per_hour

    def order(self, queue, now_s):
        def key(it: Item) -> float:
            age_h = max(0.0, now_s - it.created_s) / 3600
            return it.density * (1 + self.aging * age_h)
        return sorted(queue, key=key, reverse=True)


# ----------------------------------------------------------------------------- simulation


@dataclass
class Result:
    policy: str
    encoder: str
    sent: list[Sent] = field(default_factory=list)
    dropped_storage: int = 0
    bytes_sent: float = 0.0
    bytes_offered: float = 0.0
    capacity: float = 0.0


def simulate(items: list[Item], passes: list[Pass], start: datetime, policy: Policy,
             storage_bytes: float = 8e9, encoder: str = "") -> Result:
    """Replay item arrivals and passes; return what reached the ground and when."""
    items = sorted(items, key=lambda it: it.created_s)
    res = Result(policy.name, encoder, bytes_offered=sum(i.size for i in items),
                 capacity=sum(p.capacity_bytes for p in passes))
    queue: list[Item] = []
    k = 0
    for p in passes:
        rise_s = (p.rise - start).total_seconds()
        dur_s = p.duration_s
        if not policy.buffers:
            queue = []                                    # everything older is lost
        while k < len(items) and items[k].created_s <= rise_s:
            queue.append(items[k])
            k += 1
        # onboard storage limit: drop the newest (FIFO) / least dense (greedy) items
        total = sum(i.size for i in queue)
        if total > storage_bytes:
            victims = sorted(queue, key=(lambda i: -i.created_s) if not policy.truncate else (lambda i: i.density))
            keep = set(id(i) for i in queue)
            for v in victims:
                if total <= storage_bytes:
                    break
                keep.discard(id(v))
                total -= v.size
                res.dropped_storage += 1
            queue = [i for i in queue if id(i) in keep]
        plan = policy.plan(queue, p.capacity_bytes, rise_s)
        cum = 0.0
        sent_ids = set()
        for it, frac in plan:
            cum += it.size * frac
            res.sent.append(Sent(it, frac, rise_s + dur_s * min(1.0, cum / p.capacity_bytes)))
            res.bytes_sent += it.size * frac
            sent_ids.add(id(it))
        queue = [i for i in queue if id(i) not in sent_ids]
    return res


def simulate_online(wl: Workload, lod: LoDConfig, passes: list[Pass], start: datetime,
                    policy: Policy, storage_bytes: float = 8e9, encoder: str = "",
                    horizon_h: float = 12.0) -> tuple[Result, dict]:
    """Encode and schedule together, the way the satellite actually works.

    ``simulate`` takes a finished list of items: every tile was compressed before anything
    was known about the downlink. Onboard that is not true -- the buffer is right there when
    the compressor runs. This loop walks the tiles in capture order, measures the pressure
    (bytes queued against the link capacity coming up in the next ``horizon_h`` hours, which
    the satellite can read straight off its own ephemeris) and lets ``coast_mode="queue"``
    pick the coastal level of detail from it.

    Returns the usual Result plus a small trace, so the control law can be inspected rather
    than trusted.
    """
    nid, rng = _Ids(), np.random.default_rng(wl.cfg.seed + 1)
    order = sorted(range(len(wl.tiles)), key=lambda i: wl.tiles[i][0])
    res = Result(policy.name, encoder, capacity=sum(p.capacity_bytes for p in passes))
    queue: list[Item] = []
    queued_bytes = 0.0
    pass_times = [((p.rise - start).total_seconds(), p) for p in passes]
    pi = 0
    trace = {"pressure": [], "escalate": [], "queued_MB": []}

    def capacity_ahead(now_s: float) -> float:
        """Link capacity the satellite can count on soon. It knows its own ephemeris, so
        when the horizon happens to contain no pass we fall back to the next one rather
        than pretending the link is gone -- otherwise every tile captured late in the
        window would see infinite pressure and get the cheapest treatment for no reason."""
        end = now_s + horizon_h * 3600
        cap = sum(p.capacity_bytes for rise_s, p in pass_times if now_s <= rise_s <= end)
        if cap > 0:
            return cap
        later = [p.capacity_bytes for rise_s, p in pass_times if rise_s >= now_s]
        return later[0] if later else 0.0

    for idx in order:
        t = wl.tiles[idx][0]
        # ---- any pass that falls due before this tile is captured
        while pi < len(pass_times) and pass_times[pi][0] <= t:
            rise_s, p = pass_times[pi]
            queue, queued_bytes = _run_pass(res, policy, queue, p, rise_s, storage_bytes)
            pi += 1
        # ---- encode this tile, knowing how full the buffer is
        esc = None
        if lod.coast_mode == "queue":
            cap = capacity_ahead(t)
            pressure = queued_bytes / cap if cap > 0 else float("inf")
            esc = escalation_for_pressure(lod, pressure)
            if wl.tiles[idx][1] == "coast":
                trace["pressure"].append(pressure)
                trace["escalate"].append(esc)
                trace["queued_MB"].append(queued_bytes / 1e6)
        new = encode_tile(wl, idx, lod, nid, rng, escalate=esc)
        queue += new
        queued_bytes += sum(i.size for i in new)
        res.bytes_offered += sum(i.size for i in new)

    for rise_s, p in pass_times[pi:]:
        queue, queued_bytes = _run_pass(res, policy, queue, p, rise_s, storage_bytes)
    return res, trace


def _run_pass(res: Result, policy: Policy, queue: list[Item], p: Pass, rise_s: float,
              storage_bytes: float) -> tuple[list[Item], float]:
    """One contact window: enforce storage, send what the policy picks, keep the rest.

    A non-buffering policy discards its leftovers at the *end* of the window, not the
    start. Online, the items captured since the previous pass are already in the queue by
    the time the pass comes round, so clearing on entry would throw away exactly the data
    that is supposed to survive -- which is what ``simulate`` avoids by clearing before it
    adds new arrivals.
    """
    total = sum(i.size for i in queue)
    if total > storage_bytes:
        victims = sorted(queue, key=(lambda i: -i.created_s) if not policy.truncate
                         else (lambda i: i.density))
        keep = set(id(i) for i in queue)
        for v in victims:
            if total <= storage_bytes:
                break
            keep.discard(id(v))
            total -= v.size
            res.dropped_storage += 1
        queue = [i for i in queue if id(i) in keep]
    plan = policy.plan(queue, p.capacity_bytes, rise_s)
    cum = 0.0
    sent_ids = set()
    for it, frac in plan:
        cum += it.size * frac
        res.sent.append(Sent(it, frac, rise_s + p.duration_s * min(1.0, cum / p.capacity_bytes)))
        res.bytes_sent += it.size * frac
        sent_ids.add(id(it))
    queue = [i for i in queue if id(i) not in sent_ids]
    if not policy.buffers:
        queue = []                     # nothing unsent survives to the next contact
    return queue, sum(i.size for i in queue)


def metrics(res: Result, wl: Workload, lod: LoDConfig | None = None) -> dict:
    """Ground-side metrics: recall, dark-vessel recall, value, latency, bytes."""
    lod = lod or LoDConfig()
    first_seen: dict[int, float] = {}
    value = 0.0
    for s in res.sent:
        value += s.item.value * (math.sqrt(s.fraction) if s.item.progressive else 1.0)
        for sid in s.item.ships:
            if sid not in first_seen or s.delivered_s < first_seen[sid]:
                first_seen[sid] = s.delivered_s
    ships = wl.ships
    dark = [s for s in ships if s.dark]
    lat_h = np.array([(first_seen[s.id] - s.created_s) / 3600 for s in ships if s.id in first_seen])
    lat_dark = np.array([(first_seen[s.id] - s.created_s) / 3600 for s in dark if s.id in first_seen])
    total_value = sum(_w(s, lod) for s in ships)
    pct = lambda a, q: float(np.percentile(a, q)) if a.size else float("nan")
    return {
        "encoder": res.encoder,
        "policy": res.policy,
        "ship_recall": len(first_seen) / len(ships) if ships else float("nan"),
        "dark_recall": sum(s.id in first_seen for s in dark) / len(dark) if dark else float("nan"),
        "value_frac": value / total_value if total_value else float("nan"),
        "latency_med_h": pct(lat_h, 50),
        "latency_p90_h": pct(lat_h, 90),
        "dark_latency_med_h": pct(lat_dark, 50),
        "MB_sent": res.bytes_sent / 1e6,
        "MB_offered": res.bytes_offered / 1e6,
        "MB_capacity": res.capacity / 1e6,
        "dropped_storage": res.dropped_storage,
    }


def ship_ci95(recall: float, n: int) -> tuple[float, float]:
    """Wilson 95% confidence interval for a recall measured on n ships."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    denom = 1 + z**2 / n
    centre = (recall + z**2 / (2 * n)) / denom
    half = z * math.sqrt(recall * (1 - recall) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)
