"""Two-path downlink simulator [SIM]: direct (primary -> ground) and relay (primary -> relay -> ground).

SIM. Nothing here is a real satellite connection. Orbits are propagated (SGP4, sat7.orbit), contact
windows are computed from geometry, link rates come from a stated link-budget model and powers are
ASSUMPTIONS. Every result, summary line and JSON block this module emits carries the label "SIM".

What already existed, and is kept:
  * sat7.orbit.find_passes   -- SGP4 propagation of the primary and of the relay, ground-station
                                visibility above a minimum elevation, and each pass's capacity as the
                                integral of a range-dependent rate (Shannon, 1/d^2, capped, x efficiency)
  * sat7.relay.isl_windows   -- line-of-sight windows between the two propagated satellites
  * sat7.scheduler           -- the value-greedy / FIFO policies that decide what a pass carries
  * sat7.relay.route_item    -- a per-item min(J) path choice from window START times only: no data
                                volume, no rate, no capacity (audit E3). Kept for report 20's numbers.

What this module adds: the two paths as a store-and-forward simulation in which those windows are
the ONLY times a link exists and every link has a finite capacity:

    direct   primary --(ground pass)--> ground station
    relay    primary --(ISL window)--> relay satellite --(relay's ground pass)--> ground station

How the propagated windows drive routing. The primary knows its ephemeris, so at the start of each
ISL window it knows (a) its own upcoming ground passes and their capacities, (b) this ISL window's
capacity, (c) the relay's upcoming ground passes. For each queued item, in the scheduler's order,
it estimates the delivery time on each path FROM THE DATA VOLUME AHEAD OF IT AND THE LINK RATE:

    t_direct = rise of the first primary pass with room + (bytes queued ahead + own) * 8 / R_ground
    t_relay  = ISL start + (bytes ahead on the ISL + own) * 8 / R_isl           (stage 1: ISL)
               -> rise of the first relay pass after that with room
               + (bytes reserved ahead + own) * 8 / R_ground_relay              (stage 2: ground)

and picks min J = lam_E * E + lam_T * T, after disqualifying a path that would miss the item's
deadline. Two different rates are involved and they are not interchangeable:

  * WHEN an item lands uses the rate this payload effectively gets: the pass capacity it was given
    (link share included) over the pass duration -- other traffic is interleaved with ours.
  * HOW LONG the transmitter is keyed for it, and so the energy, uses the PHYSICAL link rate (the
    whole pass capacity / duration; coding overhead included, share excluded): a 25% share sends
    our bytes at the full rate for a quarter of the time. This is wp17's convention.

    tx_time  = bytes * 8 / R_physical                    (per stage)
    E_direct = P_tx  * tx_time(ground)
    E_relay  = P_isl * tx_time(ISL)  +  P_tx * tx_time(relay's ground pass)

Comparable assumptions: both satellites use the same ground link model, transmit power and link
share unless a relay-specific share is set; the ISL has its own rate, power and overhead. The relay
never detects anything: it is a queue of opaque, already-encoded items that it forwards. Raw
imagery is not an eligible relay cargo (`relay_kinds`).

With the relay disabled this reduces to sat7.scheduler.simulate exactly (same passes, same policy,
same delivered times) -- the regression gate in tests and in scripts/wp27.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np

from .orbit import (LINK_PRESETS, GroundStation, LinkConfig, OrbitConfig, Pass, find_passes,
                    make_satellite)
from .relay import ISLWindow, isl_windows
from .scheduler import Item, Policy, Result, Sent

LABEL = "SIM"
SEMANTIC_KINDS = ("P1", "P2", "P3", "L0", "L1", "L2", "tile", "thumb", "mosaic", "patch", "fp")


@dataclass
class CommsConfig:
    """Every knob of the two paths. Rates and powers are ASSUMPTIONS (reports 17 / 19), swept in wp27."""
    # ---- ground link: the SAME model for the primary and the relay (comparable assumptions)
    link: LinkConfig = field(default_factory=lambda: LINK_PRESETS["cubesat_sband"])
    link_share: float = 0.25               # share of each primary ground pass given to this payload
    relay_link_share: float | None = None  # share of each relay ground pass given to forwarded data
                                           # (None = the primary's share)
    p_tx_W: float = 15.0                   # ground transmitter, both satellites (report 17)
    # ---- inter-satellite link
    isl_rate_bps: float = 2.53e6           # raw ISL rate (report 19)
    isl_efficiency: float = 0.8            # coding / protocol overhead, as LinkConfig.efficiency
    isl_share: float = 1.0                 # share of each ISL window given to this payload
    p_isl_W: float = 12.0                  # ISL transmitter (report 19)
    isl_setup_s: float = 0.0               # acquisition / pointing time lost at the start of a window
    isl_grazing_km: float = 100.0          # line of sight must clear Earth by this much
    isl_max_range_km: float | None = None  # optional range limit
    relay_raan_deg: float = 90.0           # the relay's orbital plane (complementary coverage)
    # ---- availability: probability that a geometrically visible window is usable (outage, weather)
    ground_availability: float = 1.0
    isl_availability: float = 1.0
    availability_seed: int = 0
    # ---- routing
    relay_enabled: bool = True
    lam_T: float = 1.0 / 3600              # J = lam_E * E[J] + lam_T * T[s]
    lam_E: float = 0.0
    deadline_s: float | None = None        # latency constraint applied to every item (None = none)
    relay_kinds: tuple = SEMANTIC_KINDS    # what the relay may forward: processed products only
    relay_storage_bytes: float = 8e9

    @property
    def relay_share(self) -> float:
        return self.link_share if self.relay_link_share is None else self.relay_link_share

    @property
    def isl_effective_bps(self) -> float:
        return self.isl_rate_bps * self.isl_efficiency * self.isl_share


@dataclass
class Links:
    """The contact plan [SIM]: when each link exists and how many bytes it can carry."""
    primary_passes: list[Pass]             # capacity already scaled by primary_share
    isl: list[ISLWindow]
    relay_passes: list[Pass]               # capacity already scaled by relay_share
    primary_share: float = 1.0             # the shares the capacities were scaled by, so the
    relay_share: float = 1.0               # physical link rate can be recovered for tx time / energy
    label: str = LABEL


def build_links(cfg: CommsConfig, start: datetime, hours: float = 36.0,
                station: GroundStation | None = None, ts=None) -> Links:
    """Propagate both satellites and derive every contact window [SIM] (sat7.orbit / sat7.relay)."""
    from skyfield.api import load
    ts = ts or load.timescale()
    station = station or GroundStation()
    primary = make_satellite(OrbitConfig(epoch=start), ts=ts)
    relay = make_satellite(OrbitConfig(epoch=start, raan_deg=cfg.relay_raan_deg), ts=ts)
    rng = np.random.default_rng(cfg.availability_seed)

    def usable(windows, p):
        return [w for w in windows if p >= 1.0 or rng.random() < p]
    pp = usable(find_passes(primary, station, start, hours, cfg.link, ts=ts), cfg.ground_availability)
    rp = usable(find_passes(relay, station, start, hours, cfg.link, ts=ts), cfg.ground_availability)
    isl = usable(isl_windows(primary, relay, start, hours, grazing_km=cfg.isl_grazing_km,
                             max_range_km=cfg.isl_max_range_km, ts=ts), cfg.isl_availability)
    return Links([replace(p, capacity_bytes=p.capacity_bytes * cfg.link_share) for p in pp], isl,
                 [replace(p, capacity_bytes=p.capacity_bytes * cfg.relay_share) for p in rp],
                 primary_share=cfg.link_share, relay_share=cfg.relay_share)


def pass_rate_bps(p: Pass, share: float = 1.0) -> float:
    """Mean rate of a ground pass, from what sat7.orbit integrated (range and coding included).
    share=1.0: the rate this payload effectively gets (its capacity / the pass duration).
    share=the link share the capacity was scaled by: the PHYSICAL link rate (tx time, energy)."""
    if p.duration_s <= 0 or share <= 0:
        return 0.0
    return p.capacity_bytes * 8.0 / p.duration_s / share


@dataclass
class Delivery:
    item: Item
    fraction: float
    path: str                      # "direct" | "relay"
    delivered_s: float             # seconds since simulation start
    tx_s: dict                     # {"ground": s} or {"isl": s, "relay_ground": s}
    energy_J: dict                 # same keys
    deadline_met: bool | None = None

    @property
    def latency_s(self) -> float:
        return self.delivered_s - self.item.created_s


@dataclass
class CommsResult:
    deliveries: list[Delivery]
    result: Result                 # scheduler-compatible: feed it to sat7.scheduler.metrics
    link_use: dict                 # per link: bytes carried / capacity
    relay_left_bytes: float = 0.0  # handed to the relay but not delivered within the horizon
    label: str = LABEL

    def by_path(self) -> dict:
        out = {}
        for p in ("direct", "relay"):
            d = [x for x in self.deliveries if x.path == p]
            lat = np.array([x.latency_s for x in d]) / 3600 if d else np.array([])
            out[p] = {"items": len(d), "MB": round(sum(x.item.size * x.fraction for x in d) / 1e6, 3),
                      "latency_h": ({"median": round(float(np.median(lat)), 2),
                                     "p90": round(float(np.percentile(lat, 90)), 2),
                                     "max": round(float(lat.max()), 2)} if d else None),
                      "tx_s": {k: round(sum(x.tx_s.get(k, 0.0) for x in d), 1)
                               for k in ("ground", "isl", "relay_ground")},
                      "energy_J": {k: round(sum(x.energy_J.get(k, 0.0) for x in d), 1)
                                   for k in ("ground", "isl", "relay_ground")}}
        return out

    def totals(self) -> dict:
        d = self.deliveries
        dl = [x.deadline_met for x in d if x.deadline_met is not None]
        return {"label": self.label, "items": len(d),
                "MB_delivered": round(sum(x.item.size * x.fraction for x in d) / 1e6, 3),
                "relay_fraction_items": round(sum(x.path == "relay" for x in d) / max(1, len(d)), 4),
                "tx_s": round(sum(sum(x.tx_s.values()) for x in d), 1),
                "energy_J": round(sum(sum(x.energy_J.values()) for x in d), 1),
                "deadline_met_fraction": round(sum(dl) / len(dl), 4) if dl else None,
                "relay_left_MB": round(self.relay_left_bytes / 1e6, 3)}

    def summary(self) -> list[str]:
        t, bp = self.totals(), self.by_path()
        lines = [f"[{self.label}] delivered {t['MB_delivered']} MB in {t['items']} items; relay carried "
                 f"{100 * t['relay_fraction_items']:.1f}% of them; transmit time {t['tx_s']:.0f} s; "
                 f"transmit energy {t['energy_J'] / 1e3:.2f} kJ"]
        for p, v in bp.items():
            if v["items"]:
                lines.append(f"[{self.label}]   {p:<6} {v['items']:6d} items {v['MB']:9.3f} MB  latency med "
                             f"{v['latency_h']['median']} h / max {v['latency_h']['max']} h")
        return lines


def simulate_comms(items: list[Item], links: Links, start: datetime, policy: Policy,
                   cfg: CommsConfig | None = None, storage_bytes: float = 8e9,
                   encoder: str = "") -> CommsResult:
    """Replay arrivals over the contact plan [SIM]. Ground passes of the primary are scheduled by
    `policy` exactly as sat7.scheduler.simulate does; ISL windows offload to the relay the items for
    which the relay path wins on J; the relay's ground passes forward what it holds."""
    cfg = cfg or CommsConfig()
    items = sorted(items, key=lambda it: it.created_s)
    sec = lambda t: (t - start).total_seconds()
    events = [(sec(p.rise), 1, "ground", p) for p in links.primary_passes]
    events += [(sec(p.rise), 2, "relay_ground", p) for p in links.relay_passes]
    if cfg.relay_enabled:
        events += [(sec(w.start) + cfg.isl_setup_s, 0, "isl", w) for w in links.isl]
    events.sort(key=lambda e: (e[0], e[1]))
    primary_rises = [sec(p.rise) for p in links.primary_passes]
    relay_rises = [sec(p.rise) for p in links.relay_passes]
    relay_reserved = [0.0] * len(links.relay_passes)       # bytes promised to each relay pass
    relay_hold: list[list] = [[] for _ in links.relay_passes]
    res = Result(policy.name, encoder, bytes_offered=sum(i.size for i in items),
                 capacity=sum(p.capacity_bytes for p in links.primary_passes))
    deliveries: list[Delivery] = []
    use = {"ground": [0.0, sum(p.capacity_bytes for p in links.primary_passes)],
           "isl": [0.0, 0.0], "relay_ground": [0.0, sum(p.capacity_bytes for p in links.relay_passes)]}
    queue: list[Item] = []
    k, prev_rise, relay_stored = 0, -math.inf, 0.0
    r_isl = cfg.isl_effective_bps                           # what this payload gets on the ISL
    r_isl_phys = cfg.isl_rate_bps * cfg.isl_efficiency      # what the ISL transmitter runs at
    phys = lambda p, share: pass_rate_bps(p, share)         # physical ground rate of a pass

    def met(latency_s):
        return None if cfg.deadline_s is None else bool(latency_s <= cfg.deadline_s)

    for t_ev, _order, kind, w in events:
        while k < len(items) and items[k].created_s <= t_ev:
            queue.append(items[k])
            k += 1

        if kind == "ground":
            if not policy.buffers:                          # keep only what arrived since the last pass
                queue = [i for i in queue if i.created_s > prev_rise]
            total = sum(i.size for i in queue)
            if total > storage_bytes:                       # same eviction rule as scheduler.simulate
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
            rate, cum, sent_ids = pass_rate_bps(w), 0.0, set()
            for it, frac in policy.plan(queue, w.capacity_bytes, t_ev):
                cum += it.size * frac
                delivered = t_ev + w.duration_s * min(1.0, cum / w.capacity_bytes)
                r_phys = phys(w, links.primary_share)
                tx = it.size * frac * 8.0 / r_phys if r_phys else 0.0
                res.sent.append(Sent(it, frac, delivered))
                res.bytes_sent += it.size * frac
                deliveries.append(Delivery(it, frac, "direct", delivered, {"ground": tx},
                                           {"ground": cfg.p_tx_W * tx}, met(delivered - it.created_s)))
                sent_ids.add(id(it))
            use["ground"][0] += cum
            queue = [i for i in queue if id(i) not in sent_ids]
            prev_rise = t_ev

        elif kind == "isl":
            dur = max(0.0, w.duration_s - cfg.isl_setup_s)
            cap_isl = r_isl * dur / 8.0
            use["isl"][1] += cap_isl
            if cap_isl <= 0 or not queue:
                continue
            # the primary's own upcoming passes: tentative ledger, for the direct estimate only
            ahead = [(primary_rises[j], links.primary_passes[j]) for j in range(len(primary_rises))
                     if primary_rises[j] > t_ev]
            direct_used = [0.0] * len(ahead)
            isl_used, moved = 0.0, set()
            for it in policy.order(queue, t_ev):
                # ---- direct estimate: first own pass with room for the whole item
                t_direct = e_direct = math.inf
                dj = None
                for j, (rise, p) in enumerate(ahead):
                    if direct_used[j] + it.size <= p.capacity_bytes:
                        rate = pass_rate_bps(p)
                        t_direct = rise + (direct_used[j] + it.size) * 8.0 / rate - it.created_s
                        e_direct = cfg.p_tx_W * it.size * 8.0 / phys(p, links.primary_share)
                        dj = j
                        break
                # ---- relay estimate: stage 1 on this ISL window, stage 2 on a relay ground pass
                t_relay = e_relay = math.inf
                rj = None
                if it.kind in cfg.relay_kinds and isl_used + it.size <= cap_isl \
                        and relay_stored + it.size <= cfg.relay_storage_bytes:
                    isl_done = t_ev + (isl_used + it.size) * 8.0 / r_isl
                    for j, rise in enumerate(relay_rises):
                        p = links.relay_passes[j]
                        if rise >= isl_done and relay_reserved[j] + it.size <= p.capacity_bytes:
                            rate = pass_rate_bps(p)
                            t_relay = rise + (relay_reserved[j] + it.size) * 8.0 / rate - it.created_s
                            e_relay = (cfg.p_isl_W * it.size * 8.0 / r_isl_phys
                                       + cfg.p_tx_W * it.size * 8.0 / phys(p, links.relay_share))
                            rj = j
                            break
                if rj is None:
                    if dj is not None:
                        direct_used[dj] += it.size
                    continue
                j_direct = cfg.lam_E * e_direct + cfg.lam_T * t_direct if dj is not None else math.inf
                j_relay = cfg.lam_E * e_relay + cfg.lam_T * t_relay
                if cfg.deadline_s is not None:              # a path that misses the deadline loses first
                    ok_d, ok_r = t_direct <= cfg.deadline_s, t_relay <= cfg.deadline_s
                    if ok_d != ok_r:
                        j_direct, j_relay = (0.0, math.inf) if ok_d else (math.inf, 0.0)
                    elif not ok_d:                          # neither makes it: the faster one
                        j_direct, j_relay = t_direct, t_relay
                if j_relay < j_direct:
                    t_isl = it.size * 8.0 / r_isl_phys
                    isl_used += it.size
                    relay_reserved[rj] += it.size
                    relay_stored += it.size
                    relay_hold[rj].append((it, t_isl))
                    moved.add(id(it))
                elif dj is not None:
                    direct_used[dj] += it.size
            use["isl"][0] += isl_used
            queue = [i for i in queue if id(i) not in moved]

        else:                                               # the relay's ground pass
            j = links.relay_passes.index(w)
            rate, cum = pass_rate_bps(w), 0.0
            for it, t_isl in relay_hold[j]:
                cum += it.size
                delivered = t_ev + cum * 8.0 / rate
                tx = it.size * 8.0 / phys(w, links.relay_share)
                res.sent.append(Sent(it, 1.0, delivered))
                res.bytes_sent += it.size
                relay_stored -= it.size
                deliveries.append(Delivery(it, 1.0, "relay", delivered, {"isl": t_isl, "relay_ground": tx},
                                           {"isl": cfg.p_isl_W * t_isl, "relay_ground": cfg.p_tx_W * tx},
                                           met(delivered - it.created_s)))
            use["relay_ground"][0] += cum
            relay_hold[j] = []
    link_use = {name: {"MB": round(u / 1e6, 3), "capacity_MB": round(c / 1e6, 3),
                       "used": round(u / c, 4) if c else 0.0} for name, (u, c) in use.items()}
    return CommsResult(deliveries, res, link_use, relay_left_bytes=max(0.0, relay_stored))
