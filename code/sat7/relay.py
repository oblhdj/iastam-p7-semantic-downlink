"""B4 -- optional inter-satellite relay: visibility windows (SIM) + path choice (direct vs relay).

START_HERE section 5 item 3 / report 18 section 3. Scope is deliberately narrow -- no new orbital
mechanics:
  * The relay satellite is modeled the SAME way as the primary -- SGP4 via sat7.orbit.make_satellite
    -- as a second reproducible Keplerian orbit (a RAAN-offset plane), no external TLE. Mutual
    visibility windows come straight out of the propagated positions.
  * The path-choice rule is J = lam_E*E + lam_T*T, choose min(J_direct, J_relay), per item.

Honesty labels (every figure this module emits carries one):
  * ISL windows, direct windows, and LATENCY T        SIM  -- real orbit propagation (sat7.orbit).
  * per-item ENERGY E (E_direct / E_ISL / E_GS / E_relay)  TARGET -- NOT measured. Supplied by the
    energy track (companion work) through the `direct_energy`/`relay_energy` callables. This module
    ships clearly-labeled PLACEHOLDER energies so the logic runs and is testable; they are NOT
    joules and must be replaced before any B4 energy number is reported.
  * lam_E, lam_T trade-off weights                    ASSUMPTION -- invented knobs, swept.

The energy track fills E/T by passing its own `direct_energy(d_bytes, gs_window, powers)` and
`relay_energy(d_bytes, isl_window, gs_window, powers)` into route(); the placeholder defaults below
match that signature exactly so the real functions drop in unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from skyfield.api import load

from .orbit import OrbitConfig, R_EARTH, make_satellite


@dataclass
class RelayConfig:
    """Second satellite + ISL geometry. Same construction as the primary (SGP4, reproducible)."""
    orbit: OrbitConfig = field(default_factory=lambda: OrbitConfig(raan_deg=30.0))
    grazing_km: float = 100.0          # line of sight must clear Earth + this atmospheric margin
    max_range_km: float | None = None  # optional ISL link-range cap (None = pure line-of-sight)


@dataclass
class ISLWindow:
    start: datetime
    end: datetime
    min_range_km: float
    mean_range_km: float

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()


def _start(x) -> datetime:
    """Window start for either a ground Pass (.rise) or an ISLWindow (.start)."""
    return getattr(x, "start", None) or x.rise


def _next(windows, t: datetime):
    """First window whose start is at or after t (lists are chronological)."""
    for w in windows:
        if _start(w) >= t:
            return w
    return None


def _los_clear(ra: np.ndarray, rb: np.ndarray, r_min: float) -> np.ndarray:
    """True where the segment ra--rb (geocentric km, shape (3, n)) clears a sphere of radius r_min.

    Closest approach of the SEGMENT to Earth's centre: if it stays above r_min the two endpoints
    can see each other; if the line dips below (Earth between them) it is blocked.
    """
    d = rb - ra
    dd = np.sum(d * d, axis=0)
    t = np.clip(-np.sum(ra * d, axis=0) / np.where(dd > 0, dd, 1.0), 0.0, 1.0)
    closest = ra + t * d
    return np.linalg.norm(closest, axis=0) >= r_min


def isl_windows(sat_a, sat_b, start: datetime, hours: float = 36.0, step_s: float = 10.0,
                grazing_km: float = 100.0, max_range_km: float | None = None, ts=None):
    """Mutual-visibility windows between two satellites, from propagated positions (SIM).

    Reuses the same SGP4 propagation as sat7.orbit (skyfield), so no orbital mechanics is re-derived
    -- it only tests line-of-sight between the two propagated position tracks.
    """
    ts = ts or load.timescale()
    n = int(round(hours * 3600 / step_s)) + 1
    t0 = ts.from_datetime(start)
    times = ts.tt_jd(np.linspace(t0.tt, t0.tt + hours / 24.0, n))
    ra = sat_a.at(times).position.km
    rb = sat_b.at(times).position.km
    rng = np.linalg.norm(ra - rb, axis=0)
    vis = _los_clear(ra, rb, R_EARTH + grazing_km)
    if max_range_km is not None:
        vis = vis & (rng <= max_range_km)
    dts = list(times.utc_datetime())
    wins, i = [], 0
    while i < n:
        if vis[i]:
            j = i
            while j + 1 < n and vis[j + 1]:
                j += 1
            seg = rng[i:j + 1]
            wins.append(ISLWindow(dts[i], dts[j], float(seg.min()), float(seg.mean())))
            i = j + 1
        else:
            i += 1
    return wins


def make_relay(cfg: RelayConfig | None = None, ts=None):
    """The relay satellite, built the same way as the primary (sat7.orbit.make_satellite)."""
    cfg = cfg or RelayConfig()
    return make_satellite(cfg.orbit, ts=ts)


# ----------------------------------------------------------------------------- energy link params
@dataclass
class LinkParams:
    """Scalars the energy callables need. These MIRROR report 19 (the energy track owns them); pass
    its config to override. p_tx is report 17's ASSUMPTION; p_isl / r_isl are report 19's new
    ASSUMPTIONs (swept there); r_gs is the orbit sim's effective rate."""
    p_tx_W: float = 15.0          # ASSUMPTION (report 17)
    p_isl_W: float = 12.0         # ASSUMPTION-new (report 19)
    r_gs_bps: float = 2.53e6      # SIM (orbit sim, report 17/19)
    r_isl_bps: float = 2.53e6     # ASSUMPTION-new (report 19)


# --------------------------------------------------------------------------- energy placeholders
# TARGET. Signatures match the energy track's report 19 section 4 EXACTLY, so wp19_relay_energy's
# real functions drop in with no change to route(). The magnitudes here are normalized placeholders
# (a direct hop = 1 unit), NOT joules -- do not report them as energy.
def default_direct_energy(d_bytes: float, p_tx: float, r_gs_bps: float) -> dict:
    return {"E_comm_J": 1.0, "tx_time_s": 0.0, "label": "TARGET-placeholder"}


def default_relay_energy(d_bytes: float, p_tx: float, p_isl: float,
                         r_gs_bps: float, r_isl_bps: float) -> dict:
    e_isl, e_gs = 0.6, 1.0
    return {"E_ISL_J": e_isl, "E_GS_J": e_gs, "E_relay_J": e_isl + e_gs,
            "isl_time_s": 0.0, "gs_time_s": 0.0, "label": "TARGET-placeholder"}


def route_item(item_bytes: float, t_now: datetime, direct_passes, isl_wins, relay_passes, *,
               lam_E: float, lam_T: float, direct_energy=default_direct_energy,
               relay_energy=default_relay_energy, link: LinkParams | None = None,
               isl_enabled: bool = True, direct_latency_s: float | None = None) -> dict:
    """Per-item path choice: J = lam_E*E + lam_T*T, choose min(J_direct, J_relay).

    T (latency to the item reaching the ground, seconds) is SIM -- read from the propagated windows
    (this track's half of J). E is TARGET -- whatever the injected energy callables return; the
    energy track (report 19) supplies the real `direct_energy`/`relay_energy`, placeholders by
    default. With isl_enabled=False the relay option is removed and this collapses to ground-only.

    `direct_latency_s` is the direct-path latency seam: by default (None) the direct baseline is
    route's own next-pass-WAIT estimate (delivery assumed at pass rise). The B0-B4 campaign's B3
    delivers MID-pass, so it passes its ACTUAL per-item delivered latency here; route then uses that
    exact value for T_direct (and takes delivery as having happened, so E_direct is the real energy),
    while still owning the relay leg (T_relay from windows) and the min(J) decision. Default None
    leaves report 20's behaviour -- and every number in it -- unchanged.
    """
    link = link or LinkParams()
    gd = _next(direct_passes, t_now)                               # primary's next ground pass
    if direct_latency_s is not None:                              # caller supplies the real direct latency
        T_direct = direct_latency_s
        E_direct = direct_energy(item_bytes, link.p_tx_W, link.r_gs_bps)["E_comm_J"]
    else:
        T_direct = (_start(gd) - t_now).total_seconds() if gd else math.inf
        E_direct = direct_energy(item_bytes, link.p_tx_W, link.r_gs_bps)["E_comm_J"] if gd else math.inf
    J_direct = lam_E * E_direct + lam_T * T_direct

    out = {"path": "direct", "T_s": T_direct, "E": E_direct, "J": J_direct,
           "J_direct": J_direct, "T_direct_s": T_direct,
           "J_relay": math.inf, "T_relay_s": math.inf}

    if isl_enabled:
        wi = _next(isl_wins, t_now)                                # next sat-to-relay window
        gr = _next(relay_passes, _start(wi)) if wi else None       # relay's next ground pass after it
        if wi is not None and gr is not None:
            T_relay = (_start(gr) - t_now).total_seconds()
            E_relay = relay_energy(item_bytes, link.p_tx_W, link.p_isl_W,
                                   link.r_gs_bps, link.r_isl_bps)["E_relay_J"]
            J_relay = lam_E * E_relay + lam_T * T_relay
            out.update(J_relay=J_relay, T_relay_s=T_relay)
            if J_relay < J_direct:
                out.update(path="relay", T_s=T_relay, E=E_relay, J=J_relay)
    return out


def route(plan, links, *, lam_E: float, lam_T: float, isl_enabled: bool = True, **kw) -> list[dict]:
    """report 18 section 3 signature. `plan` = iterable of items with 'bytes' and 't_now'; `links` =
    (direct_passes, isl_wins, relay_passes). Returns one routing decision dict per item (path/E/T/J).
    This is the seam wp18_campaign_runner.py's B4 stub reserves; it stays TARGET until the energy
    track's real direct_energy/relay_energy (report 19) are passed in.
    """
    direct_passes, isl_wins, relay_passes = links
    return [route_item(it["bytes"], it["t_now"], direct_passes, isl_wins, relay_passes,
                       lam_E=lam_E, lam_T=lam_T, isl_enabled=isl_enabled,
                       direct_latency_s=it.get("direct_latency_s"), **kw) for it in plan]
