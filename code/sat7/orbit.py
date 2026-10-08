"""Ground-station contact windows and per-pass downlink capacity.

Orbit: by default a *representative* sun-synchronous LEO orbit built from
Keplerian elements (no TLE needed, fully reproducible). A real satellite can be
used instead by passing a TLE (e.g. from CelesTrak).

Capacity of one pass (as in the "scalable transmission" paper):

    C_pass = sum_k R_k * dt_k

where the rate R_k depends on the slant range at time k. We use a simple,
explainable link model: SNR scales with 1/d^2 (free-space loss) from a reference
SNR at zenith, and the rate follows Shannon's formula R = B log2(1 + SNR),
capped at the modem's maximum rate. All numbers are ASSUMPTIONS, set in
``LinkConfig`` and reported in the paper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import WGS72, Satrec
from skyfield.api import EarthSatellite, load, wgs84

MU_EARTH = 398600.4418   # km^3 / s^2
R_EARTH = 6378.137       # km


@dataclass
class OrbitConfig:
    altitude_km: float = 500.0
    inclination_deg: float = 97.4     # ~sun-synchronous at 500 km
    raan_deg: float = 0.0
    epoch: datetime = datetime(2026, 9, 18, tzinfo=timezone.utc)


@dataclass
class GroundStation:
    name: str = "Sfax (ENIS)"
    lat_deg: float = 34.74
    lon_deg: float = 10.76
    min_elevation_deg: float = 5.0


@dataclass
class LinkConfig:
    """Downlink assumptions. Two presets below; state which one you use."""
    name: str = "cubesat_sband"
    bandwidth_hz: float = 2e6         # occupied bandwidth
    snr_zenith_db: float = 12.0       # SNR when the satellite is straight overhead
    max_rate_bps: float = 8e6         # modem limit
    efficiency: float = 0.8           # coding / protocol overhead (0..1)


LINK_PRESETS = {
    "cubesat_sband": LinkConfig(),
    "smallsat_xband": LinkConfig("smallsat_xband", 50e6, 15.0, 300e6, 0.8),
}


def make_satellite(cfg: OrbitConfig | None = None, ts=None, tle: tuple[str, str] | None = None) -> EarthSatellite:
    """Skyfield satellite from a TLE, or from circular-orbit Keplerian elements."""
    ts = ts or load.timescale()
    if tle is not None:
        return EarthSatellite(tle[0], tle[1], "SAT", ts)
    cfg = cfg or OrbitConfig()
    a = R_EARTH + cfg.altitude_km
    n_rad_per_min = math.sqrt(MU_EARTH / a**3) * 60.0
    epoch_1949 = (cfg.epoch - datetime(1949, 12, 31, tzinfo=timezone.utc)).total_seconds() / 86400.0
    sat = Satrec()
    sat.sgp4init(WGS72, "i", 99999, epoch_1949,
                 0.0, 0.0, 0.0,               # bstar, ndot, nddot (no drag: 1-day study)
                 0.0,                         # eccentricity (circular)
                 0.0,                         # argument of perigee
                 math.radians(cfg.inclination_deg),
                 0.0,                         # mean anomaly
                 n_rad_per_min,
                 math.radians(cfg.raan_deg))
    return EarthSatellite.from_satrec(sat, ts)


def rate_bps(range_km: np.ndarray, altitude_km: float, link: LinkConfig) -> np.ndarray:
    """Achievable rate at a given slant range (free-space scaling + Shannon, capped)."""
    snr0 = 10 ** (link.snr_zenith_db / 10)
    snr = snr0 * (altitude_km / np.asarray(range_km)) ** 2
    shannon = link.bandwidth_hz * np.log2(1 + snr)
    return np.minimum(shannon, link.max_rate_bps) * link.efficiency


@dataclass
class Pass:
    rise: datetime
    culminate: datetime
    set: datetime
    max_elevation_deg: float
    capacity_bytes: float
    station: str = ""                 # which ground station saw this pass (multi-station sweeps)

    @property
    def duration_s(self) -> float:
        return (self.set - self.rise).total_seconds()


def find_passes(sat: EarthSatellite, gs: GroundStation, start: datetime, hours: float = 24.0,
                link: LinkConfig | None = None, altitude_km: float = 500.0, step_s: float = 5.0,
                ts=None) -> list[Pass]:
    """All passes above ``gs.min_elevation_deg`` in the window, with their capacity."""
    ts = ts or load.timescale()
    link = link or LinkConfig()
    site = wgs84.latlon(gs.lat_deg, gs.lon_deg)
    t0, t1 = ts.from_datetime(start), ts.from_datetime(start + timedelta(hours=hours))
    times, events = sat.find_events(site, t0, t1, altitude_degrees=gs.min_elevation_deg)

    passes, rise = [], None
    culm = None
    for t, ev in zip(times, events):
        if ev == 0:
            rise, culm = t, None
        elif ev == 1 and rise is not None:
            culm = t
        elif ev == 2 and rise is not None:
            n = max(2, int((t.tt - rise.tt) * 86400 / step_s) + 1)
            samples = ts.tt_jd(np.linspace(rise.tt, t.tt, n))
            alt, _, dist = (sat - site).at(samples).altaz()
            r = rate_bps(dist.km, altitude_km, link)
            dt = (t.tt - rise.tt) * 86400 / (n - 1)
            cap_bits = float(np.trapezoid(r, dx=dt)) if hasattr(np, "trapezoid") else float(np.trapz(r, dx=dt))
            peak = culm if culm is not None else rise
            passes.append(Pass(rise.utc_datetime(), peak.utc_datetime(), t.utc_datetime(),
                               float(alt.degrees.max()), cap_bits / 8.0, gs.name))
            rise = None
    return passes


# ---------------------------------------------------------------- multi-ground-station extension
# ARCHITECTURE.md / START_HERE section 5: "single ground station only; 11.4 h max gap dominates
# latency". A real mission books several stations; more stations = more, earlier contacts = shorter
# gaps and lower latency. This adds that without touching the single-station path or the scheduler:
# it just returns MORE passes on one merged timeline (each tagged with the station that saw it), and
# `simulate`/`simulate_online` consume them exactly as before.

# A few real high-latitude / mid-latitude stations often used for LEO downlink (public locations).
# Coordinates are approximate and ASSUMPTION-level; swap in the mission's actual network.
STATION_PRESETS = {
    "sfax":     GroundStation("Sfax (ENIS)", 34.74, 10.76, 5.0),
    "svalbard": GroundStation("Svalbard (KSAT)", 78.23, 15.39, 5.0),
    "troll":    GroundStation("Troll (Antarctica)", -72.01, 2.53, 5.0),
    "fairbanks":GroundStation("Fairbanks (Alaska)", 64.80, -147.50, 5.0),
    "awarua":   GroundStation("Awarua (NZ)", -46.52, 168.38, 5.0),
}


def find_passes_multi(sat: EarthSatellite, stations: list[GroundStation], start: datetime,
                      hours: float = 24.0, link: LinkConfig | None = None, altitude_km: float = 500.0,
                      step_s: float = 5.0, ts=None) -> list[Pass]:
    """All passes over a NETWORK of ground stations, merged onto one chronological timeline.

    Each Pass keeps the name of the station that saw it (``Pass.station``). Capacity is computed
    per station with the same link model, so a pass over a station with a better geometry carries
    more bytes. Overlapping passes at different stations are left as separate passes on purpose:
    a satellite can only key one downlink at a time, but modelling that contention is the
    scheduler's job (it already sends at most ``capacity_bytes`` per pass), and two near-simultaneous
    passes simply give it two chances -- any double-count is bounded by the per-pass capacity.
    """
    ts = ts or load.timescale()
    link = link or LinkConfig()
    out: list[Pass] = []
    for gs in stations:
        out += find_passes(sat, gs, start, hours, link, altitude_km, step_s, ts)
    return sorted(out, key=lambda p: p.rise)


def max_gap_h(passes: list[Pass], start: datetime, hours: float) -> float:
    """Longest stretch (hours) with no contact in [start, start+hours] -- the latency-dominating gap.

    Measured edge to edge: from window start to the first rise, between consecutive passes (rise of
    the next minus set of the previous), and from the last set to the window end. This is the number
    the multi-station extension is meant to cut, so it is reported directly."""
    if not passes:
        return hours
    end = start + timedelta(hours=hours)
    ps = sorted(passes, key=lambda p: p.rise)
    gaps = [(ps[0].rise - start).total_seconds()]
    for a, b in zip(ps, ps[1:]):
        gaps.append((b.rise - a.set).total_seconds())
    gaps.append((end - ps[-1].set).total_seconds())
    return max(0.0, max(gaps) / 3600.0)
