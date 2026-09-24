"""Build a scheduler workload out of REAL detections (WP6).

``sat7.scheduler.generate_workload`` invents a day: contexts from assumed
probabilities, detector confidence from a Beta(5,2). This module replaces those
draws with the measurements in ``results/wp6_tiles.csv`` and ``wp6_ships.csv``:
5,320 held-out Airbus tiles, 8,173 ground-truth ships, the trained detector's
real confidence on each one, and its real false alarms.

What is REAL here                       | What is still an ASSUMPTION
----------------------------------------|----------------------------------------
ships per tile, their lengths           | when each tile is captured (no timestamps
detector confidence, misses, false      |   in the data -> uniform over the day)
  alarms, per tile                      | which ships are "dark" (no AIS in the data)
tile context (classic pre-filter)       | how many tiles/day the satellite images
payload bytes (WP4 power law)           | the context mix, unless mix="dataset"

The Airbus set is a *ship-finding benchmark*: 59% of its tiles contain a ship and
27% touch a coast, which no real orbit ever sees. So by default we keep the
operational context mix as a stated assumption (``mix="orbit"``) and draw a real
tile from inside each context; ``mix="dataset"`` uses the file's own mix instead.
Either way everything *inside* a tile is measured, never invented.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .scheduler import Ship, SizeModel, Workload, WorkloadConfig

CONTEXTS = ("cloud", "ships", "coast", "empty")

# WP3: the learned gate's operating threshold, chosen on the val split (never on test) at the
# 99% ship-recall target. Below this the gate says "nothing here". Used by _gate_says_empty
# when the catalogue carries a gate_score column.
GATE_EMPTY_THR = 0.126


@dataclass
class RealWorkloadConfig:
    """Knobs for the real workload. Only the ones marked ASSUMPTION are invented."""
    hours: float = 24.0
    tiles_per_day: int = 40_000        # ASSUMPTION: duty cycle of the imager
    mix: str = "orbit"                 # "orbit" (assumed mix) | "dataset" (file's own mix)
    orbit_mix: tuple[float, float, float, float] = (0.15, 0.20, 0.05, 0.60)  # ASSUMPTION
    p_dark: float = 0.10               # ASSUMPTION: no AIS in the Airbus data
    det_thr: float = 0.25              # onboard detection threshold (real conf compared to it)
    seed: int = 0


@dataclass
class RealWorkloadStats:
    """What the sampled day is made of - print this next to every result."""
    tiles: int
    ships: int
    dark: int
    detected: int                      # ships whose real confidence >= det_thr
    false_alarms: int
    context_counts: dict
    source_tiles: int
    mix: str


def load_size_model(path: str | Path) -> SizeModel:
    """Power law fitted by scripts/wp6_fit_size_model.py."""
    m = json.loads(Path(path).read_text())
    return SizeModel(l1_a=m["L1"]["a"], l1_b=m["L1"]["b"], l2_a=m["L2"]["a"], l2_b=m["L2"]["b"])


def load_catalogue(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tile and ship catalogues, with the per-prediction false-alarm flags folded in.

    ``wp6_pred_flags.csv`` tags every one of the detector's boxes as a true positive or
    a false alarm, so the number of false alarms can be counted at *any* onboard
    threshold rather than only at the three cut-offs stored in wp6_tiles.csv.
    """
    d = Path(results_dir)
    tiles = pd.read_csv(d / "wp6_tiles.csv")
    ships = pd.read_csv(d / "wp6_ships.csv")
    flags = d / "wp6_pred_flags.csv"
    if flags.exists():
        f = pd.read_csv(flags)
        f = f[f.false_alarm & f.image.isin(set(tiles.image))]
        tiles.attrs["fa_conf"] = {img: g.conf.to_numpy() for img, g in f.groupby("image")}
    return tiles, ships


def workload_from_catalogue(tiles: pd.DataFrame, ships: pd.DataFrame,
                            cfg: RealWorkloadConfig | None = None
                            ) -> tuple[Workload, RealWorkloadStats]:
    """Sample a day of real tiles and return a Workload the scheduler can replay."""
    cfg = cfg or RealWorkloadConfig()
    if cfg.mix not in ("orbit", "dataset"):
        raise ValueError(f"mix must be 'orbit' or 'dataset', got {cfg.mix!r}")
    rng = np.random.default_rng(cfg.seed)

    fps_all = _false_alarms_per_tile(tiles, cfg.det_thr)
    gate_all = _gate_says_empty(tiles, ships, cfg.det_thr, fps_all)
    unconf_all = _unconfirmed_candidates(tiles, ships, cfg.det_thr)
    by_tile: dict[str, pd.DataFrame] = {k: v for k, v in ships.groupby("image")}
    pools = {c: tiles.index[tiles.ctx3 == c].to_numpy() for c in CONTEXTS}
    missing = [c for c, p in pools.items() if len(p) == 0]

    n = int(round(cfg.tiles_per_day * cfg.hours / 24))
    if cfg.mix == "dataset":
        picks = rng.integers(0, len(tiles), n)
        ctx_of = tiles.ctx3.to_numpy()
        contexts = ctx_of[picks]
    else:
        p = np.array(cfg.orbit_mix, float)
        for c in missing:                      # cannot draw a context we have no tile for
            p[CONTEXTS.index(c)] = 0.0
        p = p / p.sum()
        contexts = rng.choice(CONTEXTS, size=n, p=p)
        picks = np.empty(n, int)
        for c in CONTEXTS:
            m = contexts == c
            if m.any():
                picks[m] = rng.choice(pools[c], size=int(m.sum()))

    # ASSUMPTION: no capture times in the data -> spread uniformly over the day
    times = np.sort(rng.uniform(0, cfg.hours * 3600, n))

    names = tiles.image.to_numpy()
    ship_list: list[Ship] = []
    tile_list: list[tuple[float, str, tuple[int, ...]]] = []
    fa_list: list[int] = []
    gate_list: list[bool] = []
    unconf_list: list[int] = []
    for t, row, ctx in zip(times, picks, contexts):
        ids: list[int] = []
        g = by_tile.get(names[row])
        # Ships on cloud tiles are kept in the denominator on purpose: the onboard cloud
        # gate throws those tiles away, and that loss must show up in the recall we report.
        if g is not None:
            for size_px, conf in zip(g.size_px.to_numpy(), g.conf.to_numpy()):
                s = Ship(len(ship_list), float(t),
                         bool(rng.random() < cfg.p_dark),          # ASSUMPTION
                         float(conf),                              # REAL detector confidence
                         ctx == "coast",
                         float(size_px))                           # REAL length
                ship_list.append(s)
                ids.append(s.id)
        tile_list.append((float(t), str(ctx), tuple(ids)))
        fa_list.append(0 if ctx == "cloud" else int(fps_all[row]))
        gate_list.append(bool(gate_all[row]))
        unconf_list.append(int(unconf_all[row]))

    wl = Workload(ship_list, tile_list,
                  WorkloadConfig(hours=cfg.hours, tiles_per_day=cfg.tiles_per_day, seed=cfg.seed),
                  false_alarms=fa_list, gate_empty=gate_list, unconfirmed=unconf_list,
                  source="real detections (WP6)")
    stats = RealWorkloadStats(
        tiles=n, ships=len(ship_list), dark=sum(s.dark for s in ship_list),
        detected=sum(s.confidence >= cfg.det_thr for s in ship_list),
        false_alarms=int(sum(fa_list)),
        context_counts={c: int((contexts == c).sum()) for c in CONTEXTS},
        source_tiles=len(tiles), mix=cfg.mix)
    return wl, stats


def _gate_says_empty(tiles: pd.DataFrame, ships: pd.DataFrame, det_thr: float,
                     fps: np.ndarray) -> np.ndarray:
    """Per tile: would the onboard software claim there is nothing here?

    Two independent opinions must agree -- the classic pre-filter sees no candidate at all
    (``context == "empty_sea"``) *and* the detector fired nothing above the threshold. Only
    then is the safety-net thumbnail worth skipping. On the real catalogue this covers 21%
    of the truly empty tiles and puts 5 of the 1,134 missed ships at risk, which is the
    honest price of the saving; the classic filter's high false-candidate rate (WP3) is
    what keeps the number this low, so a learned gate would raise it.

    WP3 trained a learned gate that reaches 0.988 ship recall where the classic filter reaches
    0.645, at 1/365th of the cost. When the catalogue carries a ``gate_score`` column (written by
    ``scripts/wp3_score_tiles.py``) it replaces the ``context == "empty_sea"`` test, using the
    threshold picked on **val** in WP3. The classic test is kept as the fallback so results
    without the column are unchanged and the two can be compared.
    """
    if "context" not in tiles.columns:      # catalogue built before WP7: gate nothing
        return np.zeros(len(tiles), bool)
    fired = ships[ships.conf >= det_thr].groupby("image").size()
    n_fired = tiles.image.map(fired).fillna(0).to_numpy()
    nothing_fired = (n_fired == 0) & (np.asarray(fps) == 0)
    if "gate_score" in tiles.columns:
        return (tiles.gate_score.to_numpy() < GATE_EMPTY_THR) & nothing_fired
    return (tiles.context.to_numpy() == "empty_sea") & nothing_fired


def _unconfirmed_candidates(tiles: pd.DataFrame, ships: pd.DataFrame, det_thr: float) -> np.ndarray:
    """Bright objects the cheap classic stage found that the network did not confirm.

    Both stages already run onboard, so this costs nothing extra, and it is the closest
    thing we have to the network saying "I may have missed something here". Measured on
    the 1,415 real coastal tiles: where this is 0 the tile holds 5% of the ships the
    network missed, where it is 11+ it holds 75% of them.
    """
    if "n_candidates" not in tiles.columns:     # catalogue built before WP7
        return np.zeros(len(tiles), int)
    det = ships[ships.conf >= det_thr].groupby("image").size()
    n_det = tiles.image.map(det).fillna(0).to_numpy()
    # a false alarm is still the network firing on that candidate, so it counts as
    # "confirmed" here; subtracting only true positives over-counts the disagreement
    # (measured: it flips the escalation decision on 0.6% of real coastal tiles)
    n_det = n_det + _false_alarms_per_tile(tiles, det_thr)
    return np.clip(tiles.n_candidates.to_numpy() - n_det, 0, None).astype(int)


def _false_alarms_per_tile(tiles: pd.DataFrame, det_thr: float) -> np.ndarray:
    """How many false detections each tile produces at this onboard threshold.

    Exact when the per-prediction flags are loaded; otherwise it falls back to the
    nearest measured ``n_fp_*`` column in the tile catalogue.
    """
    fa = tiles.attrs.get("fa_conf")
    if fa is not None:
        return np.array([int((fa[n] >= det_thr).sum()) if n in fa else 0 for n in tiles.image])
    cols = {float(c.split("_")[-1]): c for c in tiles.columns if c.startswith("n_fp_")}
    if not cols:
        raise ValueError("catalogue has no false-alarm data; rerun wp6_build_catalogue.py")
    usable = [t for t in cols if t <= det_thr + 1e-9]
    return tiles[cols[max(usable)] if usable else cols[min(cols)]].to_numpy()
