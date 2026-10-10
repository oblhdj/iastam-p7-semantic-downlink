"""Regenerate the synthetic stand-in tiles bundled with the quickstart demo.

WHY SYNTHETIC. The Airbus Ship Detection Challenge rules (section 7B, "Data Security") bind
entrants not to "publish, redistribute or otherwise provide" the Competition Data to anyone who
has not accepted those rules (kaggle.com/competitions/airbus-ship-detection/rules). So no real
Airbus tile is checked in. Instead, each bundled tile is a procedurally rendered 768x768 stand-in
for one REAL test tile.

WHAT IS COPIED FROM THE REAL TILE, AND WHAT IS NOT. Only the *layout* is taken from committed repo
artefacts -- never a pixel:
  * ship positions/sizes = the committed detector boxes for that tile in
    code/results/wp1_predictions.csv (conf >= 0.25, de-fragmented, true positives only per
    code/results/wp6_pred_flags.csv); a committed false alarm is rendered as a bright speck;
  * the scene type (open sea / coast / cloud / empty) = the pre-filter context recorded for that
    tile in code/results/wp6_tiles.csv.
Sea texture, hull shapes, colours, wakes, coastline and clouds are invented by this script. The
renderer was adjusted by eye until the trained detector fires on its hulls; nothing measured is
claimed from what the detector does on these images (see README "Labels").

Deterministic: same seed -> byte-identical JPEGs (sha256 recorded in tiles/manifest.json).

    python demo/quickstart/make_synthetic_tiles.py          # needs numpy + opencv only
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "code" / "results"
OUT = HERE / "tiles"
TILE = 768
SEED = 7

# The 9 REAL test-split tiles the demo stands in for, in swath order (row-major 3x3). Chosen from
# code/results so that every branch of the P0-P3 scheme appears: confident ships (P1), uncertain
# ships (P2), sub-threshold boxes (P0), coastal ships (P3 + context tile), an empty tile, a cloud
# tile (discarded by context) and a tile whose only >=0.25 box is a measured false alarm.
TILES = [
    ("00113a75c.jpg", "seven ships, mostly confident"),
    ("0002756f7.jpg", "two uncertain ships + a sub-threshold box"),
    ("00f34434e.jpg", "a 170 px ship beside small craft"),
    ("03204a586.jpg", "coast: three ships near shore"),
    ("00293fb9e.jpg", "empty sea"),
    ("01933be65.jpg", "one small ship"),
    ("05d407505.jpg", "coast: a large ship"),
    ("0c5bf1395.jpg", "no ship; one measured false alarm"),
    ("024e6ba29.jpg", "cloud: dropped before detection"),
]


# ------------------------------------------------------------------------------ committed inputs

def _read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(newline="") as f:
        return list(csv.DictReader(f))


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _inside(a, b) -> float:
    """Share of the smaller box covered by the other: catches a fragment nested in its hull."""
    ix = max(0.0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
    iy = max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    return ix * iy / max(1e-9, min(a[2] * a[3], b[2] * b[3]))


def layout(image: str, preds: list[dict], flags: dict) -> tuple[list, list]:
    """(ships, specks) to render for one real tile, from its committed detections.

    The detector often reports one hull as 2-4 overlapping fragments; rendering each would draw a
    pile-up, so boxes are de-fragmented (highest confidence kept): a box is dropped when it
    overlaps a kept one at IoU >= 0.3, or lies at least half inside it.
    """
    rows = sorted((r for r in preds if r["image"] == image and float(r["conf"]) >= 0.25),
                  key=lambda r: -float(r["conf"]))
    ships, specks = [], []
    for r in rows:
        box = tuple(float(r[k]) for k in ("cx", "cy", "w", "h"))
        fa = flags.get((image, round(float(r["conf"]), 6)), False)
        target = specks if fa else ships
        if all(_iou(box, k) < 0.3 and _inside(box, k) < 0.5 for k in ships + specks):
            target.append(box)
    return ships, specks


# ------------------------------------------------------------------------------ rendering

def ocean(rng, tint: float = 0.0) -> np.ndarray:
    """Dark teal sea with a slow illumination drift and fine wave texture (float32 BGR)."""
    base = np.array([66.0, 54.0, 20.0]) + tint * np.array([18.0, 16.0, 10.0])
    lf = cv2.resize(rng.normal(0, 1, (5, 5)).astype(np.float32), (TILE, TILE),
                    interpolation=cv2.INTER_CUBIC) * 3.0
    hf = cv2.GaussianBlur(rng.normal(0, 1, (TILE, TILE)).astype(np.float32), (0, 0), 1.0) * 5.0
    img = np.empty((TILE, TILE, 3), np.float32)
    img[:] = base
    img += (lf + hf)[..., None] * np.array([1.0, 0.85, 0.55], np.float32)
    return img


def _ship_geometry(w: float, h: float, k: float = 0.2):
    """Length, beam and heading of a slim hull whose axis-aligned box is about (w, h)."""
    r = h / max(w, 1e-6)
    if r <= k:
        return w, max(3.0, h), 0.0
    if r >= 1 / k:
        return h, max(3.0, w), math.pi / 2
    t = (r - k) / (1 - r * k)
    th = math.atan(t)
    length = w / (math.cos(th) + k * math.sin(th))
    return length, max(3.0, k * length), th


def _poly(cx, cy, pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[cx + u * c - v * s, cy + u * s + v * c] for u, v in pts], np.float32)


def draw_ship(img: np.ndarray, wake: np.ndarray, box, rng) -> None:
    cx, cy, w, h = box
    length, beam, th = _ship_geometry(w, h)
    ang = th if rng.random() < 0.5 else math.pi - th          # "/" or "\" diagonal
    if rng.random() < 0.5:
        ang += math.pi                                       # which end is the bow
    L2, B2 = length / 2, beam / 2
    hull = [(-L2, -B2), (L2 - beam, -B2), (L2, 0.0), (L2 - beam, B2), (-L2, B2)]
    palette = [(205, 208, 210), (60, 70, 150), (70, 72, 78), (225, 228, 230), (120, 130, 140)]
    colour = palette[int(rng.integers(len(palette)))]
    s = 8  # sub-pixel precision for fillPoly
    to_i = lambda p: np.round(p * s).astype(np.int32)

    # wake: a faint V of churned water behind the stern, on its own layer (blurred later)
    if length > 18 and rng.random() < 0.8:
        stern = _poly(cx, cy, [(-L2, 0.0)], ang)[0]
        for side in (-1, 1):
            tip = _poly(cx, cy, [(-L2 - 1.6 * length, side * 0.45 * length)], ang)[0]
            cv2.line(wake, tuple(to_i(stern)), tuple(to_i(tip)), 0.7,
                     max(1, int(beam * 0.2)), cv2.LINE_AA, shift=3)
        tail = _poly(cx, cy, [(-L2 - 1.0 * length, 0.0)], ang)[0]
        cv2.line(wake, tuple(to_i(stern)), tuple(to_i(tail)), 1.0, max(2, int(beam * 0.5)),
                 cv2.LINE_AA, shift=3)

    shadow = _poly(cx + 0.12 * beam, cy + 0.12 * beam, hull, ang)
    cv2.fillPoly(img, [to_i(shadow)], (14, 16, 12), cv2.LINE_AA, shift=3)
    cv2.fillPoly(img, [to_i(_poly(cx, cy, hull, ang))], colour, cv2.LINE_AA, shift=3)
    # superstructure near the stern + a deck line: what makes a hull read as a ship
    sup = [(-L2 + 0.08 * length, -0.6 * B2), (-L2 + 0.28 * length, -0.6 * B2),
           (-L2 + 0.28 * length, 0.6 * B2), (-L2 + 0.08 * length, 0.6 * B2)]
    cv2.fillPoly(img, [to_i(_poly(cx, cy, sup, ang))], (240, 242, 244), cv2.LINE_AA, shift=3)
    if length > 40:
        a, b = _poly(cx, cy, [(-L2 + 0.32 * length, 0.0), (L2 - beam, 0.0)], ang)
        dark = tuple(int(0.6 * c) for c in colour)
        cv2.line(img, tuple(to_i(a)), tuple(to_i(b)), dark, max(1, int(beam * 0.25)),
                 cv2.LINE_AA, shift=3)


def draw_speck(img: np.ndarray, box, rng) -> None:
    """A whitecap / floating-debris speck where the detector raised a false alarm."""
    cx, cy, w, h = box
    for _ in range(4):
        dx, dy = rng.normal(0, w / 6, 2)
        cv2.ellipse(img, (int(cx + dx), int(cy + dy)), (max(1, int(w / 4)), max(1, int(h / 5))),
                    float(rng.uniform(0, 180)), 0, 360, (200, 210, 205), -1, cv2.LINE_AA)


def add_coast(img: np.ndarray, ships, rng) -> None:
    """Land on the tile edge farthest from the ships: sand and scrub behind a beach."""
    if ships:
        mx = np.mean([b[0] for b in ships]); my = np.mean([b[1] for b in ships])
    else:
        mx = my = TILE / 2
    side = max({"top": my, "bottom": TILE - my, "left": mx, "right": TILE - mx}.items(),
               key=lambda kv: kv[1])[0]
    depth = 0.30 * TILE
    t = np.linspace(0, TILE, 33)
    walk = np.cumsum(rng.normal(0, 9, t.size)); walk -= walk.mean()
    line = np.clip(depth + walk, 0.15 * TILE, 0.42 * TILE)
    if side in ("top", "left"):
        edge = [(ti, li) for ti, li in zip(t, line)] + [(TILE, 0), (0, 0)]
    else:
        edge = [(ti, TILE - li) for ti, li in zip(t, line)] + [(TILE, TILE), (0, TILE)]
    if side in ("left", "right"):
        edge = [(y, x) for x, y in edge]
    mask = np.zeros((TILE, TILE), np.uint8)
    cv2.fillPoly(mask, [np.array(edge, np.int32)], 1)
    beach = cv2.dilate(mask, np.ones((15, 15), np.uint8)) - mask
    tex = cv2.GaussianBlur(rng.normal(0, 1, (TILE, TILE)).astype(np.float32), (0, 0), 3.0)
    tex /= tex.std()
    blot = cv2.resize(rng.normal(0, 1, (10, 10)).astype(np.float32), (TILE, TILE),
                      interpolation=cv2.INTER_CUBIC)
    veg = (blot + 0.4 * tex) > 0.3
    land = np.empty_like(img)
    land[:] = (150, 172, 186)                                # dry sand / built-up
    land[veg] = (92, 128, 104)                               # scrub
    land += (tex * 9.0)[..., None]
    img[mask > 0] = land[mask > 0]
    img[beach > 0] = 0.5 * img[beach > 0] + 0.5 * np.array([190, 210, 220], np.float32)


def add_cloud(img: np.ndarray, rng) -> None:
    """Thick cumulus over ~95% of the tile, a few gaps of sea showing through."""
    tex = cv2.resize(rng.normal(0, 1, (12, 12)).astype(np.float32), (TILE, TILE),
                     interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (TILE, TILE)).astype(np.float32), (0, 0), 4.0)
    cover = (tex > -1.55).astype(np.float32)
    cover = cv2.GaussianBlur(cover, (0, 0), 6.0)
    cloud = np.empty_like(img)
    cloud[:] = (236, 236, 234)
    cloud += (tex * 6 + fine * 10)[..., None]
    img[:] = img * (1 - cover[..., None]) + cloud * cover[..., None]


def render(image: str, context: str, ships, specks, rng) -> np.ndarray:
    img = ocean(rng, tint=float(rng.uniform(-0.5, 1.0)))
    wake = np.zeros((TILE, TILE), np.float32)
    if context == "land_coast":
        add_coast(img, ships, rng)
    for b in ships:
        draw_ship(img, wake, b, rng)
    for b in specks:
        draw_speck(img, b, rng)
    wake = cv2.GaussianBlur(wake, (0, 0), 2.0)
    img += (wake * 38.0)[..., None]
    if context == "cloud":
        add_cloud(img, rng)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)                 # sensor optics
    img += rng.normal(0, 1.2, img.shape).astype(np.float32)   # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    preds = _read_csv("wp1_predictions.csv")
    flags = {(r["image"], round(float(r["conf"]), 6)): r["false_alarm"] == "True"
             for r in _read_csv("wp6_pred_flags.csv")}
    ctx = {r["image"]: r["context"] for r in _read_csv("wp6_tiles.csv")}
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    entries = []
    for slot, (image, note) in enumerate(TILES):
        ships, specks = layout(image, preds, flags)
        img = render(image, ctx[image], ships, specks, rng)
        name = f"{slot + 1:02d}_synthetic_{Path(image).stem}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        assert ok
        (OUT / name).write_bytes(buf.tobytes())
        entries.append({"slot": slot, "row": slot // 3, "col": slot % 3, "file": name,
                        "synthetic": True, "stands_in_for": image,
                        "reference_context": ctx[image], "scene": note,
                        "rendered_ships": len(ships), "rendered_specks": len(specks),
                        "sha256": hashlib.sha256(buf.tobytes()).hexdigest()})
        print(f"  {name}  ({ctx[image]}, {len(ships)} ships, {len(specks)} specks)")
    manifest = {
        "label": "SYNTHETIC -- procedurally rendered stand-ins; no Airbus pixels. Layout only "
                 "(ship boxes, scene type) is taken from committed REAL results for the named tile.",
        "why": "Airbus Ship Detection Challenge rules s.7B forbid redistributing Competition Data.",
        "generator": "demo/quickstart/make_synthetic_tiles.py", "seed": SEED, "tile_px": TILE,
        "swath": {"rows": 3, "cols": 3, "px": 3 * TILE, "stitch": "abutting, overlap 0 (wp15)"},
        "layout_sources": ["code/results/wp1_predictions.csv", "code/results/wp6_pred_flags.csv",
                           "code/results/wp6_tiles.csv"],
        "tiles": entries,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(entries)} tiles + manifest.json to {OUT}")


if __name__ == "__main__":
    main()
