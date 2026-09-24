"""Classic computer-vision pre-filter (stage [1] of the onboard pipeline).

Goal: decide *cheaply* what to do with a tile before any neural network runs.

    tile --> context  in {"cloud", "land_coast", "empty_sea", "candidates"}
         --> candidate boxes (possible ships) when context == "candidates"

Only techniques from the GI2 computer-vision course are used:
  * colour space HSV + histogram statistics      -> cloud check
  * white top-hat (morphology) + Otsu thresholding -> bright objects on the sea
  * opening                                       -> remove wave speckle
  * connected components                          -> candidate boxes
  * distance transform + watershed                -> split touching ships
  * Canny edge density                            -> coast / port hint

Every threshold lives in ``PrefilterConfig`` so it can be tuned on the
Airbus data with ``scripts/eval_prefilter.py``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .rle import Box

CLOUD, LAND, EMPTY, CANDIDATES = "cloud", "land_coast", "empty_sea", "candidates"


@dataclass
class PrefilterConfig:
    blur_ksize: int = 3                 # Gaussian denoising before everything else
    tophat_ksize: int = 31              # must be larger than the biggest ship (px)
    min_contrast: float = 12.0          # floor on the top-hat threshold (grey levels)
    k_sigma: float = 4.0                # threshold >= mean + k * std of the top-hat
    open_ksize: int = 3                 # opening kernel: removes 1-2 px wave speckle
    min_area: int = 12                  # smallest blob kept as a ship candidate (px)
    max_area: int = 6000                # bigger blobs are land / cloud, not a ship
    watershed_min_area: int = 400       # only try to split blobs bigger than this
    cloud_v_min: int = 170              # HSV value (brightness) of cloud pixels
    cloud_s_max: int = 40               # HSV saturation of cloud pixels
    cloud_frac: float = 0.90            # drop the whole tile only above this fraction;
                                        # below it, clouds are masked and the clear sea searched
    cloud_margin: int = 9               # px of dilation around clouds (their edges look like objects)
    cloud_min_size: int = 61            # opening kernel: bright regions smaller than this are
                                        # ships (white too!), not clouds
    land_delta: float = 45.0            # land = brighter than the sea median by this much
    land_blob_frac: float = 0.08        # tile is "land_coast" if big blobs cover this
    edge_density_land: float = 0.12     # ... or if Canny edges cover this fraction


@dataclass
class PrefilterResult:
    context: str
    boxes: list[Box] = field(default_factory=list)
    cloud_frac: float = 0.0
    edge_density: float = 0.0
    big_blob_frac: float = 0.0
    runtime_ms: float = 0.0


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _split_touching(blob: np.ndarray) -> list[np.ndarray]:
    """Watershed on the distance transform: split a blob into touching objects."""
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [blob]
    _, peaks = cv2.threshold(dist, 0.6 * dist.max(), 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(peaks)
    if n_markers <= 2:                        # background + one peak -> nothing to split
        return [blob]
    markers = markers + 1                     # background of the blob region becomes 1
    markers[(blob > 0) & (peaks == 0)] = 0    # unknown zone, watershed decides
    color = cv2.cvtColor(blob * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers.astype(np.int32))
    parts = []
    for label in range(2, n_markers + 1):
        part = ((markers == label) & (blob > 0)).astype(np.uint8)
        if part.any():
            parts.append(part)
    return parts or [blob]


def run_prefilter(img_bgr: np.ndarray, cfg: PrefilterConfig | None = None) -> PrefilterResult:
    """Classify one tile and extract ship candidates. ``img_bgr`` is an OpenCV BGR image."""
    cfg = cfg or PrefilterConfig()
    t0 = time.perf_counter()
    h, w = img_bgr.shape[:2]
    n_px = h * w

    # 1) Cloud check in HSV: bright and unsaturated pixels.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bright_grey = ((hsv[..., 2] >= cfg.cloud_v_min) & (hsv[..., 1] <= cfg.cloud_s_max)).astype(np.uint8)
    # Ships are bright and grey too: an opening larger than any ship keeps only big
    # bright regions (clouds), so ships are never masked out as "cloud".
    kcl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.cloud_min_size),) * 2)
    cloud_mask = cv2.morphologyEx(bright_grey, cv2.MORPH_OPEN, kcl) > 0
    cloud_frac = float(cloud_mask.mean())

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (_odd(cfg.blur_ksize),) * 2, 0)

    if cloud_frac >= cfg.cloud_frac:
        return PrefilterResult(CLOUD, [], cloud_frac, 0.0, 0.0, (time.perf_counter() - t0) * 1e3)

    # Partly cloudy: do not drop the tile (ships in the gaps would be lost).
    # Mask clouds plus a margin, and analyse only the clear pixels.
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(2 * cfg.cloud_margin + 1),) * 2)
    cloud_zone = cv2.dilate(cloud_mask.astype(np.uint8), kc) > 0
    clear = ~cloud_zone
    n_clear = max(1, int(clear.sum()))

    # 2) Canny edge density on clear pixels: coasts, ports and land are full of edges.
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(((edges > 0) & clear).sum() / n_clear)

    # 3) White top-hat = image - opening(image): keeps small bright objects,
    #    removes the slowly varying sea background (sun glint gradients, colour).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.tophat_ksize),) * 2)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)

    # 4) Threshold: Otsu, but never below mean + k*std (Otsu alone "finds" objects
    #    in pure noise because it always splits the histogram in two).
    sea_vals = tophat[clear]
    otsu_t, _ = cv2.threshold(sea_vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    stat_t = float(sea_vals.mean() + cfg.k_sigma * sea_vals.std())
    thr = max(otsu_t, stat_t, cfg.min_contrast)
    binary = ((tophat > thr) & clear).astype(np.uint8)

    # 5) Opening removes isolated wave pixels.
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.open_ksize),) * 2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ko)

    # 6) Big bright structures in the *raw* image (land, quays) -> coast / port context.
    #    Not Otsu here: on a uniform sea Otsu splits the illumination gradient in two
    #    halves. Land must be clearly brighter than the (dominant) sea level.
    raw_bin = ((gray > float(np.median(gray[clear])) + cfg.land_delta) & clear).astype(np.uint8)
    n_raw, _, raw_stats, _ = cv2.connectedComponentsWithStats(raw_bin, connectivity=8)
    big_area = sum(int(s[cv2.CC_STAT_AREA]) for s in raw_stats[1:] if s[cv2.CC_STAT_AREA] > cfg.max_area)
    big_blob_frac = big_area / n_clear

    # 7) Connected components -> candidate boxes; watershed splits touching ships.
    boxes: list[Box] = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < cfg.min_area or area > cfg.max_area:
            continue
        if area >= cfg.watershed_min_area:
            blob = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
            for part in _split_touching(blob):
                ys, xs = np.nonzero(part)
                if xs.size >= cfg.min_area:
                    boxes.append(Box(x + int(xs.min()), y + int(ys.min()),
                                     int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
        else:
            boxes.append(Box(x, y, bw, bh))

    if big_blob_frac >= cfg.land_blob_frac or edge_density >= cfg.edge_density_land:
        context = LAND
    elif boxes:
        context = CANDIDATES
    else:
        context = EMPTY
    return PrefilterResult(context, boxes, cloud_frac, edge_density, big_blob_frac,
                           (time.perf_counter() - t0) * 1e3)


def match_ships(gt: list[Box], pred: list[Box], iou_thr: float = 0.1) -> list[bool]:
    """For each ground-truth ship: is it covered by a candidate?

    A ship counts as found if a candidate overlaps it with IoU >= ``iou_thr`` or
    contains its centre (ships are tiny, so loose matching is appropriate for a
    *pre-filter* whose only job is not to lose ships).
    """
    found = []
    for g in gt:
        cx, cy = g.x + g.w / 2, g.y + g.h / 2
        found.append(any(p.iou(g) >= iou_thr or p.contains_point(cx, cy) for p in pred))
    return found
