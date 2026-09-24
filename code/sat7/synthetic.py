"""Synthetic sea tiles with known ships -- for unit tests and quick demos only.

Real results must come from the Airbus dataset (``scripts/eval_prefilter.py``).
"""
from __future__ import annotations

import cv2
import numpy as np

from .rle import Box, mask_to_box


def draw_ship(img: np.ndarray, centre: tuple[int, int], length: int, width: int, angle_deg: float,
              brightness: int = 215, mask: np.ndarray | None = None) -> None:
    """Draw a bright rotated rectangle (a ship seen from above) in place."""
    rect = ((float(centre[0]), float(centre[1])), (float(length), float(width)), float(angle_deg))
    pts = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(img, [pts], (brightness, brightness, brightness))
    if mask is not None:
        cv2.fillPoly(mask, [pts], 1)


def make_tile(n_ships: int = 3, cloud_cover: float = 0.0, size: int = 768, seed: int = 0,
              noise: float = 4.0) -> tuple[np.ndarray, list[Box]]:
    """Return (BGR image, ground-truth ship boxes)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # dark blue-green sea, slow illumination gradient, small waves, sensor noise
    base = np.array([62, 48, 22], np.float32)                          # BGR
    gradient = 10 * (xx / size) + 6 * (yy / size)
    waves = 3 * np.sin(xx / 7.0 + rng.uniform(0, 6)) * np.sin(yy / 11.0 + rng.uniform(0, 6))
    img = base[None, None, :] + (gradient + waves)[..., None] + rng.normal(0, noise, (size, size, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)

    boxes: list[Box] = []
    placed: list[tuple[int, int]] = []
    while len(boxes) < n_ships:
        c = (int(rng.integers(40, size - 40)), int(rng.integers(40, size - 40)))
        if any(abs(c[0] - p[0]) < 60 and abs(c[1] - p[1]) < 60 for p in placed):
            continue
        m = np.zeros((size, size), np.uint8)
        draw_ship(img, c, int(rng.integers(12, 45)), int(rng.integers(4, 10)),
                  float(rng.uniform(0, 180)), int(rng.integers(170, 240)), m)
        placed.append(c)
        boxes.append(mask_to_box(m))

    if cloud_cover > 0:
        field = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 40)
        field = (field - field.min()) / (np.ptp(field) + 1e-9)
        cloud = (field > np.quantile(field, 1 - cloud_cover)).astype(np.float32)
        cloud = cv2.GaussianBlur(cloud, (0, 0), 8)[..., None]
        img = (img * (1 - cloud) + 235 * cloud).astype(np.uint8)
    return img, boxes
