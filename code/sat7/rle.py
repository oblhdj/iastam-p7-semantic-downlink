"""Airbus Ship Detection helpers: run-length encoding, masks and bounding boxes.

Airbus format (train_ship_segmentations_v2.csv):
  * one row per ship, columns ``ImageId, EncodedPixels``;
  * images with no ship have one row with an empty ``EncodedPixels``;
  * pixels are numbered from 1, top-to-bottom then left-to-right
    (column-major), as "start length start length ...".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AIRBUS_SHAPE = (768, 768)  # (height, width)


def rle_decode(rle: str | float | None, shape: tuple[int, int] = AIRBUS_SHAPE) -> np.ndarray:
    """Decode one Airbus RLE string into a uint8 mask of ``shape`` (1 = ship)."""
    h, w = shape
    flat = np.zeros(h * w, dtype=np.uint8)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or not str(rle).strip():
        return flat.reshape((h, w), order="F")
    nums = np.asarray(str(rle).split(), dtype=np.int64)
    starts, lengths = nums[0::2] - 1, nums[1::2]
    for s, n in zip(starts, lengths):
        flat[s:s + n] = 1
    return flat.reshape((h, w), order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask into an Airbus RLE string ('' if empty)."""
    flat = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    padded = np.concatenate([[0], flat, [0]])
    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels: top-left (x, y), width, height."""
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_yolo(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Normalised (cx, cy, w, h) as used by YOLO label files."""
        return ((self.x + self.w / 2) / img_w, (self.y + self.h / 2) / img_h,
                self.w / img_w, self.h / img_h)

    def iou(self, other: "Box") -> float:
        ix = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


def mask_to_box(mask: np.ndarray) -> Box | None:
    """Tight bounding box of the non-zero pixels, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return Box(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def rles_to_boxes(rles, shape: tuple[int, int] = AIRBUS_SHAPE) -> list[Box]:
    """One box per ship RLE string (empty strings / NaN are skipped)."""
    boxes = []
    for rle in rles:
        box = mask_to_box(rle_decode(rle, shape))
        if box is not None:
            boxes.append(box)
    return boxes
