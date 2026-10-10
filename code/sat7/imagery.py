"""Image input for the perception stage: load one satellite image, validate it, record its size.

Every detector path in the repo reads tiles with a bare `cv2.imread` and trusts them. A single-image
entry point (scripts/detect_image.py, demo/dashboard.py) needs the checks a real input deserves:
a supported container, an image that actually decodes, a pixel format the detector was trained on
(8-bit), a usable size -- and it must report two different "sizes" without confusing them (audit
H6: the dashboard divided by the JPEG's file size and called it raw):

  * file_bytes  -- what is on disk (compressed: JPEG / PNG / TIFF ...)
  * raw_bytes   -- H x W x 3 bytes of 8-bit RGB: the bent-pipe (B0) definition the whole repo uses
                   (sat7.scheduler.RAW_TILE_BYTES = 768 * 768 * 3 for one Airbus tile)

Output is always H x W x 3 uint8 BGR (OpenCV order), the input every sat7 detect_fn expects.
Greyscale is replicated to 3 channels and an alpha channel dropped (both noted); 16-bit or float
imagery is rejected rather than silently rescaled -- the detector was trained on 8-bit Airbus RGB,
and a stretch picked here would change what it sees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SUPPORTED = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF",
             ".bmp": "BMP"}
_MAGIC = ((b"\xff\xd8\xff", "JPEG"), (b"\x89PNG\r\n\x1a\n", "PNG"), (b"II*\x00", "TIFF"),
          (b"MM\x00*", "TIFF"), (b"BM", "BMP"))
MIN_SIDE = 16           # below this a 768-input detector sees an upscaled smear, not a scene


@dataclass
class ImageInfo:
    name: str
    format: str               # container sniffed from the file's own bytes
    width: int
    height: int
    channels_in: int          # as stored (1 = grey, 3 = colour, 4 = with alpha)
    dtype_in: str
    file_bytes: int           # size on disk (compressed)
    raw_bytes: int            # H x W x 3, 8-bit: the B0 raw-downlink definition
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.name}: {self.format}, {self.width} x {self.height} px, "
                f"{self.channels_in} channel(s) {self.dtype_in}; file {self.file_bytes:,} B, "
                f"raw (H x W x 3) {self.raw_bytes:,} B")


def sniff_format(data: bytes) -> str | None:
    for magic, fmt in _MAGIC:
        if data.startswith(magic):
            return fmt
    return None


def decode_image(data: bytes, name: str = "<bytes>", min_side: int = MIN_SIDE):
    """Validate + decode image bytes -> (H x W x 3 uint8 BGR, ImageInfo). Raises ValueError."""
    if not data:
        raise ValueError(f"{name}: empty file")
    fmt = sniff_format(data)
    ext = Path(name).suffix.lower()
    notes = []
    if fmt is None:
        raise ValueError(f"{name}: not a supported image (expected one of "
                         f"{sorted(set(SUPPORTED.values()))})")
    if ext in SUPPORTED and SUPPORTED[ext] != fmt:
        notes.append(f"extension {ext} but the bytes are {fmt}; decoded as {fmt}")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"{name}: {fmt} header found but the image does not decode (truncated or corrupt)")
    if img.dtype != np.uint8:
        raise ValueError(f"{name}: {img.dtype} pixels; the detector was trained on 8-bit RGB -- "
                         f"convert to 8-bit first (no automatic stretch is applied)")
    channels = 1 if img.ndim == 2 else img.shape[2]
    if channels == 1:
        img = cv2.cvtColor(img if img.ndim == 2 else img[..., 0], cv2.COLOR_GRAY2BGR)
        notes.append("greyscale replicated to 3 channels")
    elif channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        notes.append("alpha channel dropped")
    elif channels != 3:
        raise ValueError(f"{name}: {channels} channels; expected 1, 3 or 4")
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        raise ValueError(f"{name}: {w} x {h} px is below the {min_side} px minimum side")
    info = ImageInfo(Path(name).name, fmt, w, h, channels, "uint8", len(data), h * w * 3, notes)
    return np.ascontiguousarray(img), info


def load_image(path, min_side: int = MIN_SIDE):
    """Validate + load an image file -> (H x W x 3 uint8 BGR, ImageInfo). Raises ValueError."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such image: {p}")
    if p.suffix.lower() not in SUPPORTED:
        raise ValueError(f"{p.name}: unsupported extension {p.suffix!r} "
                         f"(supported: {', '.join(sorted(SUPPORTED))})")
    return decode_image(p.read_bytes(), str(p), min_side)
