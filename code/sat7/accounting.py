"""Units and the three bookkeeping definitions every result uses: data reduction, energy, latency.

Nothing here is a model. It states, in one place and with the units spelled out, the arithmetic
that sat7.energy, sat7.comms, scripts/wp17 and scripts/wp19 each carry inline, so that tests can
check they all mean the same thing (tests/test_validation.py) and so that latency has one written
convention.

Units, repo-wide
----------------
  data volume   bytes. 1 MB = 10^6 bytes (decimal), never 2^20.
  link rate     bit/s. A volume in bytes is multiplied by 8 before it is divided by a rate.
  power         watts.   time: seconds (stage timings are stored in ms and divided by 1000).
  energy        joules = watts x seconds. kJ only for display.

Data reduction (paper eq 27)
----------------------------
    Data_reduction_percent = 100 x (1 - D_transmitted / D_raw)
  D_transmitted  the bytes of the representation that is actually sent.
  D_raw          the full-image baseline: every imaged pixel as uncompressed 8-bit RGB,
                 H x W x 3 bytes (768 x 768 x 3 = 1,769,472 B per tile).

Energy (paper section V)
------------------------
    E_total         = E_processing + E_communication
    E_processing    = sum_k P_k x T_k                    per onboard stage k
    E_communication = P_transmission x T_transmission,   T_transmission = bytes x 8 / rate
  No power was measured in this project: every P is an ASSUMPTION. Stage times T_k are measured
  on a laptop; transmit times come from simulated links. Every joule is therefore an ESTIMATE.

Latency (paper eq 28) -- the convention
---------------------------------------
    T_total = T_processing + T_encoding + T_communication + T_decoding
  The stages are SEQUENTIAL for one item and are added, never overlapped:

    capture --T_processing--> detections --T_encoding--> packet queued
            --T_communication--> last byte at the ground station --T_decoding--> usable on ground

  * T_processing and T_encoding are per tile, a few tens of ms. Tiles arrive seconds apart
    (40,000 tiles/day = one every 2.16 s), so stage k of one tile overlaps the CAPTURE of the next
    but never its own later stages, and no processing queue builds up.
  * T_communication is what the simulators report as "latency": from the capture time to the
    end of that item's transmission. It contains the wait for a contact window and the transmission
    itself. On the relay path it contains BOTH legs: wait for an inter-satellite window + ISL
    transmission + storage on the relay until its ground pass + relay-to-ground transmission.
  * The simulators treat an item as queued at its capture time. Adding T_processing + T_encoding
    (about 0.05 s) would delay the queueing by that much, which can only matter for a tile captured
    within that margin of a pass; the stages are instead added on top, as below.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

BITS_PER_BYTE = 8
MB = 1e6                                  # bytes in a megabyte, decimal
RAW_BYTES_PER_PIXEL = 3                   # uncompressed 8-bit RGB


def raw_image_bytes(height_px: int, width_px: int) -> int:
    """D_raw of one image: every pixel as uncompressed 8-bit RGB."""
    return int(height_px) * int(width_px) * RAW_BYTES_PER_PIXEL


def data_reduction_percent(d_transmitted_bytes: float, d_raw_bytes: float) -> float:
    """100 x (1 - D_transmitted / D_raw). Both in the same unit (bytes, or both MB)."""
    if d_raw_bytes <= 0:
        raise ValueError("D_raw must be positive")
    if d_transmitted_bytes < 0:
        raise ValueError("D_transmitted cannot be negative")
    return 100.0 * (1.0 - d_transmitted_bytes / d_raw_bytes)


def reduction_factor(d_transmitted_bytes: float, d_raw_bytes: float) -> float:
    """D_raw / D_transmitted -- the same statement as data_reduction_percent, as 'N x'."""
    if d_transmitted_bytes <= 0:
        raise ValueError("D_transmitted must be positive to state a factor")
    return d_raw_bytes / d_transmitted_bytes


def tx_time_s(n_bytes: float, rate_bps: float) -> float:
    """T_transmission in seconds: bytes -> bits -> seconds at a rate in bit/s."""
    if rate_bps <= 0:
        raise ValueError("link rate must be positive (bit/s)")
    return n_bytes * BITS_PER_BYTE / rate_bps


def tx_energy_J(power_W: float, n_bytes: float, rate_bps: float) -> float:
    """E_communication = P_transmission x T_transmission, in joules."""
    return power_W * tx_time_s(n_bytes, rate_bps)


def stage_energy_J(power_W: float, time_ms: float) -> float:
    """One term of E_processing = sum_k P_k x T_k, with T_k given in milliseconds."""
    return power_W * time_ms / 1000.0


@dataclass(frozen=True)
class LatencyBudget:
    """T_total = T_processing + T_encoding + T_communication + T_decoding, all in seconds."""
    processing_s: float
    encoding_s: float
    communication_s: float
    decoding_s: float

    @property
    def total_s(self) -> float:
        return self.processing_s + self.encoding_s + self.communication_s + self.decoding_s

    @property
    def non_communication_s(self) -> float:
        return self.processing_s + self.encoding_s + self.decoding_s

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total_s"] = self.total_s
        d["non_communication_share"] = (self.non_communication_s / self.total_s) if self.total_s else 0.0
        return d


def measured_stage_times_s(results_dir: str | Path, build: str = "cpu_onnx",
                           calls_per_tile: float = 1.0) -> dict:
    """The per-tile stage times this repo has measured, in seconds, with where each comes from.

    build            "cpu_onnx" (the stated onboard build) or "gpu" (the laptop GPU budget)
    calls_per_tile   detector calls per tile: 1.0 for one call per tile, 1.5625 for SAHI's 25
                     windows over a 4x4-tile scene
    Times are REAL but measured on a laptop, not flight hardware.
    """
    d = Path(results_dir)
    e17 = json.loads((d / "wp17_energy_model.json").read_text(encoding="utf-8"))
    w25 = json.loads((d / "wp25_semantic_packets.json").read_text(encoding="utf-8"))["timing_ms_per_tile"]
    t = e17["stage_times_ms"]
    if build == "cpu_onnx":
        gate, detect = t["gate_cpu"], t["detect_cpu_onnx"]
    elif build == "gpu":
        gate, detect = t["gate_gpu"], t["detect_gpu"]
    else:
        raise ValueError(f"unknown build {build!r}")
    enc = w25["T_enc_onboard_median"]
    return {
        "processing_s": (t["preprocess_decode_cpu"] + gate + detect * calls_per_tile) / 1000.0,
        "encoding_s": {ctx: ms / 1000.0 for ctx, ms in enc.items()},
        "decoding_s": w25["T_dec_ground_median"] / 1000.0,
        "source": {"processing": "results/wp17_energy_model.json stage_times_ms (preprocess + gate + "
                                 f"detect x {calls_per_tile:g}), build {build}",
                   "encoding": "results/wp25_semantic_packets.json timing_ms_per_tile.T_enc_onboard_median",
                   "decoding": "results/wp25_semantic_packets.json timing_ms_per_tile.T_dec_ground_median"},
        "label": "REAL timings on a laptop (not flight hardware); preprocess_decode is not sourced "
                 "from a results file (wp17 unsourced_figures)",
    }
