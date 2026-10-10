"""Calibratable per-stage energy model (paper section V, eqs 8-19), with a hardware-injection seam.

`scripts/wp17_energy_model.py` built the paper's energy model but hardcodes its power assumptions in
module globals and does the arithmetic inline, so swapping in *measured* onboard power numbers means
editing the script. This module lifts the model into `sat7`, parameterised, so real P_k and T_k drop
in through one constructor and nothing else changes:

    # defaults: ASSUMPTION powers (swept), REAL stage times read from the wp result files
    em = EnergyModel()

    # calibration: inject measured edge-hardware numbers (the whole point of this module)
    em = EnergyModel.from_measurements(
        powers_W={"cpu": 5.1, "gpu": 9.0, "tx": 11.3},         # e.g. a Jetson + an S-band radio
        stage_times_ms={"detect_cpu_onnx": 95.0, "gate_cpu": 1.2, ...},
        r_tx_bps=1.0e6, provenance={"cpu": "REAL (measured, Jetson Orin Nano)", ...})

The model carries a provenance label per quantity so an honest figure never loses its REAL / SIM /
ASSUMPTION / TARGET tag (the project rule). Nothing here runs a detector or reads a results file by
itself; `EnergyModel.from_results(dir)` is a convenience that reads the same live files wp17 does.

The equations (paper section V):
    E_proc  = sum_k P_k * T_k           (preprocess / gate-or-prefilter / detect / fuse / semantic)
    E_comm,direct = P_tx * D_tx / R_tx
    E_comm,relay  = E_ISL + E_GS        (the relay leg, supplied to sat7.relay.route)
    E_total = E_proc + min(E_comm,direct, E_comm,relay)
    ES      = 1 - E_proposed / E_baseline
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# default per-stage times (ms/tile). ASSUMPTION placeholders here; `from_results` overwrites each
# with the REAL measured value read live from the wp file that owns it (see wp17 for provenance).
DEFAULT_TIMES_MS = {
    "preprocess_decode_cpu": 0.71,
    "classic_prefilter_cpu": 51.1,
    "gate_gpu": 0.30,
    "gate_cpu": 1.20,
    "detect_gpu": 11.0,
    "detect_cpu_onnx": 38.6,
    "fuse_cpu": 0.0,
    "semantic_cpu": 0.5,
}
# default powers (W). ASSUMPTION, swept (report 17). Replace via from_measurements for real hardware.
DEFAULT_POWERS_W = {"cpu": 28.0, "gpu": 60.0, "tx": 15.0}
DEFAULT_R_TX_BPS = 2.53e6          # SIM: effective downlink rate (orbit sim, report 17/19)

# which (stage -> processor) each pipeline config runs on. Mirrors wp17.e_proc_per_tile.
CONFIG_STAGES = {
    "cpu_onnx": {"preprocess": ("preprocess_decode_cpu", "cpu"),
                 "gate_or_prefilter": ("gate_cpu", "cpu"), "detect": ("detect_cpu_onnx", "cpu"),
                 "fuse": ("fuse_cpu", "cpu"), "semantic": ("semantic_cpu", "cpu")},
    "gpu": {"preprocess": ("preprocess_decode_cpu", "cpu"),
            "gate_or_prefilter": ("gate_gpu", "gpu"), "detect": ("detect_gpu", "gpu"),
            "fuse": ("fuse_cpu", "cpu"), "semantic": ("semantic_cpu", "cpu")},
}
# the baseline (no onboard gate) runs the classic prefilter where the proposed pipeline runs the gate
BASELINE_GATE_STAGE = {"cpu_onnx": ("classic_prefilter_cpu", "cpu"),
                       "gpu": ("classic_prefilter_cpu", "cpu")}


@dataclass
class EnergyModel:
    """Per-stage energy model with injectable, labelled power/time/rate quantities."""
    powers_W: dict = field(default_factory=lambda: dict(DEFAULT_POWERS_W))
    stage_times_ms: dict = field(default_factory=lambda: dict(DEFAULT_TIMES_MS))
    r_tx_bps: float = DEFAULT_R_TX_BPS
    provenance: dict = field(default_factory=lambda: {
        "powers_W": "ASSUMPTION (swept, report 17); replace with from_measurements for real hw",
        "stage_times_ms": "DEFAULT placeholders; from_results reads the REAL measured values live",
        "r_tx_bps": "SIM (orbit sim, report 17)"})

    # ---- constructors ----------------------------------------------------------------------------
    @classmethod
    def from_measurements(cls, powers_W: dict, stage_times_ms: dict | None = None,
                          r_tx_bps: float | None = None, provenance: dict | None = None) -> "EnergyModel":
        """Inject calibrated edge-hardware numbers. Unspecified stage times keep the defaults, so a
        partial calibration (e.g. only the detector timed on real hardware) still runs; the
        provenance should say which quantities are measured and which remain assumed."""
        times = dict(DEFAULT_TIMES_MS)
        if stage_times_ms:
            times.update(stage_times_ms)
        prov = {"powers_W": "REAL (measured, injected)", "stage_times_ms": "mixed: see per-stage note",
                "r_tx_bps": "REAL (measured, injected)" if r_tx_bps else "SIM (default)"}
        if provenance:
            prov.update(provenance)
        return cls(powers_W=dict(powers_W), stage_times_ms=times,
                   r_tx_bps=r_tx_bps or DEFAULT_R_TX_BPS, provenance=prov)

    @classmethod
    def from_results(cls, results_dir: str | Path, powers_W: dict | None = None) -> "EnergyModel":
        """Read the REAL measured stage times live from the same result files wp17 uses.

        Keeps powers as the (assumption) default unless given. Each time is pulled from its
        authoritative producer -- 'the .csv/.json files always win' (START_HERE section 7)."""
        d = Path(results_dir)
        times = dict(DEFAULT_TIMES_MS)

        def rd(name):
            p = d / name
            return json.loads(p.read_text()) if p.exists() else None
        gate, det_m, det_x, integ = rd("wp3_gate.json"), rd("wp1_metrics.json"), rd("wp1_export.json"), rd("wp11_integration.json")
        if integ:
            times["classic_prefilter_cpu"] = float(integ["stage_ms_per_tile"]["prefilter"])
        if gate:
            times["gate_gpu"] = float(gate["infer_ms_per_tile"])
            times["gate_cpu"] = float(gate["infer_ms_per_tile_cpu"])
        if det_m:
            times["detect_gpu"] = float(det_m["inference_ms_per_tile_gpu"])
        if det_x:
            times["detect_cpu_onnx"] = float(det_x["latency_ms_per_tile"]["onnx_fp32_cpu"])
        prov = {"powers_W": "ASSUMPTION (swept)" if not powers_W else "REAL (injected)",
                "stage_times_ms": "REAL live from wp result files (wp17 provenance)",
                "r_tx_bps": "SIM (orbit sim)"}
        return cls(powers_W=powers_W or dict(DEFAULT_POWERS_W), stage_times_ms=times, provenance=prov)

    # ---- equations -------------------------------------------------------------------------------
    def _stage_energy_J(self, time_key: str, processor: str) -> float:
        return self.powers_W[processor] * self.stage_times_ms[time_key] / 1000.0

    def e_proc_per_tile_J(self, config: str = "cpu_onnx") -> dict:
        """E_proc per tile by stage, for the baseline and proposed pipelines of one config.

        Baseline = detector + classic prefilter (no onboard gate); proposed = detector + learned
        gate. The detector is counted in both, so it cancels in ES_proc (report 13 convention)."""
        stages = CONFIG_STAGES[config]
        prop = {name: self._stage_energy_J(tk, pr) for name, (tk, pr) in stages.items()}
        base = dict(prop)
        btk, bpr = BASELINE_GATE_STAGE[config]
        base["gate_or_prefilter"] = self._stage_energy_J(btk, bpr)
        return {"baseline": base, "proposed": prop,
                "E_base": sum(base.values()), "E_prop": sum(prop.values())}

    def e_comm_direct_J(self, d_bytes: float) -> float:
        """E_comm,direct = P_tx * D_tx / R_tx, with D_tx in BITS (d_bytes * 8) and R_tx in bit/s.

        Until 10 Oct 2026 this divided bytes by bit/s, so every joule here was 8x too small against
        wp17 / wp19, which always converted (5.12 kJ/day for 107.95 MB at 15 W, 2.53 Mbps)."""
        return self.powers_W["tx"] * d_bytes * 8.0 / self.r_tx_bps

    def e_comm_relay_J(self, d_bytes: float, p_isl_W: float = 12.0,
                       r_isl_bps: float | None = None) -> dict:
        """E_comm,relay = E_ISL + E_GS (paper eq). The ISL leg keys the inter-sat radio at p_isl_W
        over r_isl_bps; the ground leg is the ordinary direct cost from the relay. p_isl_W is
        ASSUMPTION (report 19)."""
        r_isl = r_isl_bps or self.r_tx_bps
        e_isl = p_isl_W * d_bytes * 8.0 / r_isl
        e_gs = self.e_comm_direct_J(d_bytes)
        return {"E_ISL_J": e_isl, "E_GS_J": e_gs, "E_relay_J": e_isl + e_gs}

    @staticmethod
    def energy_saving_ratio(baseline_J: float, proposed_J: float) -> float:
        """ES = 1 - E_proposed / E_baseline (paper eq). Positive = the proposed pipeline saves."""
        return 1.0 - proposed_J / baseline_J if baseline_J else 0.0

    # ---- adapters for sat7.relay.route (so a calibrated model supplies the real energy callables) -
    def direct_energy_fn(self):
        """Returns a `direct_energy(d_bytes, p_tx, r_gs_bps)` matching sat7.relay's signature."""
        def f(d_bytes, p_tx, r_gs_bps):
            return {"E_comm_J": p_tx * d_bytes * 8.0 / r_gs_bps, "tx_time_s": d_bytes * 8.0 / r_gs_bps,
                    "label": "REAL" if "REAL" in self.provenance.get("powers_W", "") else "ASSUMPTION"}
        return f

    def relay_energy_fn(self):
        """Returns a `relay_energy(d_bytes, p_tx, p_isl, r_gs_bps, r_isl_bps)` for sat7.relay."""
        def f(d_bytes, p_tx, p_isl, r_gs_bps, r_isl_bps):
            e_isl = p_isl * d_bytes * 8.0 / r_isl_bps
            e_gs = p_tx * d_bytes * 8.0 / r_gs_bps
            return {"E_ISL_J": e_isl, "E_GS_J": e_gs, "E_relay_J": e_isl + e_gs,
                    "isl_time_s": d_bytes * 8.0 / r_isl_bps, "gs_time_s": d_bytes * 8.0 / r_gs_bps,
                    "label": "REAL" if "REAL" in self.provenance.get("powers_W", "") else "ASSUMPTION"}
        return f


if __name__ == "__main__":
    # Self-check: the equations, the ES sign, and that calibration actually moves the numbers.
    em = EnergyModel()
    proc = em.e_proc_per_tile_J("cpu_onnx")
    es = em.energy_saving_ratio(proc["E_base"], proc["E_prop"])
    print(f"default cpu_onnx: E_base {proc['E_base']:.4f} J  E_prop {proc['E_prop']:.4f} J  ES_proc {es:+.1%}")
    assert proc["E_prop"] < proc["E_base"] and es > 0          # the learned gate must save

    # all-CPU config: P_cpu cancels in ES_proc -> power-independent (wp17's robustness claim)
    em_hi = EnergyModel(powers_W={"cpu": 45.0, "gpu": 60.0, "tx": 15.0})
    es_hi = em_hi.energy_saving_ratio(*(em_hi.e_proc_per_tile_J("cpu_onnx")[k] for k in ("E_base", "E_prop")))
    assert abs(es - es_hi) < 1e-9, (es, es_hi)
    print(f"ES_proc(cpu_onnx) power-independent: {es:.4f} == {es_hi:.4f} at P_cpu 28 vs 45 W OK")

    # comm: direct vs relay, and the relay = ISL + GS identity
    d = 1e6
    direct = em.e_comm_direct_J(d)
    relay = em.e_comm_relay_J(d)
    assert abs(relay["E_relay_J"] - (relay["E_ISL_J"] + relay["E_GS_J"])) < 1e-12
    print(f"comm for 1 MB: direct {direct:.3f} J | relay {relay['E_relay_J']:.3f} J (ISL {relay['E_ISL_J']:.3f}+GS {relay['E_GS_J']:.3f})")

    # calibration seam: injecting a low-power edge chip changes absolute joules
    cal = EnergyModel.from_measurements(powers_W={"cpu": 5.0, "gpu": 9.0, "tx": 11.0}, r_tx_bps=1e6)
    assert cal.e_comm_direct_J(d) != direct
    print(f"calibrated (5 W CPU, 11 W TX, 1 Mbps): comm for 1 MB {cal.e_comm_direct_J(d):.3f} J")
    print("energy self-check OK")
