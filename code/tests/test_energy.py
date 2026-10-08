from sat7.energy import DEFAULT_POWERS_W, EnergyModel


def test_learned_gate_saves_processing_energy():
    em = EnergyModel()
    proc = em.e_proc_per_tile_J("cpu_onnx")
    assert proc["E_prop"] < proc["E_base"]
    assert em.energy_saving_ratio(proc["E_base"], proc["E_prop"]) > 0


def test_es_proc_is_power_independent_for_all_cpu_build():
    # all-CPU config: P_cpu cancels in the ratio, so ES_proc must not move with P_cpu (wp17 claim)
    def es(p_cpu):
        em = EnergyModel(powers_W={"cpu": p_cpu, "gpu": 60.0, "tx": 15.0})
        p = em.e_proc_per_tile_J("cpu_onnx")
        return em.energy_saving_ratio(p["E_base"], p["E_prop"])
    assert abs(es(15.0) - es(45.0)) < 1e-9


def test_comm_direct_is_linear_in_bytes():
    em = EnergyModel()
    assert abs(em.e_comm_direct_J(2e6) - 2 * em.e_comm_direct_J(1e6)) < 1e-9


def test_relay_energy_is_isl_plus_gs():
    em = EnergyModel()
    r = em.e_comm_relay_J(1e6)
    assert abs(r["E_relay_J"] - (r["E_ISL_J"] + r["E_GS_J"])) < 1e-12
    assert r["E_relay_J"] > em.e_comm_direct_J(1e6)          # relay adds the ISL leg on top


def test_from_measurements_injects_and_labels():
    em = EnergyModel.from_measurements(powers_W={"cpu": 5.0, "gpu": 9.0, "tx": 11.0},
                                       stage_times_ms={"detect_cpu_onnx": 95.0}, r_tx_bps=1e6)
    assert em.powers_W["cpu"] == 5.0 and em.stage_times_ms["detect_cpu_onnx"] == 95.0
    assert em.r_tx_bps == 1e6 and "REAL" in em.provenance["powers_W"]
    # unspecified stage times keep defaults (partial calibration still runs)
    assert em.stage_times_ms["gate_cpu"] == EnergyModel().stage_times_ms["gate_cpu"]


def test_relay_adapter_matches_relay_signature():
    em = EnergyModel.from_measurements(powers_W=dict(DEFAULT_POWERS_W))
    direct = em.direct_energy_fn()
    relay = em.relay_energy_fn()
    d = direct(1e6, 15.0, 2.53e6)
    r = relay(1e6, 15.0, 12.0, 2.53e6, 2.53e6)
    assert "E_comm_J" in d and "E_relay_J" in r
    assert abs(r["E_relay_J"] - (r["E_ISL_J"] + r["E_GS_J"])) < 1e-12


def test_relay_adapter_drops_into_route():
    # the whole point: a calibrated model supplies sat7.relay.route's energy callables unchanged
    from datetime import datetime, timezone
    from sat7.orbit import Pass
    em = EnergyModel()
    t0 = datetime(2026, 9, 18, tzinfo=timezone.utc)
    direct_passes = [Pass(t0, t0, t0, 45.0, 1e9)]
    from sat7.relay import route
    out = route([{"bytes": 1e6, "t_now": t0}], (direct_passes, [], []),
                lam_E=1.0, lam_T=0.0, isl_enabled=False,
                direct_energy=em.direct_energy_fn(), relay_energy=em.relay_energy_fn())
    assert out[0]["path"] == "direct" and out[0]["E"] > 0
