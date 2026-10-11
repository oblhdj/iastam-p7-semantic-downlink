#!/usr/bin/env bash
# Re-run every result that depends on LoDConfig. First written after adopting conf_high 0.9 -> 0.670
# (WP2); extended on 10 Oct 2026 when the coastal context tile went from a flat 32,420 B to each
# tile's measured size (wp28) and B1/B2 moved to the same-input run (wp26).
# Non-torch scripts run in .venv (3.14); WP11 and WP18 need torch so they run in .venv312.
# Run WP11 on an idle machine: it re-measures the stage timings that WP17 reads.
set -u
PY=./.venv/Scripts/python.exe
PY312=./.venv312/Scripts/python.exe
fail=0

run() {
  echo ""
  echo "=============== $* ==============="
  "$@" 2>&1 || { echo "!!! FAILED: $*"; fail=$((fail+1)); }
}

run $PY scripts/wp28_coast_tile_model.py              # add --measure to re-encode the tiles (needs the data)
run $PY scripts/wp6_simulate_real.py
run $PY scripts/wp6_simulate_real.py --mix dataset --repeats 1 --tag datasetmix
run $PY scripts/wp6_simulate_real.py --no-size-model --repeats 1 --tag nosizemodel
run $PY scripts/wp6_simulate_real.py --coast-model wp7 --tag modeledcoast   # the earlier 557x estimate
run $PY scripts/wp6_simulate_real.py --semantic priority --tag paper     # the paper's Table I P0-P3 (PHASE3 2b)
run $PY scripts/wp7_budget_sweep.py
run $PY scripts/wp8_queue_aware.py
run $PY scripts/wp5_optimality_gap.py --frac-steps 16
run $PY scripts/wp5_joint_bound.py
run $PY scripts/wp10_sensitivity.py
run $PY scripts/wp10b_interaction.py
run $PY scripts/wp14_gate_signal.py
run $PY scripts/wp23_semantic_compare.py
run $PY312 -W ignore scripts/wp11_integration_demo.py --tiles 5320
run $PY scripts/wp17_energy_model.py
run $PY scripts/wp19_relay_energy.py
run $PY312 -W ignore scripts/wp18_campaign_runner.py  # reads wp26_b0_b4.json for B1 / B2
run $PY312 -W ignore scripts/wp18_campaign_runner.py --semantic priority --tag paper   # B3 / B4 under P0-P3
run $PY scripts/wp24_detection_eval.py --live-b1  # needs onnxruntime (not in .venv)
run $PY scripts/wp25_semantic_packets.py
run $PY scripts/wp27_comms_direct_vs_relay.py
run $PY scripts/wp30_paper_ablations.py               # six paper ablations + curves; --rerun-detection re-measures wp26 / wp24
run $PY scripts/wp29_validation.py                    # checks all of the above; regenerates none of it

echo ""
echo "=============== DONE: $fail failure(s) ==============="
exit $fail
