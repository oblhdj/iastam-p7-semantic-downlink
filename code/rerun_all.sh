#!/usr/bin/env bash
# Re-run every result that depends on LoDConfig, after adopting conf_high 0.9 -> 0.670 (WP2).
# Non-torch scripts run in .venv (3.14); WP11 needs torch so it runs in .venv312.
set -u
PY=./.venv/Scripts/python.exe
PY312=./.venv312/Scripts/python.exe
fail=0

run() {
  echo ""
  echo "=============== $* ==============="
  "$@" 2>&1 || { echo "!!! FAILED: $*"; fail=$((fail+1)); }
}

run $PY scripts/wp6_simulate_real.py
run $PY scripts/wp6_simulate_real.py --mix dataset --repeats 1 --tag datasetmix
run $PY scripts/wp6_simulate_real.py --no-size-model --repeats 1 --tag nosizemodel
run $PY scripts/wp7_budget_sweep.py
run $PY scripts/wp8_queue_aware.py
run $PY scripts/wp5_optimality_gap.py --frac-steps 16
run $PY scripts/wp5_joint_bound.py
run $PY scripts/wp10_sensitivity.py
run $PY312 -W ignore scripts/wp11_integration_demo.py --tiles 400

echo ""
echo "=============== DONE: $fail failure(s) ==============="
exit $fail
