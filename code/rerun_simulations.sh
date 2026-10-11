#!/usr/bin/env bash
# The simulated experiments: every script that needs ONLY the committed catalogues in results/.
# No dataset, no weights, no torch, no GPU; the analysis environment (requirements.txt) is enough.
#
#   cd code && bash rerun_simulations.sh                 # about 8.5 minutes on the development laptop
#   PY=python bash rerun_simulations.sh                  # with another interpreter
#
# It rewrites its outputs in results/ in place. They are deterministic: run from a fresh clone on
# 10 Oct 2026, all 19 runs exited 0 and `git status` showed no tracked file changed. The last lines
# below repeat that check, so a difference after a code change is visible at once.
#
# Not here, because they read the Airbus split or run the detector: wp25 and wp27 (real packets from
# real tiles), wp11, wp18, wp24, wp26 and everything else in rerun_all.sh, which is the full list.
set -u
PY="${PY:-./.venv/Scripts/python.exe}"
fail=0

run() {
  echo ""
  echo "=============== $* ==============="
  "$PY" -W ignore "$@" 2>&1 || { echo "!!! FAILED: $*"; fail=$((fail+1)); }
}

run scripts/contact_windows.py                        # orbit passes over the ground station
run scripts/wp6_fit_size_model.py                     # crop-size power law
run scripts/wp28_coast_tile_model.py                  # coastal sizes from the committed table (no --measure)
run scripts/wp6_simulate_real.py                      # the headline day: 341x, 0.685
run scripts/wp6_simulate_real.py --mix dataset --repeats 1 --tag datasetmix
run scripts/wp6_simulate_real.py --no-size-model --repeats 1 --tag nosizemodel
run scripts/wp6_simulate_real.py --coast-model wp7 --tag modeledcoast   # the earlier 557x estimate
run scripts/wp7_budget_sweep.py
run scripts/wp8_queue_aware.py
run scripts/wp5_optimality_gap.py --frac-steps 16
run scripts/wp5_joint_bound.py
run scripts/wp10_sensitivity.py                       # the 80-setting sweep
run scripts/wp10b_interaction.py
run scripts/wp14_gate_signal.py
run scripts/wp23_semantic_compare.py                  # P0-P3 against the level-of-detail ladder
run scripts/wp17_energy_model.py                      # energy, default and the SAHI variant
run scripts/wp19_relay_energy.py
run scripts/wp20_relay_path_choice.py
run scripts/wp30_paper_ablations.py                   # six paper ablations + three trade-off curves (pure P0-P3)
run scripts/wp29_validation.py                        # checks all of the above; regenerates none of it

echo ""
echo "=============== DONE: $fail failure(s) ==============="
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  changed=$(git status --short -- results | wc -l)
  echo "tracked result files that now differ from the commit: $changed"
  [ "$changed" -gt 0 ] && git status --short -- results
fi
exit $fail
