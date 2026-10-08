#!/usr/bin/env bash
# IASTAM P7 -- one-command end-to-end DEMO.
#
# Runs the two spine scripts live on a small sample, then prints the consolidated summary:
#   1. wp18_campaign_runner.py   -> the B0->B4 progression (B0..B4, ~25 s)
#   2. wp11_integration_demo.py  -> real JPEGs through the real chain + integrity check
#   3. demo/summary.py           -> one presentation-ready account of all the numbers
#
# Needs the torch GPU env (.venv312) and the dataset in code/data/. If you do NOT have those,
# skip this and just run:   python demo/summary.py   -- it reports the committed results.
#
#   bash demo/run_demo.sh            # small live sample (fast; good for a live demo)
#   bash demo/run_demo.sh --full     # full sample (5320 tiles; reproduces the headline exactly)
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/code"
PY312=./.venv312/Scripts/python.exe
TILES=400
[ "${1:-}" = "--full" ] && TILES=5320
# Live outputs go to a scratch dir so a small sample NEVER overwrites the authoritative
# results in code/results/. summary.py always reports the committed full-run canon.
LIVE_OUT="../demo/_live"
mkdir -p "$LIVE_OUT"

if [ ! -x "$PY312" ]; then
  echo ""
  echo "!!! torch env not found at code/.venv312 -- cannot run the LIVE pipeline here."
  echo "    Showing the committed results instead:"
  echo ""
  (cd "$REPO" && python.exe demo/summary.py || python demo/summary.py)
  exit 0
fi

fail=0
run() { echo ""; echo "=============== $* ==============="; "$@" 2>&1 || { echo "!!! FAILED: $*"; fail=$((fail+1)); }; }

echo "### [1/3] B0 -> B4 campaign (small scale) -> demo/_live/ ..."
run $PY312 -W ignore scripts/wp18_campaign_runner.py --out "$LIVE_OUT"

echo ""
echo "### [2/3] end-to-end integration on $TILES real tiles -> demo/_live/ ..."
# wp11 reads its catalogue (wp6_*.{csv,json}) from the SAME dir it writes to, so stage copies
# of those inputs into the scratch dir; wp11 then both reads and writes entirely in demo/_live
# and never touches the authoritative code/results/.
cp results/wp6_tiles.csv results/wp6_ships.csv results/wp6_size_model.json "$LIVE_OUT"/ 2>/dev/null
run $PY312 -W ignore scripts/wp11_integration_demo.py --tiles "$TILES" --out "$LIVE_OUT"

echo ""
echo "### [3/3] consolidated demo summary (reports the AUTHORITATIVE full-run canon in code/results/) ..."
echo "    note: the live run above executed the real pipeline on a $TILES-tile sample and wrote to"
echo "          demo/_live/ (scratch). The summary below is the committed full-run result, not the sample."
(cd .. && code/.venv312/Scripts/python.exe demo/summary.py) || fail=$((fail+1))

echo ""
echo "=============== DEMO DONE: $fail failure(s) ==============="
exit $fail
