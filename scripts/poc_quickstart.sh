#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  if command -v py >/dev/null 2>&1; then
    py -3.10 -m venv .venv
  else
    python -m venv .venv
  fi
fi

if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
else
  PY=".venv/bin/python"
fi

"$PY" -m pip install -e . >/dev/null

SNAPSHOT_JSON="$("$PY" -m bridgeengine.ingest --source bridge_v2 --episodes 13)"
SNAPSHOT_ID="$(SNAPSHOT_JSON="$SNAPSHOT_JSON" "$PY" -c "import json, os; print(json.loads(os.environ['SNAPSHOT_JSON'])['snapshot_id'])")"
echo "Snapshot: $SNAPSHOT_ID"

"$PY" -m bridgeengine.label --snapshot "$SNAPSHOT_ID"
"$PY" -m bridgeengine.query --snapshot "$SNAPSHOT_ID"
"$PY" -m bridgeengine.export --snapshot "$SNAPSHOT_ID" --output-path training_cuts --cut-name cut_mode_a_all_labels
"$PY" -m bridgeengine.benchmark.run_grid --snapshot "$SNAPSHOT_ID" --output-dir bench_results

echo "Done. Artifacts:"
echo "  bridgeengine_data/snapshots/$SNAPSHOT_ID"
echo "  training_cuts/cut_mode_a_all_labels"
echo "  bench_results/bench_results.csv"
echo "  bench_results/bench_bar.png"
