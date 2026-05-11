#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_ID="${1:?Usage: scripts/bench_run_all.sh <snapshot_id>}"
PY="${PYTHON:-python}"
"$PY" -m bridgeengine.benchmark.run_grid --snapshot "$SNAPSHOT_ID" --output-dir bench_results

