#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ALLOW_FALLBACK=0
RUN_BENCHMARK=0
VLM_BACKEND="openai"
VLM_MODEL=""
for arg in "$@"; do
  case "$arg" in
    --allow-fallback) ALLOW_FALLBACK=1 ;;
    --run-benchmark) RUN_BENCHMARK=1 ;;
    --vlm-backend=*) VLM_BACKEND="${arg#*=}" ;;
    --vlm-model=*) VLM_MODEL="${arg#*=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

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

if [ "$ALLOW_FALLBACK" -eq 0 ]; then
  if [ "$VLM_BACKEND" = "openai" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f ".secrets/openai_api_key.txt" ]; then
    echo "OPENAI_API_KEY is not set. Set it, save .secrets/openai_api_key.txt, choose --vlm-backend=moondream, or pass --allow-fallback." >&2
    exit 1
  fi
  if [ "$VLM_BACKEND" = "moondream" ] && [ -z "${MOONDREAM_API_KEY:-}" ] && [ ! -f ".secrets/moondream_api_key.txt" ]; then
    echo "Moondream key is not set. Run scripts/set_moondream_key.ps1, set MOONDREAM_API_KEY, or pass --allow-fallback." >&2
    exit 1
  fi
fi

LABEL_ARGS=(-m bridgeengine.label --snapshot "$SNAPSHOT_ID" --vlm-backend "$VLM_BACKEND")
if [ -n "$VLM_MODEL" ]; then
  LABEL_ARGS+=(--vlm-model "$VLM_MODEL")
fi
if [ "$ALLOW_FALLBACK" -eq 1 ]; then
  LABEL_ARGS+=(--allow-fallback)
fi
"$PY" "${LABEL_ARGS[@]}"
"$PY" -m bridgeengine.inspect_labels --snapshot "$SNAPSHOT_ID"
"$PY" -m bridgeengine.query --snapshot "$SNAPSHOT_ID"
"$PY" -m bridgeengine.export --snapshot "$SNAPSHOT_ID" --output-path training_cuts --cut-name cut_mode_a_all_labels

if [ "$RUN_BENCHMARK" -eq 1 ]; then
  BENCH_ARGS=(-m bridgeengine.benchmark.run_grid --snapshot "$SNAPSHOT_ID" --output-dir bench_results)
  if [ "$ALLOW_FALLBACK" -eq 1 ]; then
    BENCH_ARGS+=(--allow-scaffolding-labels)
  fi
  "$PY" "${BENCH_ARGS[@]}"
else
  echo "Benchmark skipped. Inspect labels first, then rerun with --run-benchmark after green-lighting live Moondream outputs."
fi

echo "Done. Artifacts:"
echo "  bridgeengine_data/snapshots/$SNAPSHOT_ID"
echo "  training_cuts/cut_mode_a_all_labels"
if [ "$RUN_BENCHMARK" -eq 1 ]; then
  echo "  bench_results/bench_results.csv"
  echo "  bench_results/bench_bar.png"
fi
