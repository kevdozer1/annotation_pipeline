# BridgeEngine

Prototype implementation of pi0.7 annotation pipeline on BridgeDataV2

Mode A proof-of-concept data engine and label-value benchmark for robot foundation-model training on BridgeData V2.

BridgeEngine is the smallest useful version of the project: local versioned Parquet snapshots, LEWM-derived label artifacts, DuckDB queries, deterministic training-cut export, and a 4-family x 3-seed latent-MSE benchmark table. It is designed to make the LEWM supervision question reproducible as a data-engine workflow.

Related context:

- Kevin's LEWM writeup: <https://kevdozer1.com/blog/2026/lewm-finetune/>
- LeWorldModel project page: <https://le-wm.github.io/>

## Quickstart

PowerShell on this workstation:

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\scripts\poc_quickstart.ps1
.\scripts\viewer.ps1
```

Bash from a clean clone:

```bash
cd annotation_pipeline
bash scripts/poc_quickstart.sh
```

The quickstart uses `D:\bridgedata_v2_subset` when present. If that local LEWM-curated BridgeData subset is unavailable, it creates a deterministic synthetic Bridge-like fallback under `bridgeengine_data/raw/bridge_v2` so the repository still runs end to end.

Expected outputs:

- `bridgeengine_data/snapshots/<snapshot_id>/manifest.json`
- `bridgeengine_data/snapshots/<snapshot_id>/episodes.parquet`
- `bridgeengine_data/snapshots/<snapshot_id>/steps.parquet`
- `bridgeengine_data/snapshots/<snapshot_id>/sensors.parquet`
- `bridgeengine_data/snapshots/<snapshot_id>/labels.parquet`
- `training_cuts/cut_mode_a_all_labels/manifest.json`
- `bench_results/bench_results.csv`
- `bench_results/bench_bar.png`
- `bench_results/bench_summary.md`

The viewer runs at <http://localhost:8501>. It shows the selected snapshot, episode frames, mask overlays, depth maps, point tracks, caption/provenance payloads, the five DuckDB queries, benchmark artifacts, and a local readiness check.

## Manual Commands

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.ingest --source bridge_v2 --episodes 13
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot <snapshot_id>
.\.venv\Scripts\python.exe -m bridgeengine.query --snapshot <snapshot_id>
.\.venv\Scripts\python.exe -m bridgeengine.export --snapshot <snapshot_id> --output-path training_cuts --cut-name cut_mode_a_all_labels
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.run_grid --snapshot <snapshot_id> --output-dir bench_results
.\.venv\Scripts\python.exe -m bridgeengine.system_check
.\.venv\Scripts\python.exe -m streamlit run bridgeengine/viewer/app.py
.\.venv\Scripts\python.exe -m pytest
```

## Architecture

```text
BridgeData V2 / synthetic fallback
        |
        v
1. Ingest + storage
   local raw paths + versioned Parquet snapshots + manifest JSON
        |
        v
2. Auto-labeling adapters
   captions, masks, depth, tracks with provenance
        |
        +---------------------+
        |                     |
        v                     v
3. DuckDB query helpers   4. export_cut
   notebook interface        deterministic manifest + episode list + label paths
        |                     |
        +----------+----------+
                   v
5. Label-Value Benchmark
   4 families x 13 episodes x 3 seeds, latent MSE table + bar chart
```

## What Is Implemented

- Local filesystem storage with snapshot-versioned Parquet metadata.
- Deterministic snapshot IDs based on episode source records, schema version, and labeler versions.
- Four Mode A labeler families:
  - `captions`: deterministic Moondream-compatible adapter.
  - `masks`: wraps LEWM `object_mask.npy` artifacts when present, with 8px and 20px dilation variants.
  - `depth`: wraps LEWM `depth.npy` artifacts when present.
  - `tracks`: wraps LEWM `tracks.npy` and `visibility.npy` artifacts when present.
- `bridgeengine.query` DuckDB helpers and `notebooks/explore.ipynb` with five queries.
- `bridgeengine.export.cut.export_cut` plus `BridgeCutDataset`.
- Mode A 12-row benchmark grid and plot generation.
- Streamlit viewer for inspecting episode frames, masks, depth, tracks, caption/provenance payloads, query outputs, and benchmark results.

## Known Scope Boundary

This repository intentionally does not include Iceberg, Delta, lakeFS, Qdrant, FoundationPose, WebDataset tar shards, Ray Data, IDM accuracy, tactile/audio labels, or a paper/demo video. Those are Mode C items in the plan. A Streamlit viewer is included as a local demo aid.

## Live Model Readiness

Current quickstart uses existing LEWM artifacts from `D:\bridgedata_v2_subset`. To regenerate labels from raw frames, the model environment still needs:

- CUDA-capable PyTorch installed in the venv.
- SAM2 Python package or repo import path plus `D:\extraction_models\sam2_checkpoints\sam2.1_hiera_tiny.pt`.
- `D:\extraction_models\Video-Depth-Anything` on `PYTHONPATH` plus `video_depth_anything_vits.pth`.
- `D:\extraction_models\co-tracker` installed editable or on `PYTHONPATH`.

Run:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.system_check
```

That command reports which pieces are present and which imports are still missing.
