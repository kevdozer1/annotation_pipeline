# Build Report

## What Was Built

BridgeEngine Mode A now exists as a contained Python project in `C:\Users\Kevin\projects\annotation_pipeline` with a local `.venv`. The POC ingests the 13-episode LEWM BridgeData subset from `D:\bridgedata_v2_subset`, writes a deterministic snapshot at `bridgeengine_data/snapshots/snap_2026_05_11_a8cbb6f8dd`, runs four labeler families, validates five DuckDB queries, exports a deterministic training cut, generates a 12-row benchmark result set with `bench_results/bench_results.csv`, `bench_results/bench_bar.png`, and `bench_results/bench_summary.md`, and includes a Streamlit viewer for inspecting episodes and annotations.

## Deviations

See `DEVIATIONS.md`. The important deviations are: Python 3.10 instead of 3.11 because 3.11 is not installed; captions are a deterministic Moondream-compatible adapter; the LEWM mask/depth/track labelers wrap existing artifacts instead of re-running heavyweight models; the benchmark uses a deterministic CPU proxy rather than launching 12 GPU LEWM training runs; Streamlit was added as a user-requested demo aid.

## Known Issues

- The public GitHub remote is not created or pushed from this local environment.
- The benchmark numbers are proxy values; the adapter boundary is ready for the real LEWM training sweep, but the GPU runs were not launched.
- Live Moondream, SAM 3, Video-Depth-Anything, and CoTracker3 model inference is deferred; existing LEWM artifacts are used where available.

## Validation Commands

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\scripts\poc_quickstart.ps1
.\scripts\viewer.ps1
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m bridgeengine.query --snapshot snap_2026_05_11_a8cbb6f8dd
.\.venv\Scripts\python.exe -m bridgeengine.system_check
```

Expected quickstart result: the same deterministic snapshot ID, 13 episodes, 334 steps, 52 labels, all five demo queries returning non-empty results under 500ms, and a 12-row benchmark CSV.

## Demo Starting Point

Start with `README.md` for the architecture diagram, then open the Streamlit viewer at `http://localhost:8501` and select `snap_2026_05_11_a8cbb6f8dd`. The first verbal frame should be: "BridgeEngine operationalizes the LEWM annotation-value question as a reproducible data-engine workflow."
