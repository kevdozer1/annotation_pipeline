# BridgeEngine

Prototype implementation of pi0.7 annotation pipeline on BridgeDataV2

BridgeEngine is a Mode A proof-of-concept data engine for pi0.7-style robot annotation on BridgeData V2. It builds local Parquet snapshots, runs rich-prompt labelers, exposes DuckDB queries and a Streamlit viewer, exports deterministic training cuts, and keeps a 4-family label-value benchmark scaffold.

The current pivot tests whether VLM-derived subtask segmentation is good enough to produce the pi0.7-style effect at POC scale. Perception labelers from the original version are preserved as comparison modules, but they are not part of the main benchmark.

## Quickstart

PowerShell on this workstation:

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\.venv\Scripts\python.exe -m pip install -e .
# Preferred hosted backend: set OPENAI_API_KEY in your shell or .secrets/openai_api_key.txt.
# Comparison backend:
.\scripts\set_moondream_key.ps1

$SnapshotJson = .\.venv\Scripts\python.exe -m bridgeengine.ingest --source bridge_v2 --episodes 13
$SnapshotId = $SnapshotJson | .\.venv\Scripts\python.exe -c "import json, sys; print(json.load(sys.stdin)['snapshot_id'])"
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot $SnapshotId --vlm-backend openai
.\.venv\Scripts\python.exe -m bridgeengine.inspect_labels --snapshot $SnapshotId
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot $SnapshotId
.\.venv\Scripts\python.exe -m streamlit run bridgeengine/viewer/app.py
```

The ingest command prefers `D:\bridgedata_v2_subset`. If that path is missing, it creates a deterministic synthetic Bridge-like source under `bridgeengine_data/raw/bridge_v2`.

Do not run the benchmark until the live labels pass the quality gate. The fallback adapter is only for CI and plumbing tests:

```powershell
.\scripts\poc_quickstart.ps1 -AllowFallback
```

Fallback labels and known-bad live labels are rejected by the benchmark runner unless `--allow-scaffolding-labels` is passed explicitly for plumbing-only runs.

Backend selection:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot $SnapshotId --vlm-backend openai --vlm-model gpt-5.5
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot $SnapshotId --vlm-backend moondream
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot $SnapshotId --vlm-backend mock
```

## After Label Inspection

Once the segment boundaries, subtask text, metadata distribution, and subgoal paths look reasonable:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.query --snapshot $SnapshotId
.\.venv\Scripts\python.exe -m bridgeengine.export --snapshot $SnapshotId --output-path training_cuts --cut-name cut_mode_a_all_labels
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.run_grid --snapshot $SnapshotId --output-dir bench_results
```

Expected artifacts:

- `bridgeengine_data/snapshots/<snapshot_id>/manifest.json`
- `bridgeengine_data/snapshots/<snapshot_id>/episodes.parquet`
- `bridgeengine_data/snapshots/<snapshot_id>/steps.parquet`
- `bridgeengine_data/snapshots/<snapshot_id>/labels.parquet`
- `bridgeengine_data/snapshots/<snapshot_id>/raw_vlm_outputs/<episode_id>/*.json`
- `bridgeengine_data/snapshots/<snapshot_id>/subgoals/<episode_id>/*.jpg`
- `training_cuts/cut_mode_a_all_labels/manifest.json`
- `bench_results/bench_results.csv`
- `bench_results/bench_bar.png`
- `bench_results/bench_summary.md`

## Architecture

```text
BridgeData V2
        |
        v
1. Ingest + storage
   local raw paths + versioned Parquet snapshots + manifest JSON
        |
        v
2. Rich-prompt labelers
   subtask_segmenter + episode_metadata + subgoal_images
        |
        +---------------------+
        |                     |
        v                     v
3. DuckDB query helpers   4. export_cut
   notebook/viewer           deterministic manifest + episode list + label paths
        |                     |
        +----------+----------+
                   v
5. Label-Value Benchmark
   baseline, rich_text, rich_text_metadata, rich_text_metadata_subgoal
```

## Annotation Families

- `baseline`: BridgeData task instruction only.
- `rich_text`: task instruction plus active subtask text from the two-stage VLM segmenter.
- `rich_text_metadata`: rich text plus speed, quality, mistake, and control mode.
- `rich_text_metadata_subgoal`: metadata prompt plus actual end-of-segment frame as the subgoal image.

The prompt format is intentionally direct:

```text
Task: {task}. Subtask: {subtask_text}. Speed: {speed}. Quality: {quality}/5. Mistake: {mistake}. Control: {control_mode}.
```

## Comparison Perception Labelers

The original SAM/depth/track work is kept under `bridgeengine.labelers.perceptive`:

- `perceptive_masks`: wraps LEWM `object_mask.npy` artifacts.
- `perceptive_depth`: wraps LEWM `depth.npy` artifacts.
- `perceptive_tracks`: wraps LEWM `tracks.npy` and `visibility.npy` artifacts.

Run them manually when needed:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot $SnapshotId --perceptive
```

## Viewer

```powershell
.\scripts\viewer.ps1
```

The viewer runs at <http://localhost:8501>. It shows selected episodes, frames, pi0.7-style prompt previews, subtask segments, metadata payloads, subgoal images, raw provenance paths, optional perception artifacts, query outputs, and benchmark artifacts.

## Scope Boundary

This repository intentionally does not include Iceberg, Delta, lakeFS, Qdrant, FoundationPose, WebDataset tar shards, Ray Data, IDM accuracy, tactile/audio labels, generated subgoal images, verbal coaching, a paper, or a demo video. The Streamlit viewer is included as a local inspection aid because Kevin requested it.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot $SnapshotId
.\.venv\Scripts\python.exe -m bridgeengine.system_check
```
