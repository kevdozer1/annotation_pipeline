# BridgeEngine

Prototype implementation of pi0.7 annotation pipeline on BridgeDataV2

BridgeEngine is a Mode A proof-of-concept data engine for pi0.7-style robot annotation on BridgeData V2. It builds local Parquet snapshots, runs rich-prompt labelers, exposes DuckDB queries and viewers, exports deterministic training cuts, and runs real LeWM frozen-adapter label-value benchmarks.

The current pivot tests whether VLM-derived subtask segmentation plus calibrated episode metadata is enough to produce a pi0.7-style conditioning effect at POC scale. Perception labelers from the original version are preserved as comparison modules, but they are not part of the main benchmark. The current score-calibrated 100-episode LeWM smoke ablation is positive for the rich-text + metadata + subgoal family, but it is still a two-seed smoke result, not a robust robotics conclusion.

## Quickstart

PowerShell on this workstation:

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\.venv\Scripts\python.exe -m pip install -e .
# Preferred low-cost hosted backend:
.\scripts\set_gemini_key.ps1

$SnapshotJson = .\.venv\Scripts\python.exe -m bridgeengine.ingest --source bridge_v2 --episodes 13
$SnapshotId = $SnapshotJson | .\.venv\Scripts\python.exe -c "import json, sys; print(json.load(sys.stdin)['snapshot_id'])"
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot $SnapshotId --vlm-backend gemini --vlm-model gemini-2.5-flash
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
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot $SnapshotId --vlm-backend gemini --vlm-model gemini-2.5-flash
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot $SnapshotId --vlm-backend moondream
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot $SnapshotId --vlm-backend mock
```

Real benchmark dependencies on Kevin's workstation:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e "C:\Users\Kevin\projects\upstream\stable-worldmodel[train]"
.\.venv\Scripts\python.exe -m pip install --no-cache-dir "transformers==5.4.0" "huggingface-hub==1.8.0" "stable-pretraining==0.1.6"
$env:LEWM_PRETRAINED_PATH = "D:\hf_cache\models--quentinll--lewm-cube\snapshots\7d05e023b3c1114cc8e803ec23fb0177d688598b\weights.pt"
```

The checkpoint expects the `transformers==5.4.0` ViT key layout. Newer `transformers` builds changed module names and fail to load this checkpoint.

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

Benchmark note: this is a smoke-scale real LeWM frozen-adapter ablation. It is useful for seeing whether the pipeline can produce learned held-out latent-MSE numbers, but it is not a robust robotics conclusion.

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

## Current Benchmark Finding

Current best result on `snap_2026_05_11_1dde3edf5d_human_calibrated`:

| N | baseline | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|---:|
| 25 | 0.042079 | 0.041079 | 0.039682 | 0.038022 |
| 50 | 0.031014 | 0.030213 | 0.028289 | 0.027482 |
| 100 | 0.022522 | 0.023123 | 0.022831 | 0.020807 |

Delta versus baseline:

| N | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|
| 25 | -2.38% | -5.70% | -9.64% |
| 50 | -2.58% | -8.79% | -11.39% |
| 100 | +2.67% | +1.37% | -7.61% |

Read this as: the real benchmark path works, and after Kevin reviewed all 100 clips and changed 58 scores, the metadata+subgoal family beats baseline across the tested sizes. This is still a two-seed, 100-episode LeWM frozen-adapter smoke result. Boundary and subgoal labels have not been human-gold validated.

## Final Closeout Path

The next human task is boundary/subgoal reliability review, not rescoring. The prepared 50-episode queue is:

```text
gold_sets/boundary_subgoal_review_50.json
```

Start the focused review GUI:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.review_gui `
  --snapshot snap_2026_05_11_1dde3edf5d `
  --gold-file C:\Users\Kevin\projects\annotation_pipeline\bridgeengine_data\snapshots\snap_2026_05_11_1dde3edf5d\gold\calibration_gold.json `
  --episode-file gold_sets\boundary_subgoal_review_50.json `
  --review-goal boundary_subgoal `
  --port 8787
```

For each selected episode, leave the score alone. If the auto labels look right, check the top-level subtask and subgoal accept boxes. If a row is wrong, uncheck that row and edit the subtask start/end step or the subgoal frame index. A subgoal is a representative frame, not a bounding box.

The controlled head-to-head experiment is preregistered in `HEAD_TO_HEAD_PREREGISTRATION.md`. The public fork plan is in `PUBLIC_RELEASE_PLAN.md`.

## Value-Aware Curation

BridgeEngine can score each episode by estimated curation value and write the score directly into `episodes.parquet`.

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.value report --snapshot $SnapshotId
```

The value interface has two methods:

- `prediction-error`: trains a small LeWM frozen-adapter pass and scores episodes by per-episode latent prediction error. High error means anomalous/high-value.
- `embedding-distance`: deterministic fallback using action/state/frame summary embeddings, centroid distance, and kNN sparsity.

The report prints the value-score distribution, top outliers, and a tiered Parquet compression comparison. At tiny local scale, tiered compression can be larger than uniform zstd because split-file overhead dominates; this is expected to be meaningful only on larger snapshots.

## Cost-Gated Scaling

Ingest accepts arbitrary episode counts from the mounted source:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.ingest --source D:\bridgedata_v2_subset --episodes 50 --episode-offset 0
.\.venv\Scripts\python.exe -m bridgeengine.ingest --source D:\bridgedata_v2_subset --episodes all
```

The current workstation mount exposes 100 local episodes under `D:\bridgedata_v2_subset`; the full approximately 60k BridgeData V2 corpus is not mounted here.

Before any larger hosted-VLM run, use the cost probe:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.cost_probe --snapshot $SnapshotId --projection 200 --projection 1000 --projection 60000
```

The 50-episode `gpt-5.5` probe is documented in `COST_PROBE_50.md`. It measured about `$0.063316` per episode and 39.4 seconds per episode serially, with projected 60k labeling cost around `$3,798.94` and 656.81 serial hours. The later Gemini all-100 run cost about `$1.188603` total, or `$0.011886` per episode, and the calibrated all-100 snapshot passes the quality gate.

Scale-curve ablations are planned separately from labeling:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.scale_curve --snapshot $SnapshotId --sizes 50 200 800 --heldout-count 10 --quality-stratified --output-dir scale_results\plan
```

This writes deterministic split files and a plan. It does not launch LeWM training unless `--run` is passed. Do not pass `--run` until Kevin has approved the target N and the labels pass the quality gate.

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
.\.venv\Scripts\python.exe -m bridgeengine.goldset init --snapshot $SnapshotId --output gold_sets\$SnapshotId.json
# After filling gold_sets\$SnapshotId.json:
.\.venv\Scripts\python.exe -m bridgeengine.goldset report --snapshot $SnapshotId --gold-file gold_sets\$SnapshotId.json
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.run_grid --snapshot $SnapshotId --output-dir bench_results --gold-file gold_sets\$SnapshotId.json
```
