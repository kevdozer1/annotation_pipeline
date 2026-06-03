# BridgeEngine Architecture

BridgeEngine is a local conditioning-and-curation layer for robot datasets. The current codebase is organized around deterministic snapshots, labeler adapters, query/export helpers, a quality-gated benchmark scaffold, and a Streamlit inspection surface.

## Data Flow

```text
D:\bridgedata_v2_subset or synthetic fallback
        |
        v
bridgeengine.ingest.bridge_v2
        |
        v
bridgeengine_data/snapshots/<snapshot_id>/
  episodes.parquet
  steps.parquet
  sensors.parquet
  labels.parquet
  manifest.json
        |
        v
bridgeengine.orchestrate.runner
  subtask_segmenter
  episode_metadata
  subgoal_images
  optional perceptive comparison labelers
        |
        +---------------------------+
        |                           |
        v                           v
bridgeengine.query              bridgeengine.export.cut
DuckDB snapshot views           deterministic training cut
        |                           |
        +------------+--------------+
                     v
bridgeengine.benchmark.run_grid
quality gate -> 4-family benchmark scaffold
```

## Snapshot Schema

Snapshot IDs are derived in `bridgeengine.ingest.snapshot` from source episode records, schema version, and labeler versions. Snapshot manifests use deterministic timestamps for reproducible test comparisons.

`episodes.parquet` columns:

- `episode_id`
- `source_path_video`
- `source_path_actions`
- `source_path_meta`
- `source_path_frames`
- `num_steps`
- `language_instruction`
- `snapshot_id`

`steps.parquet` columns:

- `episode_id`
- `step_idx`
- `timestamp`
- `action`
- `state`
- `snapshot_id`

`sensors.parquet` columns:

- `episode_id`
- `sensor_name`
- `calibration_json`
- `snapshot_id`

## Label Schema

`labels.parquet` is a row-level index over JSON payloads and generated artifacts. The schema is defined in `bridgeengine.ingest.schema`.

Columns:

- `episode_id`
- `step_idx`
- `segment_idx`
- `labeler_name`
- `labeler_version`
- `label_payload_path`
- `metadata_payload_json`
- `subgoal_image_path`
- `confidence`
- `provenance_json`
- `snapshot_id`

Label payloads live outside Parquet as JSON files under `bridgeengine_data/labels/...` or under snapshot artifact directories. Raw VLM receipts live under:

```text
bridgeengine_data/snapshots/<snapshot_id>/raw_vlm_outputs/<episode_id>/
```

## Labelers

Main pi0.7-style labelers:

- `bridgeengine.labelers.subtask_segmenter`: produces segment-scoped `start_step`, `end_step`, and `subtask_text`.
- `bridgeengine.labelers.episode_metadata`: produces episode-scoped `speed`, `quality`, `mistake`, and `control_mode`.
- `bridgeengine.labelers.subgoal_images`: extracts the actual end-of-segment frame as a POC subgoal image.

Semantic labelers call `bridgeengine.labelers.backends.VisionLanguageBackend`, not provider-specific APIs directly. Backends currently include:

- `openai`: OpenAI Responses API, defaulting to a GPT-5.5-class model name via `BRIDGEENGINE_VLM_MODEL` or `gpt-5.5`.
- `moondream`: Moondream API compatibility path.
- `mock`: deterministic test backend for CI and local smoke tests.

The two semantic labelers use an observe-then-label flow. Stage one asks for physical observations only and writes a raw observation receipt. Stage two consumes those observations and emits the structured label, writing a separate raw label receipt.

Comparison perception labelers:

- `bridgeengine.labelers.perceptive.MaskLabeler`
- `bridgeengine.labelers.perceptive.DepthLabeler`
- `bridgeengine.labelers.perceptive.TrackLabeler`

The perception labelers are preserved for comparison and future ablations but are not part of the main pi0.7-style benchmark families.

## Query Layer

`bridgeengine.query.duckdb_helpers` creates in-memory DuckDB views over a snapshot's Parquet files. The shipped demo queries inspect label coverage, metadata quality, subgoal paths, labeler success counts, and a pi0.7-style prompt trace.

## Export Layer

`bridgeengine.export.cut.export_cut` writes deterministic training cuts:

- `manifest.json`
- `episode_list.txt`
- `label_paths.json`
- `episode_sources.json`

`BridgeCutDataset` provides a small Python dataset wrapper for those cuts.

## Benchmark Families

The benchmark family enum lives in `bridgeengine.benchmark.train_lewm`.

- `baseline`: BridgeData task instruction only.
- `rich_text`: task instruction plus subtask text from `subtask_segmenter`.
- `rich_text_metadata`: rich text plus `episode_metadata`.
- `rich_text_metadata_subgoal`: rich text plus metadata plus `subgoal_images`.

The current benchmark implementation is still a deterministic CPU proxy for LEWM latent MSE. It is intentionally blocked by quality gates unless labels pass inspection.

## Quality Gate

`bridgeengine.benchmark.run_grid` calls `bridgeengine.quality_gate.evaluate_snapshot_quality` before exporting cuts or running the benchmark. The gate rejects fallback/scaffolding labels and known-bad live labels so benchmark artifacts are not generated from misleading annotation data.

Current explicit checks:

- no fallback/scaffolding provenance,
- no repeated subtask text within an episode,
- metadata score/reason consistency,
- object grounding against stage-one observations,
- dataset-level score dispersion.

`bridgeengine.quality_report` prints the gate report independently of the benchmark.

## Viewer

`bridgeengine.viewer.app` is the local inspection surface. It shows snapshot tables, episode frames, pi0.7 prompt previews, subtask segments, metadata payloads, subgoal images, optional perception artifacts, query outputs, and benchmark artifacts.
