# BridgeEngine Status - 2026-06-02

## Bottom Line

BridgeEngine reached a working demo/infrastructure state, but not a benchmark-claim state.

The project can ingest the 13-episode BridgeData V2 subset, create deterministic Parquet snapshots, run live Moondream-backed pi0.7-shaped labelers, save raw VLM provenance, extract subgoal images, query the result with DuckDB, export a deterministic training cut, and show the labels in Streamlit. The remaining blocker is label quality: current live Moondream semantic labels are inspectable but not clean enough to justify running the 12-run label-value benchmark or claiming rich-text annotations improve LEWM.

## What Works

- Contained Python venv exists and the package installs editable.
- Ingest works against Kevin's local BridgeData subset at `D:\bridgedata_v2_subset`.
- Current snapshot: `snap_2026_05_11_a8256b172c`.
- Snapshot contains 13 episodes, 334 steps, and 65 label rows.
- Live Moondream API integration works.
- The API-key BOM bug was fixed.
- Raw VLM responses are saved under `bridgeengine_data/snapshots/<snapshot_id>/raw_vlm_outputs/<episode_id>/`.
- Subtask labels are generated with action/gripper-transition temporal boundaries plus Moondream semantic text.
- Episode metadata labels are generated with Moondream plus deterministic speed/control-mode fields.
- Subgoal images are extracted as actual end-of-segment frames.
- Perception labelers were preserved under `bridgeengine.labelers.perceptive` as comparison modules.
- Streamlit viewer exists for inspecting episodes, prompts, labels, subgoals, raw provenance paths, queries, and benchmark artifacts.
- Query and export paths work.
- Benchmark runner has guardrails and refuses bad labels instead of producing misleading results.

## What Did Not Get Finished

- We did not complete a valid benchmark result.
- We did not run or trust the 4-family x 3-seed label-value grid.
- We did not produce a credible `bench_results.csv` for the pivoted pi0.7-style families.
- We did not reach publication/demo-quality annotation semantics.
- We did not validate that rich text, metadata, or subgoal images improve LEWM latent MSE.
- We did not wire a real heavyweight LEWM GPU training loop into the benchmark; the scaffold still uses a deterministic CPU proxy.
- We did not produce human-validated subtask segmentations.

## Current Label Quality

The current live labels are real, non-fallback, and useful for inspecting the system, but they are not benchmark-grade.

Current label audit:

```text
subtask_segmenter rows: 13
episode_metadata rows: 13
subgoal_images rows: 39
segment_count_distribution: {3: 13}
contiguous_all: True
episodes_with_duplicate_text: 6 / 13
quality_counts: {1: 4, 4: 9}
mistake_counts: {False: 9, True: 4}
fallback/scaffolding labels: none detected
```

Main quality problems:

- Six episodes have repeated subtask text across multiple segments.
- Some semantic labels are object-confused.
- Metadata judge output is coarse, mostly quality 1 or 4.
- Some metadata reasons contradict the numeric score.
- Moondream sometimes emits placeholder-ish reasons.

The correct interpretation is: the infrastructure is working, and the label inspection caught meaningful quality failures before the benchmark burned compute.

## Testing And Evaluation Done

### Unit/Smoke Tests

Verified on 2026-06-02:

```text
pytest: 4 passed
```

Covered tests:

- ingest snapshot creation
- labeler population
- cut reproducibility
- benchmark smoke shape with scaffolding labels explicitly allowed for test mode

### Query Evaluation

Verified on 2026-06-02:

```text
episodes: 13
steps: 334
sensors: 13
labels: 65
```

DuckDB query timings:

```text
subtask_coverage: 13 rows in 22.3ms
metadata_quality: 13 rows in 20.2ms
subgoal_paths: 39 rows in 18.9ms
labeler_success_counts: 3 rows in 18.4ms
pi07_prompt_trace: 1 rows in 21.9ms
```

### Label Inspection

Ran:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.inspect_labels --snapshot snap_2026_05_11_a8256b172c
```

This confirmed no fallback labels, displayed representative episode payloads, and surfaced repeated subtask text plus metadata contradictions.

### Benchmark Guard Evaluation

Ran:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.run_grid --snapshot snap_2026_05_11_a8256b172c --output-dir bench_results_guard_test
```

Expected result:

```text
RuntimeError: Refusing to run the benchmark because label quality gates failed.
Examples: episode_003087: repeated subtask_text, episode_004196: repeated subtask_text, episode_017546: repeated subtask_text, episode_024911: repeated subtask_text, episode_035339: repeated subtask_text
```

This is a successful safety check. The system is preventing misleading benchmark output.

### Training Cut Export

Previously completed successfully:

```text
training_cuts/cut_mode_a_all_labels
```

## Current Demo Command

Start the viewer:

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\.venv\Scripts\python.exe -m streamlit run bridgeengine/viewer/app.py --server.port 8502
```

Then inspect labels:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.inspect_labels --snapshot snap_2026_05_11_a8256b172c
```

## Recommended Next Step

Do not benchmark yet. The next useful work is either:

1. improve the VLM semantic labeler with a two-stage observation-then-label prompt,
2. use a stronger hosted VLM for the 13-episode semantic pass, or
3. add a small human validation layer and benchmark VLM-raw versus human-validated labels.

Once repeated subtask text and metadata contradictions are fixed, rerun the benchmark.
