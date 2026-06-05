# BridgeEngine Handoff Almanac

Last updated: 2026-06-04

This document is the handoff map for a third-party coding model or researcher taking over BridgeEngine. It records what the project is trying to prove, what has actually been built, which results are trustworthy, which are smoke-scale only, and exactly how to resume work.

## Executive State

BridgeEngine is a local robot-dataset annotation and curation pipeline for BridgeData V2. Its core demo is not "a new robot model." Its core demo is a data layer that turns raw robot episodes into queryable pi0.7-style conditioning artifacts:

- temporally segmented subtask instructions
- episode metadata: task-success quality, curation quality, mistake, speed, control mode
- subgoal keyframes at segment boundaries
- raw VLM provenance for every semantic label
- a quality gate that refuses bad labels before benchmark entry
- DuckDB query, deterministic export, Streamlit viewer, figures, and a LeWM smoke benchmark

Current honest scientific result: the first real 13-episode LeWM frozen-adapter ablation did not show a metadata win. Baseline had the best mean latent MSE. This is useful as an honest smoke result, not as a robust conclusion.

Current main blocker: Gemini labeling cannot run in this process because neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY` is visible, and `.secrets/gemini_api_key.txt` does not exist yet. The backend and resume commands are ready.

## Current Workspace

Repo:

```powershell
C:\Users\Kevin\projects\annotation_pipeline
```

Local BridgeData source:

```powershell
D:\bridgedata_v2_subset
```

The mounted local source currently exposes 100 episodes, not the full BridgeData V2 corpus.

Important ignored local data:

- `bridgeengine_data/`
- `training_cuts/`
- `.secrets/`
- raw images/videos/frames
- raw VLM outputs

Do not commit dataset frames, raw VLM payloads, API keys, `.env`, `.secrets`, or large snapshot binaries.

## Goal

The research goal is to test whether richer pi0.7-style conditioning labels help robot world-model or VLA-style training compared with raw task instruction only.

The engineering goal is to make that test reliable:

1. ingest robot episodes into deterministic Parquet snapshots
2. generate semantic annotations with provenance
3. reject bad labels before training
4. query and inspect labels visually
5. export deterministic training cuts
6. run a real held-out latent-MSE ablation
7. scale from 13 to 50 to 100 episodes before attempting larger corpora

## Built Capabilities

### Ingest

Module:

```text
bridgeengine/ingest/bridge_v2.py
bridgeengine/ingest/snapshot.py
bridgeengine/ingest/schema.py
```

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.ingest --source bridge_v2 --episodes 50
.\.venv\Scripts\python.exe -m bridgeengine.ingest --source bridge_v2 --episodes all
```

Status: works.

The ingest path reads local BridgeData-style episode folders and writes:

- `manifest.json`
- `episodes.parquet`
- `steps.parquet`
- `sensors.parquet`
- empty `labels.parquet`

Snapshot IDs are deterministic from source records, labeler versions, and schema version. Re-ingesting the same episode set gives the same snapshot ID.

### Snapshot Clone

Module:

```text
bridgeengine/snapshot_clone.py
```

Purpose: same source records produce the same snapshot ID, so this command creates a new label target without overwriting existing labels.

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.snapshot_clone `
  --source-snapshot snap_2026_05_11_de43f7bf0b `
  --target-snapshot snap_2026_05_11_de43f7bf0b_gemini50
```

Status: works.

### VLM Backends

Module:

```text
bridgeengine/labelers/backends.py
```

Backends:

- `openai`
- `gemini`
- `moondream`
- `mock`

Gemini was added this pass. It uses the Google Generative Language REST endpoint through `requests`, not a new SDK dependency. It stores raw response metadata, elapsed time, model, token usage, and cost estimates when usage metadata exists.

Default Gemini model:

```text
gemini-2.5-flash
```

Cheaper alternative:

```text
gemini-2.5-flash-lite
```

Gemini pricing constants in `bridgeengine/cost_probe.py` are based on Google AI Studio pricing documentation as checked on 2026-06-04:

```text
https://ai.google.dev/gemini-api/docs/pricing
```

The code uses:

- Gemini 2.5 Flash: `$0.30 / 1M` input, `$2.50 / 1M` output
- Gemini 2.5 Flash-Lite: `$0.10 / 1M` input, `$0.40 / 1M` output

These prices can change. Re-check before any large spend.

### Setting Keys

Scripts:

```text
scripts/set_openai_key.ps1
scripts/set_moondream_key.ps1
scripts/set_gemini_key.ps1
```

Gemini setup:

```powershell
.\scripts\set_gemini_key.ps1
```

This saves the key to:

```text
.secrets/gemini_api_key.txt
```

That path is ignored by git.

Current blocker observed in this session:

```text
RuntimeError: GEMINI_API_KEY or GOOGLE_API_KEY is not set.
```

### Semantic Labelers

Modules:

```text
bridgeengine/labelers/subtask_segmenter.py
bridgeengine/labelers/episode_metadata.py
bridgeengine/labelers/subgoal_images.py
```

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot <snapshot_id> --vlm-backend gemini --vlm-model gemini-2.5-flash
```

The labelers run:

1. subtask segmenter
2. episode metadata judge
3. subgoal image extractor

The semantic labelers use a two-stage observe-then-label flow:

- stage one asks the VLM to describe physical evidence
- stage two asks it to produce structured JSON labels using the observation record

The subgoal labeler is deterministic and extracts end-of-segment frames as JPEGs. This approximates pi0.7 subgoal images. It does not generate imagined subgoal images with a world model.

### Raw VLM Provenance

Raw VLM outputs are written under:

```text
bridgeengine_data/snapshots/<snapshot_id>/raw_vlm_outputs/<episode_id>/
```

Each raw output stores:

- backend
- model
- endpoint
- prompt/question
- keyframe indices
- status code
- elapsed seconds
- response text
- response JSON
- base64 image omitted note

These are ignored and should not be committed.

### Quality Gate

Module:

```text
bridgeengine/quality_gate.py
```

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot <snapshot_id>
```

Checks:

- fallback labels are rejected
- repeated subtask text is rejected
- placeholder metadata reason is rejected
- score/reason contradictions are rejected
- score collapse is rejected for sufficiently large snapshots
- object grounding checks actual object-like tokens against stage-one observations and task text

Important current update: the object-grounding gate no longer false-fails on common function/action/attribute words such as `empty`, `beside`, `across`, `settle`, `finish`, colors, and materials. This was tested. Truly ungrounded object nouns still fail.

Current 50-episode OpenAI snapshot gate:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {1: 2, 2: 1, 3: 5, 4: 17, 5: 25}
```

### Curation Scoring

Module:

```text
bridgeengine/scoring.py
```

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.scoring --snapshot <snapshot_id>
```

Current scoring version:

```text
boundary-usefulness-v2
```

This scorer separates:

- `task_success_quality`: did the requested task succeed?
- `curation_quality`: is the episode useful training data with clear visible manipulation structure?

The core rubric after Kevin's visual critique:

- short unfinished destination misses are clear rejects
- localized but unreliable target contact/end-state attempts are near rejects
- long multi-step tasks with clear visible subtask boundaries can be clear keeps
- unusual structure should be captured by anomaly/value score, not penalized as low quality

Specific corrected examples:

- `episode_001972`, pan-to-stove: `1/5 clear reject`
- `episode_005164`, cup-to-sink: `3/5 near reject`
- `episode_015003`, egg/pot/stove: `5/5 clear keep`
- `episode_003087`, can-in-pot: `5/5 clear keep`

### Figures

Module:

```text
bridgeengine/figures.py
```

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.figures --snapshot snap_2026_05_11_de43f7bf0b --threshold-animation --threshold-diagram
```

Current outputs:

```text
figures/threshold_annotation_animation.gif
figures/threshold_annotation_diagram.png
figures/quality_summary.png
figures/snapshot_overview.png
figures/benchmark_placeholder.png
```

The threshold GIF now shows:

- clear reject
- near reject
- clear keep
- clear keep

There is intentionally no "structured clear keep" label. Kevin hated that wording. Long multi-step episodes are just clear keeps.

### DuckDB Query

Module:

```text
bridgeengine/query/duckdb_helpers.py
```

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.query --snapshot <snapshot_id>
```

Status: works. Demo queries return rows on labeled snapshots and tolerate older/pre-pivot schemas.

### Export

Module:

```text
bridgeengine/export/cut.py
```

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.export --snapshot <snapshot_id> --output-path training_cuts --cut-name <name>
```

Status: works as a deterministic manifest/episode-list/label-path export.

It is not yet a real RLDS or LeRobot export.

### Viewer

Module:

```text
bridgeengine/viewer/app.py
```

Command:

```powershell
.\.venv\Scripts\streamlit.exe run bridgeengine\viewer\app.py
```

Status: works at smoke level. It can launch, show snapshots, figures, labels, and query output. It has not had full UX QA.

### Value-Aware Curation

Module:

```text
bridgeengine/value.py
```

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.value report --snapshot <snapshot_id> --method embedding-distance --top-n 10 --high-value-percentile 0.9
```

Methods:

- `prediction-error`: uses LeWM held-out latent prediction error when the LeWM environment is available
- `embedding-distance`: fallback based on episode feature distance/density
- `auto`: tries prediction-error first, then falls back to embedding-distance

At 50 episodes, tiered compression was net-negative:

```text
source parquet bytes: 215,247
uniform zstd bytes: 156,548
tiered bytes: 193,081
tiered vs uniform savings: -23.34%
```

Interpretation: at tiny per-file scale, layout overhead dominates. Do not claim savings at 50 episodes. The idea may still be useful at larger corpus scale.

### LeWM Benchmark

Modules:

```text
bridgeengine/benchmark/train_lewm.py
bridgeengine/benchmark/run_grid.py
bridgeengine/benchmark/scale_curve.py
bridgeengine/benchmark/plot.py
```

The previous fake CPU-proxy benchmark was replaced by a real LeWM frozen-adapter smoke eval path.

13-episode result:

```text
baseline mean:                    0.039541 +/- 0.004444
rich_text mean:                   0.042126 +/- 0.004554
rich_text_metadata mean:          0.040083 +/- 0.003849
rich_text_metadata_subgoal mean:  0.039926 +/- 0.004309
```

Interpretation:

- baseline had the best mean
- rich-text + metadata was about 1.4 percent worse than baseline
- differences are within seed noise
- this is a smoke-scale negative/inconclusive result, not proof that pi0.7-style labels do not help

Output files:

```text
bench_results/bench_results.csv
bench_results/bench_bar.png
bench_results/bench_summary.md
```

### Gold-Set Reliability

Modules:

```text
bridgeengine/goldset.py
```

Status: scaffold exists, no human gold labels filled yet.

The report supports:

- subtask-boundary temporal IoU
- quality exact agreement
- quality within-one agreement
- subgoal-selection agreement
- per-episode label wall-clock/cost

This is the next serious reliability step.

## Current Snapshots

### 13-Episode OpenAI Snapshot

```text
snap_2026_05_11_68c8cb784d
```

Used for the real 13-episode LeWM frozen-adapter benchmark.

### 50-Episode OpenAI Snapshot

```text
snap_2026_05_11_de43f7bf0b
```

This is the main current labeled 50-episode snapshot.

State after revised scoring:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {1: 2, 2: 1, 3: 5, 4: 17, 5: 25}
```

OpenAI cost probe before Gemini pivot:

```text
50 episodes cost estimate: $3.165780
cost per episode: $0.063316
serial wall-clock: 1970.42s
per episode: 39.41s
```

Note: the earlier `COST_PROBE_50.md` says the gate failed because of object-grounding false positives. That is now stale. The gate has since been fixed and the same snapshot passes after rescoring.

### Gemini 50 Comparison Snapshot

```text
snap_2026_05_11_de43f7bf0b_gemini50
```

Created by cloning the OpenAI 50 snapshot and clearing labels.

Current state: unlabeled. Gemini label command failed because the key was not available.

Resume command after setting the key:

```powershell
.\scripts\set_gemini_key.ps1

.\.venv\Scripts\python.exe -m bridgeengine.label `
  --snapshot snap_2026_05_11_de43f7bf0b_gemini50 `
  --vlm-backend gemini `
  --vlm-model gemini-2.5-flash
```

Then run:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot snap_2026_05_11_de43f7bf0b_gemini50

.\.venv\Scripts\python.exe -m bridgeengine.cost_probe `
  --snapshot snap_2026_05_11_de43f7bf0b_gemini50 `
  --projection 100 --projection 200 --projection 1000

.\.venv\Scripts\python.exe -m bridgeengine.label_compare `
  --left-snapshot snap_2026_05_11_de43f7bf0b `
  --right-snapshot snap_2026_05_11_de43f7bf0b_gemini50
```

### Remaining-50 Local Snapshot

```text
snap_2026_05_11_48710ffc52
```

This is the deterministic offset-50 ingest target from the same local 100-episode source.

Current state: unlabeled.

If the Gemini 50 cost probe is safely under the $10 budget, label this next:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label `
  --snapshot snap_2026_05_11_48710ffc52 `
  --vlm-backend gemini `
  --vlm-model gemini-2.5-flash
```

### All-100 Local Snapshot

```text
snap_2026_05_11_1dde3edf5d
```

This is the deterministic all-local-episodes snapshot.

Current state: unlabeled.

If cost is low enough, the simplest path is to label this directly:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label `
  --snapshot snap_2026_05_11_1dde3edf5d `
  --vlm-backend gemini `
  --vlm-model gemini-2.5-flash
```

That duplicates the first 50 Gemini labels if `gemini50` was already labeled. A better next engineering step is a label-merge utility that combines the first-50 and remaining-50 labeled snapshots into the all-100 snapshot without duplicate API spend.

## OpenAI vs Gemini Comparison Plan

Purpose: compare whether Gemini grades episodes similarly to OpenAI on the same 50 episodes.

Prepared command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label_compare `
  --left-snapshot snap_2026_05_11_de43f7bf0b `
  --right-snapshot snap_2026_05_11_de43f7bf0b_gemini50
```

Report metrics:

- overlap episode count
- curation-quality exact agreement
- curation-quality within-one agreement
- curation mean absolute difference
- task-success exact agreement
- task-success within-one agreement
- keep/reject decision agreement
- left and right score distributions
- top disagreements with reasons

Suggested interpretation:

- exact agreement around 50 percent can still be acceptable if within-one agreement is high and keep/reject agreement is high
- large keep/reject disagreement means the scoring/rubric is not stable enough to run scale-curve training
- Gemini quality distribution should not collapse to one or two scores

## Scale-Curve Plan

The desired first real scale curve:

```text
N = [25, 50, 100]
families = baseline, rich_text, rich_text_metadata, rich_text_metadata_subgoal
seeds = 2 or 3
held-out split fixed and disjoint
quality-stratified training mixtures enabled
```

Command shape:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.scale_curve `
  --snapshot <labeled_100_snapshot> `
  --sizes 25 50 100 `
  --heldout-count 10 `
  --quality-stratified `
  --seeds 0 1 `
  --output-dir scale_results\gemini_100
```

Do not run this until:

1. the Gemini labels pass quality gate
2. the OpenAI-vs-Gemini agreement is acceptable
3. Kevin approves spend/time
4. a labeled 100-episode snapshot exists

If only the first 50 are labeled, run a 25/50 plan only, not 100.

## Known Issues

1. Gemini live run is blocked until `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set for this repo.
2. No human gold set has been filled, so label reliability is heuristic/gate-based, not measured against human labels.
3. The 13-episode benchmark is real but smoke-scale and negative/inconclusive.
4. The 50-episode OpenAI labels are useful but were generated under an evolving scoring rubric.
5. The all-100 snapshot is ingested but unlabeled.
6. Tiered compression is net-negative at 50 episodes.
7. Standard RLDS/LeRobot export is not implemented.
8. Perceptive labelers exist as comparison wrappers but SAM/VDA/CoTracker runtime dependencies are not fully installed.
9. No label-merge utility exists yet to combine separately labeled 50+50 snapshots into a clean 100 snapshot without duplicate labeling.

## Files And Modules Almanac

Top-level reports:

- `README.md`: public-facing quickstart/status
- `ARCHITECTURE.md`: architecture sketch
- `VALUE_REPORT.md`: brutally honest value assessment
- `COST_PROBE_50.md`: OpenAI 50-episode cost and scale-out note, partly stale after gate fixes
- `STATUS.md`: current figure/status page
- `BUILD_REPORT.md`: original build report
- `DEVIATIONS.md`: deviations from earlier plan
- `BRIDGEENGINE_HANDOFF_ALMANAC.md`: this document

Core package:

- `bridgeengine/ingest/`: deterministic BridgeData ingest and Parquet snapshot schema
- `bridgeengine/labelers/`: VLM labelers, subgoal extraction, perception wrappers
- `bridgeengine/orchestrate/runner.py`: runs labelers and writes `labels.parquet`
- `bridgeengine/scoring.py`: curation scoring and quality reinterpretation
- `bridgeengine/quality_gate.py`: benchmark-grade label gate
- `bridgeengine/query/`: DuckDB helpers and canned queries
- `bridgeengine/export/`: deterministic cut export
- `bridgeengine/benchmark/`: LeWM train/eval grid and scale curve
- `bridgeengine/value.py`: value/anomaly scoring and tiered compression probe
- `bridgeengine/figures.py`: data-driven figures and threshold GIF
- `bridgeengine/viewer/`: Streamlit viewer
- `bridgeengine/goldset.py`: human-gold reliability scaffold
- `bridgeengine/cost_probe.py`: token/cost/wall-clock report for VLM labels
- `bridgeengine/label_compare.py`: model-to-model metadata grading comparison
- `bridgeengine/snapshot_clone.py`: clone deterministic snapshots under a new ID

Tests:

- `tests/test_workstream_a.py`: label provenance and gate behavior
- `tests/test_scoring.py`: curation scoring rules
- `tests/test_benchmark_smoke.py`: benchmark contract path
- `tests/test_goldset.py`: reliability scaffold
- `tests/test_query_compat.py`: old/new schema query compatibility
- other tests cover ingest, labelers, export, viewer-adjacent behavior

## Validation Status

Most recent full test run:

```text
20 passed in 319.71s
```

Most recent targeted checks:

```text
quality_report snap_2026_05_11_de43f7bf0b: PASS
targeted scoring/gate tests: PASS
figures regenerated: PASS
Gemini command: BLOCKED by missing key
```

## Recommended Next Work Order

1. Set Gemini key:

```powershell
.\scripts\set_gemini_key.ps1
```

2. Label Gemini 50 comparison snapshot:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot snap_2026_05_11_de43f7bf0b_gemini50 --vlm-backend gemini --vlm-model gemini-2.5-flash
```

3. Run quality/cost/compare:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot snap_2026_05_11_de43f7bf0b_gemini50
.\.venv\Scripts\python.exe -m bridgeengine.cost_probe --snapshot snap_2026_05_11_de43f7bf0b_gemini50 --projection 100 --projection 200 --projection 1000
.\.venv\Scripts\python.exe -m bridgeengine.label_compare --left-snapshot snap_2026_05_11_de43f7bf0b --right-snapshot snap_2026_05_11_de43f7bf0b_gemini50
```

4. Inspect top disagreements manually in the viewer or raw payloads.

5. If Gemini quality and cost are acceptable, label the remaining 50:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot snap_2026_05_11_48710ffc52 --vlm-backend gemini --vlm-model gemini-2.5-flash
```

6. Either implement a label-merge utility or label the all-100 snapshot directly.

7. Run the 25/50/100 real scale curve only after a labeled 100 snapshot exists.

8. Fill a human gold set for at least 13 to 25 episodes and run reliability reporting.

9. Update `VALUE_REPORT.md`, `STATUS.md`, and README with the Gemini comparison and scale-curve result.

## How To Demo Today

Start with the GIF:

```text
figures/threshold_annotation_animation.gif
```

The demo narrative:

1. "The system ingests BridgeData into deterministic Parquet snapshots."
2. "It uses VLMs to produce pi0.7-shaped labels: subtasks, quality, mistake, control mode, and subgoal frames."
3. "The quality score is not just task success. It asks whether the video has clear visible cause-effect boundaries useful for training."
4. "We found and corrected a scoring bug by visually inspecting the threshold GIF."
5. "The quality gate catches templated/fallback labels and score collapse."
6. "The first real 13-episode benchmark did not show a conditioning win, so the honest next step is measured label reliability plus a 100-episode scale curve."

Do not claim:

- that metadata improves performance
- that tiered compression saves space at 50 episodes
- that labels are human-gold validated
- that this is a production data platform

Do claim:

- the pipeline is end-to-end runnable
- the labels are queryable and inspectable
- bad labels can be gated before training
- a real LeWM smoke ablation exists
- the project has a clean path to Gemini-based 100-episode scale testing

