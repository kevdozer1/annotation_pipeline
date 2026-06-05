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

Current honest scientific result: the first real 100-episode Gemini-labeled LeWM scale curve exists. Richer conditioning improves mean held-out latent MSE at 25 and 50 episodes, and the subgoal family remains slightly better than baseline at 100 episodes, but the 100-episode gap is small and within two-seed noise. Treat this as a smoke-scale trend probe, not a robust robotics result.

Current main blockers: no human gold set has been filled, Gemini grading is useful but top-heavy, and the local SSD source exposes only 100 episodes. Larger scale requires downloading more BridgeData V2 or pointing ingest at a larger local source.

Latest live state:

- Gemini 50 comparison snapshot is labeled, gate-passing, and much cheaper than OpenAI.
- Remaining local 50 episodes are labeled with Gemini and gate-passing.
- The all-100 snapshot has merged Gemini labels without duplicate API spend.
- A real LeWM scale curve over N = 25, 50, 100 has run with two seeds per family.

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

Created by cloning the OpenAI 50 snapshot and clearing labels. It has now been relabeled with Gemini 2.5 Flash.

Label command used:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label `
  --snapshot snap_2026_05_11_de43f7bf0b_gemini50 `
  --vlm-backend gemini `
  --vlm-model gemini-2.5-flash
```

Observed labeling output:

```text
label_rows: 247
episode_metadata rows: 50
subtask_segmenter rows: 50
subgoal_images rows: 147
semantic label wall-clock: 406.31s
```

Gate and cost:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {2: 3, 4: 5, 5: 42}
Estimated total cost: $0.627729
Cost per episode: $0.012555
Projected cost at 100: $1.26
Projected cost at 200: $2.51
Projected cost at 1000: $12.55
```

Key comparison against OpenAI 50:

```text
overlap: 50 episodes
curation exact agreement: 0.440
curation within-one agreement: 0.860
mean absolute curation difference: 0.800
task-success exact agreement: 0.420
task-success within-one agreement: 0.760
keep-decision agreement: 0.860
OpenAI distribution: {1: 2, 2: 1, 3: 5, 4: 17, 5: 25}
Gemini distribution: {2: 3, 4: 5, 5: 42}
```

Interpretation: Gemini is dramatically cheaper and gate-passing, but much more generous. The high keep-decision agreement is useful for coarse curation; the top-heavy score distribution is a limitation for metadata-conditioning experiments.

### Remaining-50 Local Snapshot

```text
snap_2026_05_11_48710ffc52
```

This is the deterministic offset-50 ingest target from the same local 100-episode source.

Current state: labeled with Gemini 2.5 Flash.

Label command used:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label `
  --snapshot snap_2026_05_11_48710ffc52 `
  --vlm-backend gemini `
  --vlm-model gemini-2.5-flash
```

Observed output:

```text
label_rows: 246
Quality gate: PASS
Quality counts: {2: 2, 4: 4, 5: 44}
Estimated total cost: $0.560874
Cost per episode: $0.011217
semantic label wall-clock: 1263.13s
mistake distribution: false 45, true 5
```

### All-100 Local Snapshot

```text
snap_2026_05_11_1dde3edf5d
```

This is the deterministic all-local-episodes snapshot.

Current state: labeled by merging the Gemini first-50 and remaining-50 label artifacts, avoiding duplicate API spend.

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.snapshot_merge `
  --target-snapshot snap_2026_05_11_1dde3edf5d `
  --source-snapshot snap_2026_05_11_de43f7bf0b_gemini50 `
  --source-snapshot snap_2026_05_11_48710ffc52 `
  --overwrite-labels
```

Merged output:

```text
merged_label_rows: 493
merged_metadata_episode_count: 100
missing_metadata_episode_count: 0
```

Merged all-100 gate and cost:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {2: 5, 4: 9, 5: 86}
Mistake distribution: false 89, true 11
Estimated total cost: $1.188603
Cost per episode: $0.011886
Projected cost at 200: $2.38
Projected cost at 1000: $11.89
Projected cost at 60000: $713.16
semantic label wall-clock total: 1669.44s
```

Important caveat: the merged all-100 labels are benchmark-usable according to the heuristic gate, but the quality distribution is strongly top-heavy. Human gold labels are still needed before treating the score calibration as reliable.

## OpenAI vs Gemini Comparison Result

Purpose: compare whether Gemini grades episodes similarly to OpenAI on the same 50 episodes.

Command:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label_compare `
  --left-snapshot snap_2026_05_11_de43f7bf0b `
  --right-snapshot snap_2026_05_11_de43f7bf0b_gemini50
```

Observed metrics:

```text
overlap episode count: 50
curation-quality exact agreement: 0.440
curation-quality within-one agreement: 0.860
curation mean absolute difference: 0.800
task-success exact agreement: 0.420
task-success within-one agreement: 0.760
keep/reject decision agreement: 0.860
OpenAI curation distribution: {1: 2, 2: 1, 3: 5, 4: 17, 5: 25}
Gemini curation distribution: {2: 3, 4: 5, 5: 42}
```

Largest disagreements:

```text
episode_001972: OpenAI 1, Gemini 5, task put pan on stove from sink
episode_013031: OpenAI 1, Gemini 5, task put spoon in pot
episode_052920: OpenAI 5, Gemini 2, task put cup from counter or drying rack into sink
episode_005164: OpenAI 3, Gemini 5, task put cup from anywhere into sink
episode_006115: OpenAI 3, Gemini 5, task put pot on stove which is near stove
```

Interpretation: Gemini is good enough for a low-cost scale probe, but not yet proven as a stable grader. It agrees with OpenAI on the keep/reject decision for 86 percent of the first 50, while assigning many more `5/5` curation scores. The next reliability step is human gold scoring on the disagreement set, not more model-to-model debate.

## Scale-Curve Result

The first real scale curve has now run:

```text
N = [25, 50, 100]
families = baseline, rich_text, rich_text_metadata, rich_text_metadata_subgoal
seeds = 2
held-out split fixed and disjoint for each N
quality-stratified training mixtures enabled
backend = real_lewm_frozen_adapter
device = CUDA
snapshot = snap_2026_05_11_1dde3edf5d
```

Command used:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.scale_curve `
  --snapshot snap_2026_05_11_1dde3edf5d `
  --sizes 25 50 100 `
  --heldout-count 10 `
  --quality-stratified `
  --benchmark-seeds 0 1 `
  --output-dir scale_results\gemini_100 `
  --run
```

Result files:

```text
scale_results/gemini_100/scale_curve_results.csv
scale_results/gemini_100/scale_curve.png
scale_results/gemini_100/scale_curve_plan.json
```

Mean held-out latent MSE:

| N | baseline | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|---:|
| 25 | 0.044892 | 0.045292 | 0.041925 | 0.041327 |
| 50 | 0.022307 | 0.022268 | 0.020579 | 0.021179 |
| 100 | 0.016242 | 0.016079 | 0.016096 | 0.015647 |

Delta versus baseline, lower is better:

| N | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|
| 25 | +0.89% | -6.61% | -7.94% |
| 50 | -0.17% | -7.74% | -5.06% |
| 100 | -1.00% | -0.90% | -3.67% |

Interpretation: richer conditioning trends better than baseline at all three sizes, with the strongest mean gap at 25 and 50 episodes. At 100 episodes the gap shrinks and overlaps seed noise. This is a useful smoke-scale trend, not a claim that pi0.7-style conditioning reliably improves robot policy learning.

Important split caveat: the held-out split is all `5/5` under Gemini's curation scores because the Gemini distribution is top-heavy. The quality-stratified training pools include the available `2/5` and `4/5` episodes, but the held-out mix does not test low-quality generalization yet.

## Known Issues

1. No human gold set has been filled, so label reliability is heuristic/gate-based, not measured against human labels.
2. Gemini grading is top-heavy: 86 of 100 local episodes are `5/5`, which may undercut metadata-conditioning experiments.
3. The 100-episode scale curve is real but smoke-scale, two-seed, and still not a robust robotics conclusion.
4. The held-out scale-curve split contains only Gemini `5/5` episodes, so low-quality generalization is not evaluated yet.
5. The 50-episode OpenAI labels are useful but were generated under an evolving scoring rubric.
6. Tiered compression is net-negative at 50 episodes because per-file/layout overhead dominates.
7. Standard RLDS/LeRobot export is not implemented.
8. Perceptive labelers exist as comparison wrappers but SAM/VDA/CoTracker runtime dependencies are not fully installed.
9. The local SSD source exposes 100 episodes, not the full BridgeData V2 corpus. Larger N requires downloading more data or pointing ingest at a larger source.

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
- `bridgeengine/snapshot_merge.py`: merge label artifacts from labeled slice snapshots into a target snapshot without duplicate API spend

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
quality_report snap_2026_05_11_de43f7bf0b_gemini50: PASS
quality_report snap_2026_05_11_48710ffc52: PASS
quality_report snap_2026_05_11_1dde3edf5d: PASS
targeted scoring/gate tests: PASS
figures regenerated: PASS
Gemini 50 labeling: PASS
Gemini remaining-50 labeling: PASS
All-100 label merge: PASS
Real scale curve N=25/50/100, 2 seeds: PASS
```

## Recommended Next Work Order

1. Inspect Gemini/OpenAI disagreement episodes manually in the viewer, especially episodes where one model says reject and the other says clear keep.
2. Fill a human gold set for at least 13 to 25 episodes and run reliability reporting.
3. Calibrate the quality rubric against the gold set, with special attention to Gemini's overuse of `5/5`.
4. Re-run the scale curve after calibration if the human-gold changes materially alter the score distribution.
5. Download or expose more BridgeData V2 episodes if Kevin wants N > 100. Do not spend on full-corpus labeling without a fresh cost gate.
6. Implement RLDS or LeRobot export only if the project is meant to interoperate with standard training stacks rather than stay as a BridgeEngine-native POC.
7. Update `VALUE_REPORT.md`, `STATUS.md`, and README with the Gemini comparison and scale-curve result before publicizing the repo.

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
