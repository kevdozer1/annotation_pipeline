# BridgeEngine: A Quality-Gated Semantic Annotation Layer for Robot Foundation-Model Data

Kevin Hopkins

Draft date: 2026-06-05

## Abstract

Robot foundation models are increasingly bottlenecked by data quality, annotation structure, and the ability to turn heterogeneous demonstrations into reliable training context. We present BridgeEngine, a local proof-of-concept data engine for converting BridgeData-style robot episodes into queryable, quality-gated, pi0.7-inspired conditioning artifacts. BridgeEngine ingests robot trajectories and videos into deterministic Parquet snapshots, generates structured semantic labels with hosted vision-language models, preserves raw label provenance, supports human score calibration through a dedicated review GUI, and evaluates label value with a real LeWorldModel frozen-adapter latent-MSE benchmark. On a 100-episode local BridgeData subset labeled with Gemini 2.5 Flash, Kevin reviewed all 100 clips and changed 58 curation scores, correcting the auto-label distribution from `{2: 5, 4: 9, 5: 86}` to `{2: 6, 3: 15, 4: 30, 5: 49}`. After applying this score-only human calibration, the metadata+subgoal conditioning family beats task-only baseline across N = 25, 50, and 100, including a 7.61% lower mean latent MSE at N = 100. The result remains smoke-scale: only two seeds are used, only quality scores were human-calibrated, and subtask boundaries/subgoal selections were not yet human-verified. BridgeEngine should therefore be interpreted as a working research scaffold for semantic robot-data curation and label-value evaluation, not as final evidence that pi0.7-style annotations reliably improve robot policy learning.

## 1. Introduction

Modern robot foundation-model work depends on large-scale demonstration data, but the practical usefulness of such data is shaped by more than raw trajectory count. Episodes differ in visual clarity, task completion, interaction structure, control format, and semantic usefulness. Recent robot learning systems have emphasized structured language and metadata as a way to exploit heterogeneous demonstrations rather than discarding all imperfect data. In particular, pi0.7-style diverse prompting motivates conditioning policies not only on a task instruction, but also on temporally localized subtask instructions, episode-level metadata, and subgoal imagery.

BridgeEngine asks a narrow engineering and research question: can a small, local pipeline turn raw robot episodes into structured, inspectable, quality-gated annotations, and can those annotations be connected to a real model-side value test? The project does not introduce a new robot policy architecture. Instead, it builds the missing substrate between raw robot data and controlled label-value experiments:

- deterministic snapshot storage for robot episodes
- VLM-derived subtask, metadata, and subgoal labels
- raw VLM provenance and cost accounting
- quality gates that block obviously bad labels from training
- human calibration tooling
- DuckDB query and deterministic export
- real LeWorldModel frozen-adapter evaluation across conditioning families

The core claim of this draft is modest: BridgeEngine is a functional POC for semantic robot-data conditioning and curation. The current benchmark is encouraging, but not definitive.

## 2. Background And Motivation

### 2.1 Robot Data Needs Structured Context

Robot datasets often contain mixed-quality demonstrations. Some episodes cleanly execute the intended task with visible cause-effect boundaries. Others are partially successful, occluded, ambiguous, or fail outright. Treating these episodes as uniformly valuable can waste training capacity or introduce noisy supervision. Discarding all imperfect data can also be wasteful, especially when imperfections are structured and could be made visible to the model through metadata.

The relevant data problem is therefore not just storage or conversion. A useful robot-data engine should answer:

1. What happened in this episode?
2. Was the intended task completed?
3. Were the manipulation boundaries visible enough to learn from?
4. What subtask is active at each point?
5. Which episodes are clear keeps, near keeps, near rejects, or clear rejects?
6. Can we measure whether these labels help a downstream model?

### 2.2 pi0.7-Style Diverse Prompting

The motivating annotation pattern is pi0.7-style diverse prompting: a robot model receives a task instruction plus optional prompt components such as temporally segmented subtask text, episode quality metadata, mistake indicators, control mode, and subgoal imagery. The key idea is not merely "caption the episode." It is to provide structured context at multiple granularities and train with dropout so the model can use whichever fields are available at inference.

BridgeEngine implements a POC approximation of this idea. It uses VLM-derived subtask segmentation, VLM-derived episode metadata, deterministic control-mode labels, and actual end-of-segment frames as subgoal images. It does not generate future subgoal images with a learned world model. This is an explicit simplification for tractability.

### 2.3 Why A Quality Gate Matters

Early BridgeEngine runs showed that deterministic fallback labels can pass through the pipeline while being semantically useless. For example, a fallback segmenter produced repeated "approach, grasp, transport, place" patterns at fixed episode quarters. Such labels are useful for debugging storage and training contracts, but they cannot support scientific claims. BridgeEngine therefore treats label provenance and quality gating as first-class parts of the system. The pipeline records which backend actually produced each label and blocks benchmark runs when label patterns are obviously collapsed, repetitive, contradictory, or ungrounded.

## 3. System Overview

BridgeEngine is organized around immutable local snapshots. A snapshot contains episode metadata, steps, sensors, labels, and a manifest:

```text
bridgeengine_data/
  snapshots/
    <snapshot_id>/
      manifest.json
      episodes.parquet
      steps.parquet
      sensors.parquet
      labels.parquet
      labels/
      subgoals/
      raw_vlm_outputs/
      gold/
```

The POC operates on a local BridgeData-style subset at:

```text
D:\bridgedata_v2_subset
```

The current local source exposes 100 episodes.

### 3.1 Snapshot Storage

BridgeEngine writes plain Parquet files rather than adopting a production table format such as Iceberg or Delta. This keeps the POC simple and reproducible. Snapshot IDs are deterministic over the source episode set and transform configuration, so the same local data and ingest settings reproduce the same snapshot ID.

Key snapshot tables:

- `episodes.parquet`: episode ID, video path, frame path, task instruction, number of steps, value score columns
- `steps.parquet`: timestep index, action, state, timestamp
- `sensors.parquet`: sensor metadata
- `labels.parquet`: label rows with payload paths, provenance, confidence, metadata JSON, segment index, and subgoal image paths

### 3.2 Labeling Pipeline

The main semantic labelers are:

1. `subtask_segmenter`: produces 2-5 temporally ordered subtask segments with `(start_step, end_step, subtask_text)`.
2. `episode_metadata`: produces speed, task-success quality, curation quality, mistake, control mode, boundary clarity, and reason fields.
3. `subgoal_images`: extracts the end-of-segment frame for each subtask.

BridgeEngine uses a two-stage observe-then-label flow. The first VLM call asks for visual observations grounded in keyframes. The second call converts those observations into structured JSON labels. This separation makes failure modes easier to inspect and reduces the chance that the model invents labels without grounding them in visible evidence.

Supported VLM backends in the current repo:

- `mock`: deterministic scaffolding backend for CI and plumbing tests
- `openai`: hosted backend used in earlier 50-episode labels
- `gemini`: Gemini Generate Content backend used for the 100-episode scale probe

The Gemini backend includes retry handling, cached raw response reuse, cost estimation, and sanitized error handling.

### 3.3 Quality Gate

The quality gate checks whether labels are benchmark-grade before training. It currently rejects:

- repeated subtask text across segments
- metadata score/reason contradictions
- collapsed score distributions
- subtask objects not grounded in stage-one observations, after filtering non-object/function words

The gate is intentionally heuristic. It catches obvious failures, but it does not replace human reliability measurement.

### 3.4 Human Calibration GUI

The Streamlit viewer now includes a `Calibration` tab. It lets a human reviewer:

- scroll through every episode in the selected snapshot
- watch the episode video or frame sequence
- inspect auto subtask timelines
- inspect subgoal frames
- inspect VLM metadata and scoring reasons
- assign a calibrated 1-5 curation score
- mark visible mistakes
- write review notes
- optionally accept auto subtask boundaries and subgoal selections

Reviews are saved in the existing gold-set format:

```text
bridgeengine_data/snapshots/<snapshot_id>/gold/calibration_gold.json
```

This file can be passed directly to:

```powershell
python -m bridgeengine.goldset report --snapshot <snapshot_id> --gold-file <gold_file>
```

### 3.5 Query, Export, And Viewer

BridgeEngine provides a DuckDB query interface over Parquet snapshots and pre-canned queries for coverage, metadata distributions, subgoal paths, and pi0.7-style prompt traces. It also provides deterministic training-cut export. The current export is BridgeEngine-native; standard LeRobot and RLDS export remain future work.

### 3.6 Value-Aware Curation

BridgeEngine includes a value/anomaly scoring module with two methods:

- prediction-error scoring from LeWorldModel held-out latent error
- embedding-distance fallback based on episode embedding distance from the corpus centroid or density proxy

Value scores can be written back into `episodes.parquet`, ranked, and used for tiered-compression probes. On the current tiny local corpus, tiered compression is net-negative because per-file overhead dominates. This is documented as a small-corpus limitation rather than a claimed storage win.

## 4. Annotation Schema

Each episode receives three semantic annotation families.

### 4.1 Subtask Segments

Subtask segments are stored as a JSON payload and referenced from `labels.parquet`:

```json
{
  "segments": [
    {
      "segment_idx": 0,
      "start_step": 0,
      "end_step": 4,
      "subtask_text": "approach the pan"
    }
  ]
}
```

The segment text is used to form the active prompt field:

```text
Task: put pan on stove from sink. Subtask: lift pan from stove.
```

### 4.2 Episode Metadata

Metadata includes:

- `speed`: number of timesteps
- `task_success_quality`: VLM judgment of task completion
- `curation_quality`: revised score focused on training usefulness and visible boundaries
- `curation_keep`: boolean thresholded at score >= 4
- `mistake`: visible mistake flag
- `control_mode`: defaults to `end_effector` for BridgeData-style actions
- `boundary_clarity`: weak, partial, or clear
- `interaction_structure_score`: simple evidence score from subtask verbs
- `reason`: VLM explanation
- `scoring_reason`: deterministic curation-score reason

The current score semantics used by the calibration GUI are:

| Score | Label | Meaning |
|---:|---|---|
| 1 | clear reject | not usable for visible manipulation-boundary learning |
| 2 | reject | weak or wrong interaction, but not totally empty |
| 3 | near reject | some useful evidence, but too occluded/incomplete/ambiguous for the main keep set |
| 4 | near keep | usable, with some imperfection |
| 5 | clear keep | clean visible cause-effect interaction with clear boundaries |

### 4.3 Subgoal Images

For each subtask segment, BridgeEngine extracts the last frame of the segment as a JPEG subgoal proxy. This approximates pi0.7-style subgoal imagery without training a separate future-image generator. The simplification is explicit and should be replaced by generated subgoals in a larger project.

## 5. Experimental Design

### 5.1 Dataset

Experiments use a local 100-episode BridgeData-style subset. The first OpenAI probe used 50 episodes. The current scale curve uses the all-100 snapshot:

```text
snap_2026_05_11_1dde3edf5d
```

The 100-episode labels were generated by labeling two 50-episode slices with Gemini 2.5 Flash and merging their label artifacts into the all-100 snapshot without duplicate API spend.

### 5.2 Labeling Backend And Cost

Gemini 2.5 Flash produced 493 label rows over 100 episodes:

- 100 metadata rows
- 100 subtask rows
- 293 subgoal rows

Measured labeling cost:

| Episodes | Total cost | Cost per episode | Wall-clock per episode |
|---:|---:|---:|---:|
| 100 | $1.188603 | $0.011886 | 16.69 s |

Projected costs at the measured rate:

| Episodes | Projected cost | Projected serial time |
|---:|---:|---:|
| 200 | $2.38 | 0.93 h |
| 1,000 | $11.89 | 4.64 h |
| 60,000 | $713.16 | 278.24 h |

These projections assume similar episode lengths, similar prompt sizes, and unchanged Gemini pricing.

### 5.3 OpenAI vs Gemini Grading Comparison

OpenAI and Gemini were compared on the same first 50 episodes.

| Metric | Value |
|---|---:|
| Overlapping episodes | 50 |
| Curation exact agreement | 0.440 |
| Curation within-one agreement | 0.860 |
| Mean absolute curation difference | 0.800 |
| Task-success exact agreement | 0.420 |
| Task-success within-one agreement | 0.760 |
| Keep-decision agreement | 0.860 |

Score distributions:

| Backend | Distribution |
|---|---|
| OpenAI | `{1: 2, 2: 1, 3: 5, 4: 17, 5: 25}` |
| Gemini | `{2: 3, 4: 5, 5: 42}` |

Gemini is much cheaper and reaches high keep/reject agreement, but it is substantially more generous. This is the main calibration concern.

### 5.4 Human Calibration

Kevin reviewed all 100 clips in the dedicated browser GUI. The review pass intentionally calibrated only the score. No free-form review notes were entered, and the subtask-boundary/subgoal approval checkboxes were not used. The GUI did save default reason text and some default metadata-accept flags, but this draft treats those as interface defaults rather than intentional human labels.

Human score calibration summary:

| Metric | Value |
|---|---:|
| Reviewed episodes | 100 / 100 |
| Score changes versus Gemini auto curation | 58 / 100 |
| Review notes intentionally entered | 0 |
| Subtask-boundary approvals | 0 |
| Subgoal-selection approvals | 0 |
| Quality exact agreement before applying gold | 0.42 |
| Quality within-one agreement before applying gold | 0.77 |
| Keep/reject agreement before applying gold | 0.76 |
| Mean temporal boundary IoU | n/a, not reviewed |
| Subgoal selection agreement | n/a, not reviewed |

Score distributions:

| Source | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Gemini auto curation | 0 | 5 | 0 | 9 | 86 |
| Kevin human calibration | 0 | 6 | 15 | 30 | 49 |

The calibrated scores were applied to a cloned snapshot, not the original Gemini snapshot:

```text
snap_2026_05_11_1dde3edf5d_human_calibrated
```

This makes the benchmark use Kevin's score thresholding while preserving the original Gemini labels for comparison.

### 5.5 Conditioning Families

The benchmark evaluates four conditioning families:

1. `baseline`: task instruction only
2. `rich_text`: task instruction plus active VLM-derived subtask text
3. `rich_text_metadata`: rich text plus speed, quality, mistake, and control metadata
4. `rich_text_metadata_subgoal`: metadata prompt plus end-of-segment subgoal image

The model path uses a real LeWorldModel frozen-adapter train/eval loop. Text fields are hashed into a learned conditioning adapter, and subgoal images are encoded through the frozen LeWM image path. This is a practical smoke benchmark, not a full VLA policy training run.

### 5.6 Scale Curve

The current scale curve uses:

- N = 25, 50, 100
- two seeds per family
- held-out count = 10
- quality-stratified training mixtures
- CUDA
- held-out latent MSE as the metric

After score calibration, the held-out split contains mixed-quality episodes:

```text
{2: 1, 3: 1, 4: 1, 5: 7}
```

The train splits are also quality-stratified. This is a material improvement over the Gemini-only curve, where the held-out split contained only `5/5` episodes.

## 6. Results

### 6.1 Quality Gate

The human-calibrated 100-episode snapshot passes the quality gate:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {2: 6, 3: 15, 4: 30, 5: 49}
```

The pass confirms that the calibrated labels are not obviously collapsed, repeated, contradictory, or ungrounded under the current heuristic gate. It does not establish boundary or subgoal reliability, because those fields were not reviewed.

### 6.2 Scale-Curve Results

Mean held-out latent MSE after score-only human calibration:

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

Lower is better. The metadata+subgoal family beats baseline at all three sizes. At N = 100, the best mean is `rich_text_metadata_subgoal`, with 0.020807 latent MSE versus 0.022522 for baseline. Text-only and metadata-only improve at N = 25 and N = 50, but regress slightly at N = 100.

### 6.3 Interpretation

The result is stronger than the Gemini-only curve because score calibration corrected the top-heavy auto distribution and produced a mixed-quality held-out split. It is also stronger than the earlier 13-episode OpenAI smoke result, which did not show a metadata win. However, the current experiment still has three important weaknesses:

1. only two seeds per family
2. only quality scores were human-calibrated; subtask boundaries and subgoal selections were not reviewed
3. the benchmark is a LeWM frozen-adapter latent-MSE probe, not full robot policy training

Therefore, the correct claim is:

> BridgeEngine produces a working pi0.7-style annotation and curation pipeline, and a score-calibrated 100-episode scale probe shows metadata+subgoal conditioning beating baseline in mean latent MSE. The result is not yet a robust scientific conclusion.

## 7. Ablations And Diagnostics

### 7.1 Mock Backend Failure

The mock backend produces deterministic scaffold labels and is useful for CI. It should not be used for benchmark claims. It fails quality gating under score-dispersion checks on sufficiently large snapshots.

### 7.2 Model-Grader Disagreement

OpenAI and Gemini agree on keep/reject decisions for 86% of the first 50 episodes but differ substantially in exact curation scores. Gemini's generous scoring can make the dataset appear cleaner than it is. This motivates the calibration GUI and gold-set report.

### 7.3 Value-Aware Curation

The current value-aware module can rank episodes by prediction-error or embedding-distance anomaly scores. It is useful for surfacing outliers and prioritizing inspection. The current tiered-compression probe is not positive at tiny scale, because file-layout overhead dominates compression savings.

## 8. Limitations

BridgeEngine has several known limitations.

First, the current dataset is small. The local SSD source exposes 100 episodes, not the full BridgeData V2 corpus. Scaling beyond 100 requires downloading or mounting more data.

Second, human calibration is partial. Kevin reviewed all 100 clips and calibrated scores, but did not review subtask boundaries, subgoal selections, free-form notes, or calibration reasons. Boundary IoU and subgoal agreement are therefore still unknown.

Third, the benchmark is a frozen-adapter LeWorldModel smoke test, not full policy training. It measures held-out latent MSE, not robot task success.

Fourth, the language conditioning path is practical but limited. Text fields are hashed into a learned adapter, rather than using the exact language-token path of a production VLA model.

Fifth, subgoal images are actual end-of-segment frames, not generated future subgoals. This is useful for a tractable POC but not fully faithful to systems that generate subgoal imagery.

Sixth, the current export path is BridgeEngine-native. Standard LeRobot and RLDS export are not yet implemented.

Seventh, Gemini scoring was top-heavy before calibration. Human score review corrected that distribution and changed the benchmark, which is useful evidence that manual calibration materially affects conclusions. The remaining risk is that subtask and subgoal fields may still contain unmeasured VLM errors.

## 9. What Would Make This Publishable

The project would need the following before being framed as a credible scientific result:

1. Human boundary and subgoal calibration over at least 25-50 episodes.
2. At least 3 seeds per family.
3. More than 100 episodes, ideally a 200-1000 episode curve if cost and time permit.
4. A held-out split that stays fixed across larger sizes and remains mixed quality.
5. Standard export or a clearly documented downstream integration path.
6. A stronger language-conditioning interface if the target audience expects VLA-native token conditioning rather than hashed adapters.
7. A rerun after any prompt/rubric changes that materially alter subtask boundaries or subgoal selections.

## 10. Practical Utility For Foundation-Model Data Work

Even before it is publishable as a scientific result, BridgeEngine is useful as an engineering artifact. It makes several hidden parts of robot foundation-model data work concrete:

- annotation provenance matters
- fallback labels can be dangerous if not surfaced
- semantic labels need quality gates before benchmark entry
- model graders can disagree even when both seem plausible
- human calibration should be part of the pipeline, not an afterthought
- label-value claims need an actual model-side evaluation path
- small-scale positive trends should be reported with restraint

This makes BridgeEngine a strong internal demo or learning scaffold for someone entering robot foundation-model data work. A cleaned-up public version should be modularized around adapters:

- dataset adapters
- VLM backends
- prompt templates
- scoring policies
- query backends
- export formats
- evaluation backends

The current repo is the maximum-functionality slice. The eventual public repo should be smaller, cleaner, and less tied to Kevin's local BridgeData and LeWM setup.

## 11. Reproducibility

### 11.1 Start Viewer

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\.venv\Scripts\streamlit.exe run bridgeengine\viewer\app.py --server.port 8765
```

### 11.2 Run Quality Report

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot snap_2026_05_11_1dde3edf5d
```

Expected current output:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {2: 5, 4: 9, 5: 86}
```

For the score-calibrated benchmark snapshot:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot snap_2026_05_11_1dde3edf5d_human_calibrated
```

Expected current output:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {2: 6, 3: 15, 4: 30, 5: 49}
```

### 11.3 Run Cost Probe

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.cost_probe `
  --snapshot snap_2026_05_11_1dde3edf5d `
  --projection 200 --projection 1000 --projection 60000
```

Expected current cost:

```text
Estimated total cost: $1.188603
Estimated cost per episode: $0.011886
```

### 11.4 Run Human Reliability Report

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.goldset report `
  --snapshot snap_2026_05_11_1dde3edf5d `
  --gold-file bridgeengine_data\snapshots\snap_2026_05_11_1dde3edf5d\gold\calibration_gold.json
```

Current score-only calibration report:

```text
reviewed episodes: 100 / 100
quality exact agreement: 0.42
quality within-one agreement: 0.77
keep/reject agreement: 0.76
subtask-boundary IoU: n/a, not reviewed
subgoal agreement: n/a, not reviewed
```

### 11.5 Scale-Curve Result Files

```text
scale_results/human_calibrated_100/scale_curve_results.csv
scale_results/human_calibrated_100/scale_curve_summary.md
figures/scale_curve_human_calibrated_100.png
```

## 12. Conclusion

BridgeEngine demonstrates that a small local robot-data pipeline can go beyond raw storage and produce structured, quality-gated, pi0.7-style conditioning artifacts with provenance, human score calibration, queryability, and a real model-side label-value benchmark. The current score-calibrated 100-episode scale curve is positive for metadata+subgoal conditioning, but not definitive. The immediate next step is not more engineering breadth; it is measuring boundary and subgoal reliability, adding more seeds, and scaling beyond 100 episodes only after a fresh cost gate.

## Appendix A. Current Artifact Inventory

Key files:

```text
BRIDGEENGINE_HANDOFF_ALMANAC.md
VALUE_REPORT.md
FINAL_STATE_ASSESSMENT.md
scale_results/human_calibrated_100/scale_curve_summary.md
figures/scale_curve_human_calibrated_100.png
bridgeengine/viewer/app.py
bridgeengine/review_gui.py
bridgeengine/apply_gold.py
bridgeengine/calibration.py
bridgeengine/goldset.py
bridgeengine/scoring.py
bridgeengine/quality_gate.py
bridgeengine/benchmark/
```

Key snapshot:

```text
snap_2026_05_11_1dde3edf5d_human_calibrated
```

Key claim status:

| Claim | Status |
|---|---|
| Pipeline ingests local BridgeData-style episodes | Supported |
| Gemini labels are cheap enough for small scale probes | Supported |
| Quality gate catches obvious scaffold-label failures | Supported |
| Human calibration GUI exists | Supported |
| Human score calibration is measured | Supported |
| Human boundary/subgoal reliability is measured | Not yet |
| Metadata+subgoal conditioning beats baseline at 100 episodes | Supported as smoke-scale trend |
| Richer conditioning robustly improves robot learning | Not proven |
| Public reusable toolkit is ready | Not yet |

## Appendix B. Human Calibration Notes

The previous draft contained placeholder human-rated reliability numbers. Those have been replaced for curation scores:

- reviewed episodes: `100 / 100`
- score changes versus Gemini auto curation: `58 / 100`
- quality exact agreement before applying gold: `0.42`
- quality within-one agreement before applying gold: `0.77`
- keep/reject agreement before applying gold: `0.76`

The following values remain unmeasured because Kevin did not use the review toggles for those fields:

- mean temporal boundary IoU
- subgoal selection agreement
- free-form calibration-reason quality
