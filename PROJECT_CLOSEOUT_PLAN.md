# BridgeEngine Closeout Plan

Last updated: 2026-06-07

## Current Direction

There are now two deliverables:

1. **Scientific claim:** on the same 100 BridgeData subset, test whether pi0.7-style semantic annotations beat the perceptive vision signals from the previous LeWM experiment.
2. **Public product:** a configurable pi0.7-style annotation pipeline that other people can adapt to robot manipulation video, and eventually human-motion or general video, by bringing their own dataset adapters, VLM providers, and rubrics.

Do the scientific claim first. The internal experiment needs a frozen rubric and fixed labels. The public product can be generalized after the result is locked.

## What Is Already Done

- 100 local BridgeData episodes labeled with Gemini.
- Kevin reviewed all 100 clips and changed 58 curation scores.
- Kevin reviewed timestamp boundaries on the 50-episode boundary/subgoal subset.
- Subgoal gold labels were derived from reviewed subtask end boundaries.
- Current human-gold-label snapshot created:

```text
snap_2026_05_11_1dde3edf5d_human_gold_labels
```

- Calibrated quality gate passes:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {2: 6, 3: 15, 4: 30, 5: 49}
```

- Source Gemini reliability against Kevin's reviewed labels:

```text
quality exact agreement: 0.42
quality within-one agreement: 0.77
subtask-boundary temporal IoU mean: 0.683
derived subgoal frame agreement: 0.347
```

- The human-gold-label 100-episode LeWM scale curve is positive for metadata+subgoal conditioning:

```text
N=100 baseline:                    0.021839
N=100 rich_text_metadata_subgoal:  0.020468
delta:                            -6.28%
```

This is still a two-seed smoke result.

## Next Critical Path

The next task is not more score review. The current blocker is real perceptive labels for the controlled pi0.7-vs-perceptive comparison.

Check readiness:

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\.venv\Scripts\python.exe -m bridgeengine.perceptive_status `
  --snapshot snap_2026_05_11_1dde3edf5d_human_gold_labels `
  --require-real
```

Current state: not ready. The human-gold-label snapshot does not yet have perceptive mask/depth/track rows.

## Reproducing Human-Gold Application

Run:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.derive_subgoals `
  --snapshot snap_2026_05_11_1dde3edf5d `
  --gold-file bridgeengine_data\snapshots\snap_2026_05_11_1dde3edf5d\gold\calibration_gold.json

.\.venv\Scripts\python.exe -m bridgeengine.apply_gold `
  --source-snapshot snap_2026_05_11_1dde3edf5d `
  --target-snapshot snap_2026_05_11_1dde3edf5d_human_gold_labels `
  --gold-file bridgeengine_data\snapshots\snap_2026_05_11_1dde3edf5d\gold\calibration_gold.json `
  --overwrite
```

## Perception Critical Path

The headline comparison cannot run until perceptive signals are real.

Check readiness:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.perceptive_status `
  --snapshot snap_2026_05_11_1dde3edf5d_human_gold_labels `
  --require-real
```

Current state: not ready. The human-gold-label snapshot does not yet have perceptive mask/depth/track rows.

The final comparison must not use synthetic perceptive fallbacks. If this command fails, fix SAM/VDA/CoTracker artifact generation or loading first.

## Final Experiment

Preregistered comparison is in:

```text
HEAD_TO_HEAD_PREREGISTRATION.md
```

Primary contrast:

```text
pi07_full = rich_text_metadata_subgoal
perceptive_all = masks + depth + tracks
```

Use:

- same 100 episodes
- same held-out split
- same LeWM frozen-adapter metric
- unified conditioning interface
- seeds `0,1,2,3,4`
- no synthetic perception labels

## Why Score Disagreement Is Not Fatal

Gemini and OpenAI did not match Kevin's scores exactly. That is not a reason to hide the result. It is the point of BridgeEngine:

- hosted VLMs are useful first-pass annotators
- VLM scores need calibration to a task-specific rubric
- human review changed the score distribution materially
- after calibration, the benchmark result became stronger

The public release should lean into that. BridgeEngine is not "trust the VLM." It is "annotate, inspect, calibrate, gate, and measure."

## Public Fork

The public fork should happen after the controlled comparison. The release plan is in:

```text
PUBLIC_RELEASE_PLAN.md
```

Public claim:

```text
A configurable pi0.7-style annotation and curation pipeline for video datasets, validated on robot manipulation video.
```

Do not claim that it is already validated for human-motion or general-video datasets. Say it is architected to adapt to those classes through rubrics and adapters.
