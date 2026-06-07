# BridgeEngine Closeout Plan

Last updated: 2026-06-06

## Current Direction

There are now two deliverables:

1. **Scientific claim:** on the same 100 BridgeData subset, test whether pi0.7-style semantic annotations beat the perceptive vision signals from the previous LeWM experiment.
2. **Public product:** a configurable pi0.7-style annotation pipeline that other people can adapt to robot manipulation video, and eventually human-motion or general video, by bringing their own dataset adapters, VLM providers, and rubrics.

Do the scientific claim first. The internal experiment needs a frozen rubric and fixed labels. The public product can be generalized after the result is locked.

## What Is Already Done

- 100 local BridgeData episodes labeled with Gemini.
- Kevin reviewed all 100 clips and changed 58 curation scores.
- Score-calibrated snapshot created:

```text
snap_2026_05_11_1dde3edf5d_human_calibrated
```

- Calibrated quality gate passes:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {2: 6, 3: 15, 4: 30, 5: 49}
```

- The score-calibrated 100-episode LeWM scale curve is positive for metadata+subgoal conditioning:

```text
N=100 baseline:                    0.022522
N=100 rich_text_metadata_subgoal:  0.020807
delta:                            -7.61%
```

This is still a two-seed smoke result.

## Next Human Review Command

The next human task is **not rescoring**. It is boundary/subgoal reliability.

Start the GUI:

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\.venv\Scripts\python.exe -m bridgeengine.review_gui `
  --snapshot snap_2026_05_11_1dde3edf5d `
  --gold-file C:\Users\Kevin\projects\annotation_pipeline\bridgeengine_data\snapshots\snap_2026_05_11_1dde3edf5d\gold\calibration_gold.json `
  --episode-file gold_sets\boundary_subgoal_review_50.json `
  --review-goal boundary_subgoal `
  --port 8787
```

Open:

```text
http://127.0.0.1:8787
```

For each episode:

1. Leave the score alone.
2. Check **Accept auto subtask boundaries** if the boundaries look good.
3. Check **Accept auto subgoal frames** if the subgoal frames look good.
4. Leave a box unchecked only if you genuinely disagree.
5. Click **Save review and next**.

The 50-episode queue has this calibrated score spread:

```text
{2: 6, 3: 8, 4: 10, 5: 26}
```

It includes 30 episodes where Kevin's score differed from Gemini and 20 where it did not.

## After Human Boundary/Subgoal Review

Run:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.goldset report `
  --snapshot snap_2026_05_11_1dde3edf5d `
  --gold-file bridgeengine_data\snapshots\snap_2026_05_11_1dde3edf5d\gold\calibration_gold.json
```

Target thresholds:

```text
quality within-one agreement: >= 0.85
keep/reject agreement:        >= 0.90
boundary temporal IoU:        >= 0.70
subgoal agreement:            >= 0.75
```

If the boundary/subgoal numbers pass, the annotation pipeline is reliable enough for the final controlled comparison.

## Perception Critical Path

The headline comparison cannot run until perceptive signals are real.

Check readiness:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.perceptive_status `
  --snapshot snap_2026_05_11_1dde3edf5d_human_calibrated `
  --require-real
```

Current state: not ready. The human-calibrated snapshot does not yet have perceptive mask/depth/track rows.

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
