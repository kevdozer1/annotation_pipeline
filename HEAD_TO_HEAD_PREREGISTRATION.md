# BridgeEngine Head-To-Head Preregistration

Last updated: 2026-06-06

## Headline Claim

Primary research question:

> On the same 100 BridgeData subset, do pi0.7-style structured annotations outperform the perceptive vision signals used in the previous LeWM experiment?

This is the controlled comparison that should close the internal research loop. Same data, same held-out split, same LeWM metric, same seed set. The only intended variable is the conditioning signal family.

## Frozen Data

Semantic source snapshot:

```text
snap_2026_05_11_1dde3edf5d
```

Score-calibrated benchmark snapshot:

```text
snap_2026_05_11_1dde3edf5d_human_calibrated
```

Current score calibration:

```text
reviewed clips: 100 / 100
score changes versus Gemini: 58 / 100
Gemini auto score counts: {2: 5, 4: 9, 5: 86}
Kevin calibrated score counts: {2: 6, 3: 15, 4: 30, 5: 49}
```

Subtask boundaries and subgoal selections looked good by visual inspection, but still need a measured reliability pass before the final comparison is treated as hardened.

## Human Reliability Gate Before Final Run

Run a focused 50-episode boundary/subgoal review before the final head-to-head. The score labels are already reviewed; the next pass should only answer:

- Are the automatic subtask boundaries acceptable?
- Are the automatic subgoal frames acceptable?

Target thresholds before final claim:

| Metric | Target |
|---|---:|
| quality within-one agreement | >= 0.85 |
| keep/reject agreement | >= 0.90 |
| subtask-boundary temporal IoU | >= 0.70 |
| subgoal-selection agreement | >= 0.75 |

Exact 1-5 score agreement is reported but is not the primary reliability threshold, because curation scores are ordinal and subjective.

## Perceptive-Signal Gate Before Final Run

The head-to-head must not use synthetic perceptive fallbacks.

Run:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.perceptive_status --snapshot snap_2026_05_11_1dde3edf5d_human_calibrated --require-real
```

This must pass before the perceptive comparison is valid. Each perceptive labeler needs one real payload per episode:

- `perceptive_masks`
- `perceptive_depth`
- `perceptive_tracks`

If the command reports synthetic adapters, missing payloads, or missing label rows, fix the extractor path first. Do not run or report the head-to-head on synthetic perceptive labels.

## Metric

Primary metric:

```text
held-out latent MSE from the real LeWM frozen-adapter evaluation
```

The held-out split is fixed and disjoint. At 100 episodes, the calibrated split has held-out quality distribution:

```text
{2: 1, 3: 1, 4: 1, 5: 7}
```

## Seeds

Final run seed set:

```text
0, 1, 2, 3, 4
```

Three seeds is the minimum acceptable final run. Five seeds is preferred because the current two-seed deltas are in the single-digit-percent range.

## Families

Primary families:

1. `baseline`: task instruction only.
2. `pi07_full`: rich text + metadata + subgoal, equivalent to current `rich_text_metadata_subgoal`.
3. `perceptive_all`: masks + depth + tracks.

Secondary families:

1. `rich_text`
2. `rich_text_metadata`
3. `perceptive_masks`
4. `perceptive_depth`
5. `perceptive_tracks`

## Injection Decision

Use a unified conditioning interface for the main claim.

Reason: the claim is about signal content, not about each method's native architecture. The base LeWM checkpoint, frozen modules, trainable adapter budget, held-out split, optimizer, and epoch count should stay fixed. The conditioning interface should change only in the information it receives.

Document separately if a secondary "methods as practiced" comparison is ever run with native injection paths.

## Stop Rules

- Do not change the score rubric after this preregistration unless the final result is explicitly marked as post-rubric.
- Do not change the held-out split after seeing final metrics.
- Do not report perceptive results if `perceptive_status --require-real` fails.
- Do not increase N beyond 100 without a new data and cost gate.
- Do not claim general video or human-motion validation; the current validation class is robot manipulation video.

## Current Best Pre-Final Result

Score-calibrated 100-episode scale curve, two seeds:

| N | baseline | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|---:|
| 25 | 0.042079 | 0.041079 | 0.039682 | 0.038022 |
| 50 | 0.031014 | 0.030213 | 0.028289 | 0.027482 |
| 100 | 0.022522 | 0.023123 | 0.022831 | 0.020807 |

At N=100, `rich_text_metadata_subgoal` is 7.61% lower mean latent MSE than baseline. This is the result to beat or falsify with the perception head-to-head and more seeds.
